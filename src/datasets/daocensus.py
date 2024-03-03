from typing import Dict, List, Union, Optional

from pathlib import Path
import datetime as dt

import torch
import torch_geometric as PyG
from torch_geometric.data import InMemoryDataset, HeteroData

ORGS_DICT: Dict[str, List[str]] = {
    'dxDAO - xDXdao': ['dxDAO', 'xDXdao'],
    'Aave - Aavegotchi': ['Aave', 'Aavegotchi', 'AAVE'],
    'MetaCartel - MetaCartel Ventures': ['MetaCartel Ventures', 'MetaCartel xDai', 'MetaCartel DAO'],
}

def download(path):
    import kaggle
    kaggle.api.dataset_download_cli('oiudoiajd/daos-census', path=path, unzip=True)
    
def _list2sql(list):
    return "".join(["(", ", ".join(map("'{}'".format, list)), ")"])

def _gen_orgs_query(parquet):
    _casestr = "    WHEN name IN {caselst} THEN '{orgname}'"

    _cases = "\n".join(_casestr.format(
        orgname=orgname,
        caselst=_list2sql(caselst),
    ) for orgname, caselst in ORGS_DICT.items())
    
    return f"""
CREATE VIEW deployments AS
SELECT * EXCLUDE (name),
    name AS deployment_name,
    CASE 
{_cases}
    ELSE name
    END AS name
FROM parquet_scan('{parquet}')
    """

def load_pandas_df(
    raw_path: str,
    filter_name: str = None, 
    filter_platforms: Union[str, List[str]] = None, 
    min_vpu: int = 0,
    min_vpp: int = 1,
    use_org_names: bool = False,
    cutoff_date: Optional[dt.datetime] = False,
):
    import pandas as pd
    import duckdb

    raw_path = Path(raw_path)
    db = duckdb.connect(database=':memory:', read_only=False)
    if use_org_names:
        db.execute(_gen_orgs_query(raw_path / "deployments.parquet"))
    else:
        db.execute("CREATE VIEW deployments AS SELECT * FROM parquet_scan('{}')".format(raw_path / "deployments.parquet"))
    db.execute("CREATE VIEW votes AS SELECT * FROM parquet_scan('{}')".format(raw_path / "votes.parquet"))
    db.execute("CREATE VIEW proposals AS SELECT * FROM parquet_scan('{}')".format(raw_path / "proposals.parquet"))

    cond_dfv = f"name='{filter_name}'"
    if filter_platforms:
        if isinstance(filter_platforms, str):
            filter_platforms = [filter_platforms]

        cond_dfv += f" AND platform IN {_list2sql(filter_platforms)}"

    if cutoff_date:
        cond_dfv += f" AND date <= '{cutoff_date.isoformat()}'"

    cond_dfp = cond_dfv
    if min_vpp:
        cond_dfp += f" AND proposals.votes_count >= {min_vpp}"

    dfv = db.execute(q := f"""
    SELECT platform, name, votes.*
    FROM deployments
    LEFT JOIN votes ON (deployments.id = votes.deployment_id)
    WHERE {cond_dfv}
    """).fetchdf().rename(columns=lambda x: x.replace('_id', ''))

    dfp = db.execute(q := f"""
    SELECT platform, name, platform_deployment_id, proposals.*
    FROM deployments
    LEFT JOIN proposals ON (deployments.id = proposals.deployment_id)
    WHERE {cond_dfp}
    """).fetchdf().rename(columns=lambda x: x.replace('_id', ''))

    dfv['voter'] = dfv['voter'].str.lower()
    dfp['author'] = dfp['author'].str.lower()

    assert not any(pd.isna(dfv['voter'])), "There are NA voters"
    assert not any(pd.isna(dfp['author'])), "There are NA authors"

    if min_vpu:
        vpu = dfv.groupby('voter').size()
        allowed_voters = vpu[vpu >= min_vpu].index
        msk = dfv['voter'].isin(allowed_voters)
        dfv = dfv[msk].reset_index(drop=True)
        print(f"Removing {(~nvoters_removed).sum()} voters with less than {min_vpu} votes")

    prop_dtype = pd.api.types.CategoricalDtype(categories=dfp['id'])
    user_dtype = pd.api.types.CategoricalDtype(categories=set(dfv['voter']).union(dfp['author']))

    # voter <-> proposal (dfv)
    dfv['voter'] = dfv['voter'].astype(user_dtype)
    dfv['proposal'] = dfv['proposal'].astype(prop_dtype)

    # author <-> proposal (dfp)
    dfp['author'] = dfp['author'].astype(user_dtype)
    dfp['id'] = dfp['id'].astype(prop_dtype)

    return dfv, dfp

def get(
    root: str,
    filter_name: str = None, 
    filter_platform: str = None, 
    min_vpu: int = 0,
):
    root = Path(root)
    raw_dir = root/'raw'
    if not raw_dir.exists():
        print(f"Folder {raw_dir} not found, downloading")
        download(raw_dir)

    return load_pandas_df(raw_dir, filter_name, filter_platform, min_vpu)

class DAOCensus(InMemoryDataset):
    def __init__(self, root: str, name: str, platform: str=None, min_vpu=None):
        self._name = name
        self._platform = platform
        self._min_vpu = min_vpu if min_vpu is not None else 0
        
        super().__init__(root)

        assert self._min_vpu >= 0, 'min_vpu must be positive'
        self.data = torch.load(self.processed_paths[0])

    def download(self):
        download_daocensus(self.raw_dir)

    def process(self):
        import pandas as pd

        dfv, dfp = load_pandas_df(self.raw_dir, self._name, self._platform, self._min_vpu)

        data = HeteroData()
        t = {}

        data['user'].num_nodes = dfv['voter'].categories.size
        data['user'].voters = dfv['voter'].cat.codes.unique()
        data['proposal'].num_nodes = dfv['proposals'].categories.size
        
        t = torch.stack([
            torch.LongTensor(dfv['voter'].cat.codes),
            torch.LongTensor(dfv['proposal'].cat.codes)
        ])

        data['user', 'vote', 'proposal'].edge_index = t
        data['proposal', 'rev_vote', 'user'].edge_index = t[(1,0), :]

        t = torch.stack([
            torch.LongTensor(dfp['author'].cat.codes),
            torch.LongTensor(dfp['id'].cat.codes),
        ])

        data['user'].authors = dfp['author'].cat.codes.unique()
        data['user', 'creates', 'proposal'].edge_index = t
        data['proposal', 'rev_creates', 'user'].edge_index = t[(1,0), :]

        data.validate()
        assert not data.is_directed()
        assert not data.has_isolated_nodes()

        db.close()
        torch.save(data, self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["deployments.parquet", "votes.parquet", "proposals.parquet"]

    @property
    def processed_file_names(self) -> str:
        pfrm_str = f"_{self._platform}" if self._platform else ""
        return f"daostack_votes_{self._name}{pfrm_str}_{self._min_vpu}.pt"