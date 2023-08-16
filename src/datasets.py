import torch

import torch_geometric as PyG
from torch_geometric.data import InMemoryDataset, HeteroData

class Daostack(InMemoryDataset):
    """ Creates a heterogeneus graph with two kinds of nodes: voters and proposals """
    def __init__(self, root: str, min_vpu=6, allowed_daos=None):
        self._min_vpu = min_vpu
        self._allowed_daos = allowed_daos
        
        super().__init__(root)

        self.data = torch.load(self.processed_paths[0])
        from pathlib import Path
        Path(self.processed_paths[0]).unlink()

    def download(self):
        import kaggle
        kaggle.api.dataset_download_cli('daviddavo/dao-analyzer', path=self.raw_dir, unzip=True)

    def process(self):
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        if self._allowed_daos:
            dfd = pd.read_csv(self.raw_paths[1]).set_index('id')
            allowed_dao_ids = set(dfd[dfd['name'].isin(self._allowed_daos)].index)
            df = df[df['dao'].isin(allowed_dao_ids)]
            assert not df.empty, "Dataframe is empty"
            
        if self._min_vpu:
            vpu = df.groupby('voter').size()
            allowed_voters = vpu[vpu >= self._min_vpu].index
            df = df[df['voter'].isin(allowed_voters)]
        
        data = HeteroData()
        node_types = ['voter', 'proposal']
        for nt in node_types:
            df[nt] = df[nt].astype('category')
            data[nt].num_nodes = df[nt].nunique()

        u_t = torch.LongTensor(df['voter'].cat.codes)
        p_t = torch.LongTensor(df['proposal'].cat.codes)

        data['voter', 'votes', 'proposal']['edge_index'] = torch.stack([u_t, p_t])
        data['proposal', 'voted', 'voter']['edge_index'] = torch.stack([p_t, u_t])

        data.validate()
        assert not data.is_directed(), "The created graph should not be directed"
        assert PyG.utils.structured_negative_sampling_feasible(data['voter', 'votes', 'proposal'].edge_index)

        torch.save(data, self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return ['daostack/votes.csv', 'daostack/daos.csv']
    
    @property
    def processed_file_names(self) -> str:
        allowed_daos_str = '-'.join(self._allowed_daos) if self._allowed_daos else 'all'
        return f"daostack_votes_{self._min_vpu}_{allowed_daos_str}.pt"

class DAOCensus(InMemoryDataset):
    def __init__(self, root: str, name: str, platform: str=None):
        self._name = name
        self._platform = platform
        
        super().__init__(root)

        self.data = torch.load(self.processed_paths[0])

    def download(self):
        import kaggle
        kaggle.api.dataset_download_cli('oiudoiajd/daos-census', path=self.raw_dir, unzip=True)

    def process(self):
        import pandas as pd
        import duckdb

        db = duckdb.connect(database=':memory:', read_only=False)
        db.execute("CREATE VIEW deployments AS SELECT * FROM parquet_scan('{}')".format(self.raw_paths[0]))
        db.execute("CREATE VIEW votes AS SELECT * FROM parquet_scan('{}')".format(self.raw_paths[1]))
        db.execute("CREATE VIEW proposals AS SELECT * FROM parquet_scan('{}')".format(self.raw_paths[2]))

        cond = f"name='{self._name}'"
        if self._platform:
            cond += f" AND platform='{self._platform}'"

        dfv = db.execute(f"""
        SELECT platform, name, votes.*
        FROM deployments
        LEFT JOIN votes ON (deployments.id = votes.deployment_id)
        WHERE {cond}
        """).fetchdf().rename(columns=lambda x: x.replace('_id', ''))

        dfp = db.execute(f"""
        SELECT platform, name, proposals.*
        FROM deployments
        LEFT JOIN proposals ON (deployments.id = proposals.deployment_id)
        WHERE {cond}
        """).fetchdf().rename(columns=lambda x: x.replace('_id', ''))

        data = HeteroData()
        t = {}

        # display(dfp)
        
        dfv['voter'] = dfv['voter'].str.lower()
        dfp['author'] = dfp['author'].str.lower()

        prop_dtype = pd.api.types.CategoricalDtype(categories=dfp['id'])
        user_dtype = pd.api.types.CategoricalDtype(categories=set(dfv['voter']).union(dfp['author']))

        # voter <-> proposal (dfv)
        dfv['voter'] = dfv['voter'].astype(user_dtype)
        dfv['proposal'] = dfv['proposal'].astype(prop_dtype)

        data['user'].num_nodes = user_dtype.categories.size
        data['user'].voters = dfv['voter'].cat.codes.unique()
        data['proposal'].num_nodes = prop_dtype.categories.size
        
        t = torch.stack([
            torch.LongTensor(dfv['voter'].cat.codes),
            torch.LongTensor(dfv['proposal'].cat.codes)
        ])

        data['user', 'vote', 'proposal'].edge_index = t
        data['proposal', 'vote', 'user'].edge_index = t[(1,0), :]

        # author <-> proposal (dfp)
        dfp['author'] = dfp['author'].astype(user_dtype)
        dfp['id'] = dfp['id'].astype(prop_dtype)
        t = torch.stack([
            torch.LongTensor(dfp['author'].cat.codes),
            torch.LongTensor(dfp['id'].cat.codes),
        ])

        data['user'].authors = dfp['author'].cat.codes.unique()
        data['user', 'creates', 'proposal'].edge_index = t
        data['proposal', 'creates', 'user'].edge_index = t[(1,0), :]

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
        return f"daostack_votes_{self._name}{pfrm_str}.pt"
    