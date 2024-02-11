from pathlib import Path

import pandas as pd

from . import daocensus
from .. import utils

def download(path):
    import kaggle
    daocensus.download(path)
    kaggle.api.dataset_download_cli('daviddavo/daos-census-proposals-text', path=path, unzip=True)
    (path / 'proposals.parquet').rename(path / 'proposals-original.parquet')
    (path / 'proposals-text.parquet').rename(path / 'proposals.parquet')

def load_pandas_df(raw_path: str, *args, remove_new_proposals=True, remove_ghost_votes=True, remove_unused_categories=True, **kwargs):
    dfv, dfp = daocensus.load_pandas_df(raw_path, *args, **kwargs)
    
    if remove_new_proposals:
        raw_path = Path(raw_path)
        dfp_original = pd.read_parquet(raw_path/'proposals-original.parquet')

        dfp = dfp[dfp['id'].isin(dfp_original['id'])].copy()

    if remove_ghost_votes:
        msk = dfv['proposal'].isin(dfp['id'])
        if msk.any():
            print(f"Warning, removing {msk.sum()} votes without proposal")
        
        dfv = dfv[msk]

    if remove_unused_categories:
        dfv = utils.remove_unused_categories(dfv)
        dfp = utils.remove_unused_categories(dfp)

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
