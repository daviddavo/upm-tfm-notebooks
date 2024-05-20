import itertools as it
from pathlib import Path

import pandas as pd
from recommenders.datasets.pandas_df_utils import filter_by

from src.paths import _gen_fname

BASELINE_BASE_PATH = Path('./data/baseline')

def baseline_mp(org_name: str, n_splits: int, base: Path=BASELINE_BASE_PATH, ext='csv') -> Path:
    base.mkdir(exist_ok=True)
    return base / f'mp-{org_name}-{n_splits}.{ext}'

def baseline_mp_freq(org_name: str, splits_freq: str, normalize: bool, base: Path = BASELINE_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('mp-freq', org_name, splits_freq, normalize, ext='parquet')

def perfect_mp_freq(org_name: str, splits_freq: str, normalize: bool, base: Path = BASELINE_BASE_PATH) -> Path:
    base.mkdir(exist_ok=True)
    return base / _gen_fname('perfect-freq', org_name, splits_freq, normalize, ext='parquet')


def getBaselineRecommendations(train: pd.DataFrame, users, proposals, k: int = 5, remove_train=True):
    train = train.sort_values('timestamp', ascending=False) # To make it replicable
    bestVotes = train[train['itemID'].isin(proposals)]['itemID'].value_counts()
    # bestVotes = bestVotes[bestVotes.index.isin(proposals)]

    df = pd.DataFrame(it.product(users, bestVotes.index), columns=['userID', 'itemID'])

    # Avoid recommending already voted proposals (they wont be in the test set)
    if remove_train:
        df = filter_by(df, train, ['userID', 'itemID'])
    
    df = df.groupby('userID').head(k).reset_index(drop=True)

    df['prediction'] = True
    return df

def write_metrics_baseline(bdf: pd.DataFrame, org_name, splits_freq, normalize):
    assert bdf.index.name == 't'
    bdf.to_parquet(baseline_mp_freq(org_name, splits_freq, normalize)) 

def write_metrics_perfect(pdf: pd.DataFrame, org_name, splits_freq, normalize):
    assert pdf.index.name == 't'
    pdf.to_parquet(perfect_mp_freq(org_name, splits_freq, normalize))


def _read_metrics(path, cutoff_date = ...):
    if cutoff_date == ...:
        raise NotImplementedError("Not using a cutoff date is not yet implemented")
    
    df = pd.read_parquet(path)
    if cutoff_date:
        df = df[df.index <= cutoff_date]
    
    return df

def read_metrics_baseline(org_name, splits_freq, normalize, *, cutoff_date = ...):
    return _read_metrics(baseline_mp_freq(org_name, splits_freq, normalize), cutoff_date=cutoff_date)    
    
def read_metrics_perfect(org_name, splits_freq, normalize, *, cutoff_date = ...):
    return _read_metrics(perfect_mp_freq(org_name, splits_freq, normalize) , cutoff_date=cutoff_date)
    