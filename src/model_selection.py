import sys
import pickle
import datetime as dt

from typing import Optional
from collections import namedtuple
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

DEFAULT_CHECKPOINT_EVERY = dt.timedelta(seconds=30)

Fold = namedtuple('Fold', ['train', 'test', 'end', 'open_proposals'])

def _getTrainTestFromTime(train_end_t, test_end_t, df, timestamp_col, remove_not_in_train_col: Optional[str] = None):
    train = df[df[timestamp_col] <= train_end_t]
    test = df[ (train_end_t < df[timestamp_col]) & (df[timestamp_col] <= test_end_t) ]

    if remove_not_in_train_col is not None:
        msk = test[remove_not_in_train_col].isin(set(train[remove_not_in_train_col]))
        test = test[msk]

    return train, test

def timeIntervalSplit(df: pd.DataFrame, splits: int, timestamp_col: str = 'timestamp', skip: int = 0, remove_not_in_train_col: Optional[str] = None):
    total_time_diff = df[timestamp_col].max() - df[timestamp_col].min()
    k_time_diff = total_time_diff / (splits+1)

    acc_time = df[timestamp_col].min() + (1+skip)*k_time_diff
    for i in range(splits - skip):
        end_time = acc_time + k_time_diff

        train,test = _getTrainTestFromTime(acc_time, end_time, df, timestamp_col, remove_not_in_train_col = remove_not_in_train_col)
        
        acc_time = end_time
        yield train, test

def current_proposals(dfp, t):
    """
    Open proposals: The ones that started before _t_, but are still open (close after _t_)
    """
    props = dfp[(dfp['start'] < t) & (t <= dfp['end']) ]
    if 'id' in props.columns:
        return props['id']
    else:
        return props.index

def filter_current(df, dfp, t):
    return df[df['itemID'].isin(current_proposals(dfp, t))]

def timeIntervalSplitCurrent(dfv: pd.DataFrame, splits: int, dfp: pd.DataFrame, return_open: bool = False, **kwargs):
    for train, test in timeIntervalSplit(dfv, splits, **kwargs):
        t = train.timestamp.max()
        open_proposals = current_proposals(dfp, t)

        test_filtered = test[test['itemID'].isin(open_proposals)]

        if return_open:
            yield train, test_filtered, t, open_proposals
        else:
            yield train, test_filtered

def timeFreqSplitCurrent(
    dfv: pd.DataFrame, freq: str, dfp: pd.DataFrame, return_open: bool = False, 
    remove_not_in_train_col = None, normalize = True, inclusive: str = "left",
    item_col='itemID', user_col='userID', timestamp_col='timestamp',
    ) -> Fold:
    times = pd.date_range(dfv[timestamp_col].min(), dfv[timestamp_col].max(), freq=freq, normalize=normalize, inclusive=inclusive)
    test_end = dfv['timestamp'].max()

    assert return_open, "Return open is deprecated and default will be True in the future"
        

    # for train_end, test_end in zip(times, times[1:]):
    for train_end in times:
        train, test = _getTrainTestFromTime(train_end, test_end, dfv, timestamp_col, remove_not_in_train_col=remove_not_in_train_col)
        all_props = np.union1d(train[item_col], test[item_col])
        
        open_proposals = np.intersect1d(all_props, current_proposals(dfp, train_end))
        test_filtered = test[test[item_col].isin(open_proposals)]

        yield Fold(train, test_filtered, train_end, np.array(open_proposals))

def save_progress(data, fname, keys):
    fname.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fname, 'wb') as f:
        pickle.dump({
            'data': data,
            'keys': keys,
        }, f)


def load_progress(fname: Path, keys):
    if not fname.exists():
        return {}
        
    with open(fname, 'rb') as f:
        p = pickle.load(f)
        assert list(p['keys']) == list(keys), f'Not the same keys! {p["keys"]} != {keys}'
        return p['data']

def explore_hparams(func, param_grid, fname, checkpoint_every=DEFAULT_CHECKPOINT_EVERY):
    if not param_grid:
        return {}

    keys = list(sorted(param_grid[0].keys()))
    results = load_progress(fname, keys)
    if results:
        print("Restored checkpoint from", fname, "with", len(results), "results")
    
    try:
        next_checkpoint = dt.datetime.now() + checkpoint_every
        for p in tqdm(param_grid):
            p = dict(sorted(p.items()))
            assert list(p.keys()) == keys, f'Changing keys in the hparams is not supported {p.keys()} != {keys}'
            k = tuple(p.values())
            if k in results:
                continue
            
            results[k] = func(**p)

            if dt.datetime.now() > next_checkpoint:
                next_checkpoint = dt.datetime.now() + checkpoint_every
                save_progress(results, fname, keys)
                print(f"[{dt.datetime.now().isoformat()}] Saving checkpoint at {fname}")
    except KeyboardInterrupt:
        print("Interrupt received, returning", file=sys.stderr)

    save_progress(results, fname, keys)

    # Convert the results to records format
    asked = { tuple(v for _,v in sorted(p.items())) for p in param_grid }
    return [ dict(zip(keys,v)) | r for v,r in results.items() if v in asked ]
