from typing import Optional

import pandas as pd

def timeIntervalSplit(df: pd.DataFrame, splits: int, timestamp_col: str = 'timestamp', skip: int = 0, remove_not_in_train_col: Optional[str] = None):
    total_time_diff = df[timestamp_col].max() - df[timestamp_col].min()
    k_time_diff = total_time_diff / (splits+1)

    acc_time = df[timestamp_col].min() + (1+skip)*k_time_diff
    for i in range(splits - skip):
        end_time = acc_time + k_time_diff
        
        train = df[df[timestamp_col] <= acc_time]
        test = df[ (acc_time < df[timestamp_col]) & (df[timestamp_col] < end_time) ]

        if remove_not_in_train_col is not None:
            msk = test[remove_not_in_train_col].isin(set(train[remove_not_in_train_col]))
            test = test[msk]
        
        acc_time = end_time
        yield train, test

def current_proposals(dfp, t):
    return dfp[(dfp['start'] < t) & (t < dfp['end']) ]['id']

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
