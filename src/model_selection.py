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

def filter_current(df, dfp, t):
    current_proposals = dfp[(dfp['start'] < t) & (t < dfp['end']) ]
    return df[df['itemID'].isin(current_proposals['id'])]

def timeIntervalSplitCurrent(dfv: pd.DataFrame, splits: int, dfp: pd.DataFrame, **kwargs):
    for train, test in timeIntervalSplit(dfv, splits, **kwargs):
        t = train.timestamp.max()
        yield train, filter_current(test, dfp, t)
