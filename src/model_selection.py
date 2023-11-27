from typing import Optional

import pandas as pd

def _getTrainTestFromTime(train_end_t, test_end_t, df, timestamp_col, remove_not_in_train_col: Optional[str] = None):
    train = df[df[timestamp_col] <= train_end_t]
    test = df[ (train_end_t < df[timestamp_col]) & (df[timestamp_col] < test_end_t) ]

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

def timeFreqSplitCurrent(dfv: pd.DataFrame, freq: str, dfp: pd.DataFrame, return_open: bool = False, remove_not_in_train_col = None):
    times = pd.date_range(dfv['timestamp'].min(), dfv['timestamp'].max(), freq=freq)
    for train_end, test_end in zip(times, times[1:]):
        train, test = _getTrainTestFromTime(train_end, test_end, dfv, 'timestamp', remove_not_in_train_col=remove_not_in_train_col)
        
        open_proposals = current_proposals(dfp, train_end)
        test_filtered = test[test['itemID'].isin(open_proposals)]

        if return_open:
            yield train, test_filtered, train_end, open_proposals 
        else:
            yield train, test_filtered
