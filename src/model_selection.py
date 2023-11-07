import pandas as pd

def timeIntervalSplit(df: pd.DataFrame, splits: int, timestamp_col: str = 'timestamp', skip: int = 0, remove_not_in_train: str = None):
    total_time_diff = df[timestamp_col].max() - df[timestamp_col].min()
    k_time_diff = total_time_diff / (splits+1)

    acc_time = df[timestamp_col].min() + (1+skip)*k_time_diff
    for i in range(splits - skip):
        end_time = acc_time + k_time_diff
        
        train = df[df[timestamp_col] <= acc_time]
        test = df[ (acc_time < df[timestamp_col]) & (df[timestamp_col] < end_time) ]

        if remove_not_in_train is not None:
            msk = test[remove_not_in_train].isin(set(train[remove_not_in_train]))
            test = test[msk]
        
        acc_time = end_time
        yield train, test