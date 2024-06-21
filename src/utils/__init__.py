import time
import itertools as it
import pandas as pd
import numpy as np

def remove_unused_categories(df):
    for c in df.select_dtypes(include=['category']):
        df[c] = df[c].cat.remove_unused_categories()
    return df

class Timer:
    def __init__(self, get_time = time.perf_counter, print=False):
        self.get_time = get_time
        self.print = print
    
    def __enter__(self):
        self.start = self.get_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = self.get_time() - self.start
        if self.print:
            print(self.time)

def testHParamsLenskit(algo, f, k_recommendations: list[int], metrics_f, window_size=None, ):
    # Get and filter train data
    train = f.train
    
    if window_size:
        offset = pd.tseries.frequencies.to_offset(window_size)
        train = train[train['timestamp'] > (f.end - offset)]

    with Timer() as t_train:
        algo.fit(train)

    # TODO: For each user, make the recommendations
    # and then generate a microsoft-like dataframe
    with Timer() as t_rec:
        users = set(f.test['user'].unique()).intersection(train['user'].unique())
        voted_props = train.groupby('user')['item'].unique()
        def _recu(u):
            # Remove proposals the user voted in
            ps = np.setdiff1d(f.open_proposals, voted_props.loc[u])
            # TODO: WHY DOES IT RETURN SO MANY NAs?
            x = (algo
                .predict_for_user(u, ps)
                .reset_index()
                .rename(columns={'index':'item', 0:'prediction'})
                # .dropna()
                .fillna(0.00)
                .assign(user=u)[['user', 'item', 'prediction']]
            )
            return x
    
        # TODO: Use lenskit.batch.recommend
        # https://lkpy.lenskit.org/en/stable/batch.html#recommendation
        recs = pd.concat(map(_recu, users))

    metrics = { 
        'time_train': t_train.time,
        'time_rec': t_rec.time,
        'open_proposals': len(f.open_proposals),
        # 'train_open_proposals': len(np.intersect1d(f.open_proposals, train['item'].unique())),
        'min_recs': recs.groupby('user').size().min(),
        'avg_recs': recs.groupby('user').size().mean(),
    }
    with Timer() as t_eval:
        for (m, e), k_recs in it.product(metrics_f.items(), k_recommendations):
            metrics[f'{m}@{k_recs}'] = e(f.test, recs, k=k_recs, col_user='user', col_item='item')
    metrics['time_eval'] = t_eval.time

    return metrics
        