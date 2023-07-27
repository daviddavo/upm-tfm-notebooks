from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path('./datawarehouse')
TESTING_DAO_NAMES = {'dxDAO', 'xDXdao'}

def get_df(name, data_dir = None):
    data_dir = data_dir or DEFAULT_PATH
    
    df = pd.read_csv(data_dir / "daostack" / f'{name}.csv', index_col='id')
    df = df.drop(columns=['Unnamed: 0'])

    time_cols = [col for col in df.columns if col.endswith('At')]
    df[time_cols] = df[time_cols].apply(pd.to_datetime, errors='coerce', unit='s', origin='unix')
    df['network'] = df['network'].astype('category')
    
    return df

def filter_df(df: pd.DataFrame, data_dir = None):
    dfd = get_df('daos', data_dir)
    TESTING_DAO_IDS = set(dfd[dfd['name'].isin(TESTING_DAO_NAMES)].index)
    return df[df['dao'].isin(TESTING_DAO_IDS)]
    