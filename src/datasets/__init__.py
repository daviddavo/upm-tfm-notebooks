def to_microsoft(dfv):
    df = dfv[['voter', 'proposal', 'date']].rename(columns={
        'voter': 'userID',
        'proposal': 'itemID',
        'date': 'timestamp',
    })
    df['userID'] = df['userID'].astype('str')
    df['itemID'] = df['itemID'].astype('str')
    df['rating'] = 1
    
    return df

try:
    import torch
    import torch_geometric as PyG
    from torch_geometric.data import InMemoryDataset, HeteroData
    
    from .daostack import Daostack
    from .daocensus import DAOCensus
except ImportError:
    # TODO: Ignore the importerror and just don't use them
    raise
