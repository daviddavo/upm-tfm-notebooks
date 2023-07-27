import pandas as pd

from torch_geometric.data import InMemoryDataset, HeteroData

class Daostack(InMemoryDataSet):
    """ Creates a heterogeneus graph with two kinds of nodes: voters and proposals """
    
    def __init__(self, root: str, min_vpu=6, allowed_daos=None):
        super().__init__(root)
        self._min_vpu = min_vpu
        self._allowd_daos = allowed_daos
        self.load(self.processed_paths[0], data_cls=HeteroData)
    