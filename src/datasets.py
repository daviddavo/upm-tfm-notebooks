import torch
from torch_geometric.data import InMemoryDataset, HeteroData

class Daostack(InMemoryDataset):
    """ Creates a heterogeneus graph with two kinds of nodes: voters and proposals """
    def __init__(self, root: str, min_vpu=6, allowed_daos=None):
        self._min_vpu = min_vpu
        self._allowed_daos = allowed_daos
        
        super().__init__(root)

        self.data = torch.load(self.processed_paths[0])

    def process(self):
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        if self._allowed_daos:
            dfd = pd.read_csv(self.raw_paths[1]).set_index('id')
            allowed_dao_ids = set(dfd[dfd['name'].isin(self._allowed_daos)].index)
            df = df[df['dao'].isin(allowed_dao_ids)]
            assert not df.empty, "Dataframe is empty"
            
        if self._min_vpu:
            vpu = df.groupby('voter').size()
            allowed_voters = vpu[vpu >= self._min_vpu].index
            df = df[df['voter'].isin(allowed_voters)]
        
        data = HeteroData()
        node_types = ['voter', 'proposal']
        for nt in node_types:
            df[nt] = df[nt].astype('category')
            data[nt].num_nodes = df[nt].nunique()

        u_t = torch.LongTensor(df['voter'].cat.codes)
        p_t = torch.LongTensor(df['proposal'].cat.codes)

        data['voter', 'votes', 'proposal']['edge_index'] = torch.stack([u_t, p_t])
        data['proposal', 'voted', 'voter']['edge_index'] = torch.stack([p_t, u_t])

        data.validate()
        assert not data.is_directed(), "The created graph should not be directed"

        torch.save(data, self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.root / 'daostack'

    @property
    def processed_dir(self) -> str:
        return self.raw_dir / 'processed'

    @property
    def raw_file_names(self) -> str:
        return ['votes.csv', 'daos.csv']
    
    @property
    def processed_file_names(self) -> str:
        return f"daostack_votes_{self._min_vpu}_{'-'.join(self._allowed_daos)}.pt"
    