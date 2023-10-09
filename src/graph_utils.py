import torch
from torch_geometric.data import Data, HeteroData
from sklearn.model_selection import StratifiedKFold

def shift_edge_indices(g: HeteroData) -> HeteroData:
    g = g.clone()
    total = 0
    for n, s in g.node_items():
        s.shift = total
        total += s.num_nodes
        s.end = total

    for (src, _, dst), s in g.edge_items():
        s.edge_index[0] += g[src].shift
        s.edge_index[1] += g[dst].shift
    
    return g

def unshift_edge_indices(g: HeteroData) -> HeteroData:
    g = g.clone()

    for (src, _, dst), s in g.edge_items():
        s.edge_index[0] -= g[src].shift
        s.edge_index[1] -= g[dst].shift

    for n, s in g.node_items():
        del s.shift

    return g

def ensure_homogeneous(*args, **kwargs):
    def _apply(g):
        if isinstance(g, HeteroData):
            hg = g.to_homogeneous(**kwargs)
            # Removing final na
            if hasattr(hg, 'edge_label'):
                assert hg.edge_label[hg.edge_label_index.size(1):].isnan().all()
                hg.edge_label = hg.edge_label[:hg.edge_label_index.size(1)].bool()
            # Shifting negative samples
            if hasattr(hg, 'negative_samples'):
                msk = hg.negative_samples != -1
                hg.negative_samples[msk] += g['voter'].num_nodes
            return hg
        else:
            return g

    ret = tuple(_apply(g) for g in args)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret

def get_train_val_test(g: Data | HeteroData, train_ratio=0.75):
    t = ft.partial(PyG.transforms.RandomLinkSplit, 
        is_undirected=True,
        num_val=1-train_ratio,
        # split_labels=True,
        add_negative_train_samples=True,
        num_test=0,
    )
    
    if isinstance(g, HeteroData):
        t = t(
            edge_types=[g.edge_types[0]],
            rev_edge_types=[g.edge_types[1]] if len(g.edge_types) > 1 else None,
        )
    elif isinstance(g, Data):
        t = t()
            
    return t(g)

def k_fold(g: Data | HeteroData, folds, edge_types=None, **kwargs):
    skf = StratifiedKFold(folds, shuffle=True, **kwargs)

    folds = []

    # Stratify by voter
    if edge_types is None:
        edge_types = list(zip(g.edge_types[::2], g.edge_types[1::2]))

    gtrain_dict = {}
    gval_dict = {}
    for edge_type, rev_edge_type in edge_types:    
        edge_index = g[edge_type].edge_index
        for train_idx, val_idx in skf.split(torch.zeros(edge_index.size(1)), edge_index[0]):
            gtrain_dict[rev_edge_type] = gtrain_dict[edge_type] = torch.tensor(train_idx)
            gval_dict[rev_edge_type] = gval_dict[edge_type] = torch.tensor(val_idx)

    gtrain = g.edge_subgraph(gtrain_dict)
    gval = g.edge_subgraph(gval_dict)

    assert gtrain.is_undirected()
    assert gval.is_undirected()

    folds.append((gtrain, gval))

    return folds
