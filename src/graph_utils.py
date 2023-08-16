from torch_geometric.data import Data, HeteroData

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
