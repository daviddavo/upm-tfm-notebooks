from typing import Union, List, Dict, Tuple, Optional, Callable

import torch_geometric as PyG
from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import (
    BaseSampler, HeteroSamplerOutput, NodeSamplerInput, SamplerOutput
)
from torch_geometric.sampler.base import SubgraphType, NegativeSampling
from torch_geometric.loader import NodeLoader
from torch_geometric.typing import (
    NodeType, InputEdges, InputNodes, OptTensor,
)

from .sampler import BPRSampler

class BPRLoader(NodeLoader):
    def __init__(
        self,
        data: HeteroData,
        # num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_edges: InputEdges,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        bpr_sampler: Optional[BPRSampler] = None,
        neg_sampling: Optional[NegativeSampling] = 'triplet',
        transform_global: bool = False,
        **kwargs,
    ):
        if isinstance(neg_sampling, str):
            neg_sampling = NegativeSampling.cast(neg_sampling)
        
        if bpr_sampler is None:
            bpr_sampler = BPRSampler(
                data,
                input_edges,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                neg_sampling=neg_sampling,
            )

        super().__init__(
            data=data,
            node_sampler=bpr_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

        if transform_global:
            self.transform = self.transform_global

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        # Avoid calling transform on super().filter_fn()
        _aux = self.transform
        self.transform = None
        data = super().filter_fn(out)
        self.transform = _aux

        # TODO: make data[input_type].out_dst when there are negative samples like in LinkLoader
        # raise NotImplementedError()
        input_type = self.input_data.input_type
        input_edges = self.node_sampler.input_edges

        assert input_type == input_edges[0]
        # assert not self.disjoint
        
        # copied from torch_geometric.loader.LinkLoader
        if self.node_sampler.neg_sampling.is_triplet():
            if input_type == input_edges[0]:
                data[input_edges[-1]].src_index = out.metadata[2]
                data[input_edges[0]].dst_pos_index = out.metadata[3]
                data[input_edges[0]].dst_neg_index = out.metadata[4]

                for t in data.edge_types:
                    if t != input_edges and t != self.node_sampler.input_edges_rev:
                        assert data[t].edge_index.numel() == 0
                        del data[t] # it shouldn't have any meaningful info
        elif self.node_sampler.neg_sampling.is_binary():
            if input_type == input_edges[0]:
                data[input_edges].edge_label_index = out.metadata[2]
                # The mask where 0 = neg and 1 = pos
                data[input_edges].edge_label = out.metadata[3]
        else:
            raise NotImplementedError('Only triplet and binary is currently supported')
        
        return data if not self.transform else self.transform(data)

    def transform_global(self, data: HeteroData) -> HeteroData:
        print(data)
        for e, es in data.edge_items():
            es.edge_index[0] = data[e[0]].n_id[es.edge_index[0]]
            es.edge_index[1] = data[e[-1]].n_id[es.edge_index[1]]

            if hasattr(es, 'edge_label_index'):
                es.edge_label_index[0] = data[e[0]].n_id[es.edge_label_index[0]]
                es.edge_label_index[1] = data[e[-1]].n_id[es.edge_label_index[1]]
    
        input_edges = self.node_sampler.input_edges

        if hasattr(data[input_edges[0]], 'dst_pos_index'):
            data[input_edges[0]].dst_pos_index = data[input_edges[-1]].n_id[data[input_edges[0]].dst_pos_index]
            data[input_edges[0]].dst_neg_index = data[input_edges[-1]].n_id[data[input_edges[0]].dst_neg_index]
            data[input_edges[-1]].src_index = data[input_edges[0]].n_id[data[input_edges[-1]].src_index]
        
        for n, ns in data.node_items():
            ns.num_nodes = self.data[n].num_nodes
            del ns.n_id
        
        return data
