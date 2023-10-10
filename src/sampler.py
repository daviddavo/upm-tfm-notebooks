from typing import Union

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.sampler import (
    NeighborSampler, NodeSamplerInput, SamplerOutput, HeteroSamplerOutput,
)
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import (
    NodeType, EdgeType, InputNodes, OptTensor, InputEdges
)

DO_ASSERTS: bool = False

class BPRSampler(NeighborSampler):
    r"""An implementation of an in-memory user-wise sampler with
    negative samples for use in :class:`BPRLoader`
    """
    def __init__(
        self,
        data: HeteroData,
        input_edges: InputEdges,
        **kwargs,
    ):
        self.neg_sampling: NegativeSampling = kwargs.pop('neg_sampling', None)
        self.input_edges = input_edges

        kwargs['num_neighbors']={v:[0] for v in data.edge_types} | { self.input_edges_rev:[1] }
        
        super().__init__(data, **kwargs)

    @property
    def input_edges_rev(self):
        return (self.input_edges[2], 'rev_'+self.input_edges[1], self.input_edges[0])

    def sample_from_nodes(
        self, 
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        out = super().sample_from_nodes(inputs)

        src_name = input_type = inputs.input_type
        dst_name = self.input_edges[-1]
        num_pos = inputs.node.numel()
        
        assert input_type == self.input_edges[0]

        if self.neg_sampling is not None:
            if input_type is None:
                raise NotImplementedError('Only heterogeneous graphs are currently supported')
            if self.disjoint:
                raise NotImplementedError('Disjoint graphs are not supported')

            if DO_ASSERTS:
                prev_col_ = out.col[self.input_edges].clone()
                prev_node_ = out.node[dst_name].clone()

            src_index = out.row[self.input_edges]
            dst_pos_index = out.col[self.input_edges]

            dst_neg, _ = neg_sample(
                src=out.node[src_name],
                dst_nodes=self.num_nodes[dst_name],
            )

            original = out.node[dst_name][out.col[self.input_edges]]
    
            dst = torch.cat((original, dst_neg), dim=0)
            # Re-index the proposals taking into account the new ones
            dst, inv_dst = dst.unique(return_inverse=True)

            out.node[dst_name] = dst
            
            if self.neg_sampling.is_triplet():
                dst_pos_index = inv_dst[:num_pos]
                dst_neg_index = inv_dst[num_pos:]
    
                # note: row = src; col = dst
                out.col[self.input_edges] = out.row[self.input_edges_rev] = dst_pos_index
    
                if DO_ASSERTS:
                    assert torch.equal(prev_node_[prev_col_], dst[dst_pos_index])
                    
                out.metadata = (*out.metadata, src_index, dst_pos_index, dst_neg_index)
            else:
                assert self.neg_sampling.is_binary(), "Neg sampling method should be triplet or binary"
                edge_label_index = torch.stack([src_index.repeat(2),inv_dst])
                edge_label = torch.cat((torch.full((num_pos,), True), torch.full((num_pos,), False)))

                if DO_ASSERTS:
                    assert torch.equal(prev_node_[prev_col_], out.node[dst_name][edge_label_index[1, edge_label]])

                out.metadata = (*out.metadata, edge_label_index, edge_label)
                
        return out

# True negative sampling is really expensive
# and our model assumes that the preferences
# are implicit (not explicit), so it should be
# fine to include false positives
def _check_dst(src, dst, coo_gt):
    # [N, 2] dimensions
    edge_index = torch.stack([src,dst], dim=1)

    # Create a custom dtype for the records (tuple) view
    base_dtype = src.numpy().dtype
    coo_dtype = np.dtype([('src', base_dtype), ('dst', base_dtype)])

    # Create the temporal ndarrays of tuples to compare
    tmp = edge_index.numpy().view(dtype=coo_dtype).reshape((-1,))
    aux_ei = coo_gt.view(dtype=coo_dtype).reshape((-1,))

    # Convert to tensor again
    msk = torch.from_numpy(np.isin(tmp, aux_ei))

    return msk

def neg_sample(src: torch.Tensor, dst_nodes: int, gt_edge_index: torch.Tensor = None, max_retries: int = 2) -> torch.Tensor:
    """Returns a negative dst index for every src, avoiding items in gt_edge_index
    Made with Hererogeneous Graphs in mind
    """
    dst = torch.randint(dst_nodes, src.size())
    # dst = torch.full(src.size(), 220)
    remaining = None

    # Only remove false negatives if the true positives are provided
    if gt_edge_index is not None:
        assert gt_edge_index.size()[0] == 2
        assert src.dtype == gt_edge_index.dtype, 'source and edge_index should have same type'

        coo_gt = gt_edge_index.numpy().T.copy()
        msk = _check_dst(src, dst, coo_gt)

        i = 0
        while msk.any() and i < max_retries:
            rest = msk.nonzero(as_tuple=False).view(-1)

            # Get new random values for the false negatives
            dst[rest] = torch.randint(dst_nodes, (rest.numel(),))
            msk = _check_dst(src, dst, coo_gt)
            
            i += 1

        # If there is any, return the mask object
        # eoc return False
        remaining = msk.any().item() and msk

    return dst, remaining
