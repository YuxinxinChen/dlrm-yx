import sys

import torch
from torch import nn

sys.path.append('/home/yuxin420/dlrm-yx/yx_modfs/build_sparse/lib.linux-x86_64-cpython-310')

class LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices, offsets):
        ctx.save_for_backward(weights, indices, offsets)
        import sparse_embedding.forward
        res = torch.ops.sparse_offset_forward.forward(weights, indices, offsets)
        return res 

import enum

class EmbeddingLocation(enum.Enum):
    DEVICE = 0
    HOST_MAPPED = 1

class UniformShardedEmbeddingBags(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim, managed=EmbeddingLocation.DEVICE):
        super(UniformShardedEmbeddingBags, self).__init__()
        # Whole tables (i.e. all rows for a table) are partitioned uniformly across devices
        assert managed in (EmbeddingLocation.DEVICE, EmbeddingLocation.HOST_MAPPED)
        embedding_data = torch.randn(num_embeddings, num_tables, embedding_dim, dtype=torch.float32)
        if managed == EmbeddingLocation.DEVICE:
            print("allocate Emb table on device: ", torch.cuda.current_device())
            embedding_data = embedding_data.to(torch.cuda.current_device())
        self.embedding_weights = nn.Parameter(embedding_data)

    def forward(self, sharded_sparse_features, sharded_offsets=None):
        return LookupFunction.apply(self.embedding_weights,
                                    sharded_sparse_features, sharded_offsets)

#class All2AllFunction(torch.autograd.Function):
#    @staticmethod
#    def forward(ctx, partitioned_embeddings):
#        (B, T, D) = partitioned_embeddings.size()
#        assert B % hvd.size() == 0
#        return sparse_embedding_cuda.forward_all2all_nccl(
#            partitioned_embeddings)