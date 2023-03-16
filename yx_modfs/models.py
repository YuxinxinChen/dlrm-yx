import torch
from torch import nn
import torch.distributed as dist

from sparse_embedding_cuda_ops import UniformShardedEmbeddingBags

class DenseArch(nn.Module):
    def __init__(self, dense_features_dim, embedding_dim) -> None:
        super().__init__()
        self._0 = nn.Parameter(
            torch.empty([512, dense_features_dim]).uniform_(
                -0.027939938008785248, 0.027939938008785248
            )
        )  # UniformFill
        self._1 = nn.Parameter(
            torch.empty([512]).uniform_(-0.027939938008785248, 0.027939938008785248)
        )  # UniformFill
        self._2 = nn.Parameter(
            torch.empty([embedding_dim, 512]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._3 = nn.Parameter(
            torch.empty([embedding_dim]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
    def forward(self, float_features):
        """
        Returns a tuple of dense projection and dense embedding
        """
        _218 = torch.addmm(self._1, float_features, self._0.t())  # FC
        _219 = torch.relu(_218)  # Relu
        _220 = torch.addmm(self._3, _219, self._2.t())  # FC

        return (_219, _220)
    
class OverArch(nn.Module):
    def __init__(self, num_tables) -> None:
        super().__init__()
        self._201 = nn.Parameter(
            torch.empty([287, 15 * (num_tables + 1) + 512]).uniform_(
                -0.016946718096733093, 0.016946718096733093
            )
        )  # UniformFill
        self._202 = nn.Parameter(
            torch.empty([287]).uniform_(-0.016946718096733093, 0.016946718096733093)
        )  # UniformFill
        self._203 = nn.Parameter(
            torch.empty([225, 15 * (num_tables + 1) + 512]).uniform_(
                -0.016946718096733093, 0.016946718096733093
            )
        )  # UniformFill
        self._204 = nn.Parameter(
            torch.empty([225]).uniform_(-0.016946718096733093, 0.016946718096733093)
        )  # UniformFill
        self._205 = nn.Parameter(
            torch.empty([256, 512]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._206 = nn.Parameter(
            torch.empty([256]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._207 = nn.Parameter(
            torch.empty([512, 256]).uniform_(-0.0625, 0.0625)
        )  # UniformFill
        self._208 = nn.Parameter(
            torch.empty([512]).uniform_(-0.0625, 0.0625)
        )  # UniformFill
        self._209 = nn.Parameter(
            torch.empty([256, 512]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._210 = nn.Parameter(
            torch.empty([256]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._211 = nn.Parameter(
            torch.empty([512, 256]).uniform_(-0.0625, 0.0625)
        )  # UniformFill
        self._212 = nn.Parameter(
            torch.empty([512]).uniform_(-0.0625, 0.0625)
        )  # UniformFill
        self._213 = nn.Parameter(
            torch.empty([1, 512]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._214 = nn.Parameter(
            torch.empty([1]).uniform_(-0.04419417306780815, 0.04419417306780815)
        )  # UniformFill
        self._215 = nn.Parameter(
            torch.empty([15, num_tables + 1]).uniform_(-0.12309149097933274, 0.12309149097933274)
        )  # XavierFill
        self._216 = nn.Parameter(torch.empty([15]).fill_(0.0))  # ConstantFill

    def forward(
        self,
        embeddings,
        dense_projection
    ):
        """
        Returns the loss only
        """
        _418 = embeddings
        _419 = _418.permute(0, 2, 1)  # Transpose
        _420 = torch.reshape(_419, (-1, embeddings.size(1)))  # Reshape
        _421 = torch.addmm(self._216, _420, self._215.t())  # FC
        _422 = torch.reshape(_421, (-1, embeddings.size(2), 15))  # Reshape
        _423 = torch.bmm(_418, _422)  # BatchMatMul
        _424 = torch.flatten(_423, start_dim=-2)  # Flatten
        _425 = torch.tanh(_424)  # Tanh
        _426 = torch.cat([_425, dense_projection], dim=1)  # Concat
        _427 = torch.addmm(self._202, _426, self._201.t())  # FC
        _428 = torch.addmm(self._204, _426, self._203.t())  # FC
        _429 = torch.cat([_427, _428], dim=1)  # Concat
        _430 = torch.relu(_429)  # Relu
        _431 = torch.addmm(self._206, _430, self._205.t())  # FC
        _432 = torch.relu(_431)  # Relu
        _433 = torch.addmm(self._208, _432, self._207.t())  # FC
        _434 = _433 + _430  # Sum
        _435 = torch.relu(_434)  # Relu
        _436 = torch.addmm(self._210, _435, self._209.t())  # FC
        _437 = torch.relu(_436)  # Relu
        _438 = torch.addmm(self._212, _437, self._211.t())  # FC
        _439 = _438 + _435  # Sum
        _440 = torch.relu(_439)  # Relu
        _441 = torch.addmm(self._214, _440, self._213.t())  # FC
        return _441

class DistributedPartitionShardedSNN(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim,
                 dense_features_dim):
        super(DistributedPartitionShardedSNN, self).__init__()
        device = torch.cuda.current_device()
        self.dense_arch = DenseArch(dense_features_dim, embedding_dim).to(device)
        #print(f"Rank {dist.get_rank()}, nn {self.dense_arch._0.device}")
        assert num_tables % dist.get_world_size() == 0
        self.embedding_tables = UniformShardedEmbeddingBags(
            num_tables // dist.get_world_size(), num_embeddings, embedding_dim)
        self.over_arch = OverArch(num_tables).to(device)

    #def forward(self, dense_features, partitioned_sparse_features):
    def forward(self, dense_features, partitioned_sparse_features, partitions_offsets):
        dense_projection, dense_embedding = self.dense_arch(dense_features)
        res = self.embedding_tables(partitioned_sparse_features, partitions_offsets)
        #print(res.detach())
        embeddings_x = torch.ops.sparse_offset_forward.all2all(res)
        #print(embedding_x)
        #embeddings_x = All2AllFunction.apply(
        #   self.embedding_tables(partitioned_sparse_features))
        logits = self.over_arch(torch.cat([embeddings_x, dense_embedding.unsqueeze(1)], dim=1), dense_projection)
        return logits