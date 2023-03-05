import os 
import sys
import socket
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from models import DistributedPartitionShardedSNN


def cleanup():
    dist.destroy_process_group()

def div_round_up(a, b):
    return int((a + b - 1) // b) * b

def benchmark_partitioned_snn_forward(
                                    num_tables,
                                    num_embeddings,
                                    embedding_dim,
                                    dense_features_dim,
                                    batch_size,
                                    bag_size,
                                    iters):
    assert batch_size % dist.get_world_size() == 0
    assert num_tables % dist.get_world_size() == 0
    print("Benchmark on rank %d, num_tables %d, num_embeddings %d, embedding_dim %d, batch_size %d, dense_feature_dim %d, bag size %d" %(dist.get_rank(), num_tables, num_embeddings, embedding_dim, batch_size, dense_features_dim, bag_size))

    net = DistributedPartitionShardedSNN(num_tables, num_embeddings, embedding_dim, dense_features_dim)
    dense_features = torch.randn(batch_size// dist.get_world_size(), dense_features_dim, device=torch.cuda.current_device())
    dense_features.requires_grad = False
    shared_sparse_features = torch.randint(low=0,
                                           high=num_embeddings,
                                           size=(batch_size, (num_tables//dist.get_world_size())*bag_size),
                                           device=torch.cuda.current_device()).int()
    shared_sparse_features.requires_grad = False
    shared_offsets = torch.ones(batch_size,(num_tables//dist.get_world_size())+1).int()
    for i in range(batch_size):
        shared_offsets[i][0] = 0
        shared_offsets[i][:] = torch.cumsum(shared_offsets[i][:]*bag_size, 0)
    
    shared_offsets.requires_grad = False
    shared_offsets = shared_offsets.to(torch.cuda.current_device())
    #print(shared_offsets.size())
    #print(shared_offsets)
    #print(shared_sparse_features.size())
    #print(shared_sparse_features)
    net.forward(dense_features, shared_sparse_features, shared_offsets)

def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value

if __name__ == "__main__":
        ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--sparse-feature-size", type=int, default=4)    
    parser.add_argument("--num-tables", type=int, default=0)
    parser.add_argument(
        "--num_embeddings", type=int, default=10
    )
    parser.add_argument("--num-batches", type=int, default=2)    
    parser.add_argument("--batches-size", type=int, default=2)    
    parser.add_argument("--indices-per-lookup", type=int, default=3)    
    parser.add_argument("--rand-seed", type=int, default=12321) 
    parser.add_argument("--dense-feature-size", type=int, default=4)

    global args 
    args = parser.parse_args()

    global world_size
    global world_rank
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    hostname = socket.gethostname()
    dist.init_process_group('mpi', rank=world_rank, world_size=world_size)

    if args.num_tables == 0:
        args.num_tables = world_size
    use_gpu = args.use_gpu and torch.cuda.is_available()
    ndevices = -1
    if use_gpu:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to test multi-process dlrm, but got {n_gpus}"
        assert n_gpus >= world_size
        ndevices = world_size
        torch.cuda.set_device(world_rank)

    print(f"I am {world_rank} of {world_size} in {hostname} current device {torch.cuda.current_device()}")

    benchmark_partitioned_snn_forward(args.num_tables, args.num_embeddings, args.sparse_feature_size, args.dense_feature_size, args.batches_size, args.indices_per_lookup, 1)

