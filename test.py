# import torch
# import torch.distributed as dist
# import extend_distributed as ext_dist

# ext_dist.init_distributed(local_rank=-1, use_gpu=True, backend="nccl")

# # ext_dist.print_all(
# #         "world size: %d, current rank: %d, local rank: %d"
# #         % (ext_dist.my_size, ext_dist.my_rank, ext_dist.my_local_rank)
# #     )
# rank = ext_dist.my_rank
# input = (torch.arange(4) + rank * 4)
# input = input.to("cuda:{}".format(rank))
# print(input)

"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    input = (torch.arange(4) + rank * 4)
    input = input.to("cuda:{}".format(rank))
    print(input)
    pass

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()