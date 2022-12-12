
import argparse
import sys

import numpy as np
from numpy import random as ra
import torch
import torch.nn as nn

def get_table_batched_offsets_from_dense(merged_indices):
    (NB, T, B, L) = merged_indices.size()
    lengths = np.ones(shape=(NB, T, B), dtype=np.int64) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.contiguous().view(-1),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())),
    )

class RandomDataset():
    def __init__(
        self,
        ln_emb,
        num_batches,
        batch_size,
        num_indices_per_lookup,
        rand_seed=12321,
        rand_data_dis="uniform",
    ):
        self.ln_emb = ln_emb
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.rand_seed = rand_seed
        self.rand_data_dis = rand_data_dis

        torch.random.manual_seed(self.rand_seed)
        n_emb = len(ln_emb)
        all_indices = []
        for _ in range(num_batches):
            xs = [torch.randint(low=0, high=ln_emb[k], size=(batch_size, num_indices_per_lookup)) for k in range(n_emb)]
            x = torch.cat([x.view(1, batch_size, num_indices_per_lookup) for x in xs], dim=0)
            all_indices.append(x)
        merged_indices = torch.cat([x.view(1, -1, batch_size, num_indices_per_lookup) for x in all_indices], dim=0) 
        
        (self.indices, self.offsets) = get_table_batched_offsets_from_dense(merged_indices)

    def print(self):
        print("num of batches: ", self.num_batches)
        print("batch size: ", self.batch_size)
        print("num of indices per lookup: ", self.num_indices_per_lookup)
        print("lookup offsets: ", self.offsets)
        print("lookup indices: ", self.indices)


def get_split_len(items, persons):
    k, m = divmod(items, persons)
    splits = [(k+1) if i < m else k for i in range(persons)]
    return splits

def get_my_slice(items, persons, my_rank):
    k, m = divmod(items, persons)
    return slice(
        my_rank * k + min(my_rank, m), (my_rank + 1) * k + min(my_rank + 1, m), 1
    )

class DLRM_Net(nn.Module):
    #def create_emb(self, emb_vec_dim, ln):
    #    emb_l = nn.ModuleList()

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ndevices=-1,
    ):
        super(DLRM_Net, self).__init__()
        if(
            (m_spa is not None)
            and (ln_emb is not None)
        ):
            self.m_spa = m_spa
            self.ln_emb = ln_emb
            self.ndevices = ndevices

        if self.ndevices > 1:
            n_emb = len(ln_emb)
            if n_emb < self.ndevices:
                sys.exit("%d embedding tables, %d GPUs, Not sufficient device!" %(n_emb, self.ndevices))
            self.n_global_emb = n_emb
            self.n_emb_partition_table = get_split_len(n_emb, ndevices)

            table_device_indices = [0] * n_emb
            for dev in range(ndevices):
                local_emb_slice = get_my_slice(n_emb, ndevices, dev)
                indices = range(local_emb_slice.start, local_emb_slice.stop, local_emb_slice.step)
                for it in indices:
                    table_device_indices[it] = dev
                #local_emb_indices = list(range(n_emb))[local_emb_slice] 

            self.device_indices = table_device_indices
    
    def print(self):
        print("sparse features array: ", self.ln_emb)
        print("number of devices: ", self.ndevices)
        print("partition table: ", self.n_emb_partition_table)
        print("table device indices: ", self.device_indices)
    
    def distribute_emb_data(self, dataset):
        n_emb  = len(self.ln_emb)
        ndevices = self.ndevices
        t_list = []
        i_list = []

        tmp_o = []
        tmp_i = []
        for _ in range(ndevices):
            tmp_o.append([])
            tmp_i.append([])
        L = int(dataset.indices.shape[0] / dataset.batch_size / n_emb / dataset.num_batches)

        for bid in range(dataset.num_batches):
            for k in range(n_emb):
                o = dataset.offsets[(bid*dataset.batch_size*n_emb+k*dataset.batch_size):(bid*dataset.batch_size*n_emb+(k+1)*dataset.batch_size+1)]
                i = dataset.indices[(bid*dataset.batch_size*n_emb*L+k*dataset.batch_size*L):(bid*dataset.batch_size*n_emb*L+(k+1)*dataset.batch_size*L)]
                if not tmp_o[self.device_indices[k]]:
                    if o[0] == 0:
                        tmp_o[self.device_indices[k]].append(o)
                    else:
                        tmp_o[self.device_indices[k]].append(o[0:]-o[0]) 
                else:
                    tmp_o[self.device_indices[k]].append(o[1:]-o[0]+tmp_o[self.device_indices[k]][-1][-1])
                tmp_i[self.device_indices[k]].append(i)

        for dev in range(ndevices):
            t_list.append(torch.cat(tmp_o[dev], dim=0))
            i_list.append(torch.cat(tmp_i[dev], dim=0))

        return t_list, i_list

    #def distribute_emb_data(self, batch_size, num_batches, lS_o, lS_i):
    #    n_emb  = len(self.ln_emb)
    #    ndevices = self.ndevices
    #    t_list = []
    #    i_list = []

    #    tmp_o = []
    #    tmp_i = []
    #    for _ in range(ndevices):
    #        tmp_o.append([])
    #        tmp_i.append([])
    #    L = int(lS_i.shape[0] / batch_size / n_emb / num_batches)

    #    for bid in range(num_batches):
    #        for k in range(n_emb):
    #            o = lS_o[(bid*batch_size*n_emb+k*batch_size):(bid*batch_size*n_emb+(k+1)*batch_size+1)]
    #            i = lS_i[(bid*batch_size*n_emb*L+k*batch_size*L):(bid*batch_size*n_emb*L+(k+1)*batch_size*L)]
    #            if not tmp_o[self.device_indices[k]]:
    #                if o[0] == 0:
    #                    tmp_o[self.device_indices[k]].append(o)
    #                else:
    #                    tmp_o[self.device_indices[k]].append(o[0:]-o[0]) 
    #            else:
    #                tmp_o[self.device_indices[k]].append(o[1:]-o[0]+tmp_o[self.device_indices[k]][-1][-1])
    #            tmp_i[self.device_indices[k]].append(i)

    #    for dev in range(ndevices):
    #        t_list.append(torch.cat(tmp_o[dev], dim=0))
    #        i_list.append(torch.cat(tmp_i[dev], dim=0))

    #    return t_list, i_list

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

def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--world-size", type=int, default=-1)    
    parser.add_argument("--sparse-feature-size", type=int, default=4)    
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    parser.add_argument("--num-batches", type=int, default=4)    
    parser.add_argument("--batches-size", type=int, default=2)    
    parser.add_argument("--indices-per-lookup", type=int, default=3)    
    parser.add_argument("--rand-seed", type=int, default=12321) 

    global args
    args = parser.parse_args()

    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu:
        ngpus = torch.cuda.device_count()
        assert ngpus >= args.world_size

    dlrm = DLRM_Net(args.sparse_feature_size, ln_emb, args.world_size)
    dlrm.print()
    dataset = RandomDataset(ln_emb, args.num_batches, args.batches_size, args.indices_per_lookup)
    dataset.print()

    (partitioned_offsets, partitioned_indices) = dlrm.distribute_emb_data(dataset)
    print("t_list size: ", len(partitioned_offsets))
    print("i_list size: ", len(partitioned_indices))
    print("GPU 0")
    print(partitioned_offsets[0])
    print(partitioned_indices[0])
    print("GPU 1")
    print(partitioned_offsets[1])
    print(partitioned_indices[1])


if __name__ == "__main__":
    run()