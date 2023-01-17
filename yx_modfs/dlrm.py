
import argparse
import sys
import enum

import numpy as np
from numpy import random as ra
import torch
import torch.nn as nn

sys.path.append('/home/yuxin420/dlrm-yx/yx_modfs/build/lib.linux-x86_64-cpython-310')

def get_table_batched_offsets_from_dense(merged_indices):
    (NB, T, B, L) = merged_indices.size()
    lengths = np.ones(shape=(NB, T, B), dtype=np.int32) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.contiguous().view(-1),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist()), dtype=torch.int32),
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
        #all_indices = []
        tmp_offsets_list = []
        tmp_indices_list = []
        iter_batch = 0
        for _ in range(num_batches):
            xs = [torch.randint(low=0, high=ln_emb[k], size=(batch_size, num_indices_per_lookup), dtype=torch.int32) for k in range(n_emb)]
            x = torch.cat([x.view(1, batch_size, num_indices_per_lookup) for x in xs], dim=0)
            #all_indices.append(x)
            merged_indices = torch.cat([x.view(1, -1, batch_size, num_indices_per_lookup) for x in x], dim=0) 
        #merged_indices = torch.cat([x.view(1, -1, batch_size, num_indices_per_lookup) for x in all_indices], dim=0) 
            (tmp_indices, tmp_offsets) = get_table_batched_offsets_from_dense(merged_indices)
            tmp_offsets_list.append(tmp_offsets)
            tmp_indices_list.append(tmp_indices)
        self.offsets_list = tmp_offsets_list
        self.indicies_list = tmp_indices_list

    def print(self):
        print("num of batches: ", self.num_batches)
        print("batch size: ", self.batch_size)
        print("num of indices per lookup: ", self.num_indices_per_lookup)
        for i in range(self.num_batches):
            print("batch ", i)
            print("offsets: ", self.offsets_list[i])
            print("indiceis: ", self.indicies_list[i])
        #print("lookup offsets: ", self.offsets)
        #print("lookup indices: ", self.indices)


def get_split_len(items, persons):
    k, m = divmod(items, persons)
    splits = [(k+1) if i < m else k for i in range(persons)]
    return splits

def get_my_slice(items, persons, my_rank):
    k, m = divmod(items, persons)
    return slice(
        my_rank * k + min(my_rank, m), (my_rank + 1) * k + min(my_rank + 1, m), 1
    )
class Optimizer(enum.Enum):
    SGD = 1
    APPROX_ROWWISE_ADAGRAD = 2
    EXACT_ROWWISE_ADAGRAD = 3


class EmbeddingLocation(enum.Enum):
    DEVICE = 0
    HOST_MAPPED = 1

class LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weights,
        table_offsets,
        indices,
        offsets,
    ):
        ndevices = len(weights)
        embedding_dimension = weights[0].shape[1]
        print("number of devices: ", ndevices)
        print("embedding dimension: ", embedding_dimension)
        BT_block_size = int(max(512 / embedding_dimension, 1))
        L_max = int(embedding_dimension/4)*4
        print("weights:")
        print(weights)
        print("table_offsets:")
        print(table_offsets)
        print("sparse indices:")
        print(indices)
        print("sparse offsets:")
        print(offsets)

        import table_batched_embeddings_yx.batched_forward
        res = torch.ops.batched_forward.forward(
            weights,
            table_offsets,
            offsets,
            indices,
            L_max,
            BT_block_size,
            False)
        print(res[0]) 
        print(res[1]) 
        #print(weights)
        #a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        #print(type(a))
        #return table_batched_embeddings_yx.hello(
        #    [a,a]
            #weights,
            #BT_block_size,
            #False
        #)


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
            self.ln_emb = sorted(ln_emb)
            self.ndevices = ndevices

        if self.ndevices > 1:
            n_emb = len(ln_emb)
            if n_emb < self.ndevices:
                sys.exit("%d embedding tables, %d GPUs, Not sufficient device!" %(n_emb, self.ndevices))
            self.n_global_emb = n_emb
            self.n_emb_partition_table = get_split_len(n_emb, ndevices)

            table_device_indices = [0] * n_emb
            device_table_indices = []
            for dev in range(ndevices):
                local_emb_slice = get_my_slice(n_emb, ndevices, dev)
                indices = range(local_emb_slice.start, local_emb_slice.stop, local_emb_slice.step)
                device_table_indices.append(self.ln_emb[local_emb_slice])
                for it in indices:
                    table_device_indices[it] = dev
                #local_emb_indices = list(range(n_emb))[local_emb_slice] 

            self.device_indices = table_device_indices
            self.device_table_partition = device_table_indices
        print(self.device_table_partition)
        self.create_emb_batched()
        #print(self.all_embedding_weights)
        #print(self.all_table_offsets)
    
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

        for _ in range(ndevices):
            t_list.append([])
            i_list.append([])

        #L = int(dataset.indices.shape[0] / dataset.batch_size / n_emb / dataset.num_batches)
        L = dataset.num_indices_per_lookup
        for bid in range(dataset.num_batches):
            tmp_offset_agg = []
            tmp_indices_agg = []
            for _ in range(ndevices):
                tmp_offset_agg.append([])
                tmp_indices_agg.append([])
            for k in range(n_emb):
                o = dataset.offsets_list[bid][k*dataset.batch_size:(k+1)*dataset.batch_size+1]
                i = dataset.indicies_list[bid][k*dataset.batch_size*L:(k+1)*dataset.batch_size*L]
                if not tmp_offset_agg[self.device_indices[k]]:
                    if o[0] == 0:
                        tmp_offset_agg[self.device_indices[k]].append(o)
                    else:
                        tmp_offset_agg[self.device_indices[k]].append(o[0:]-o[0])
                else:
                    tmp_offset_agg[self.device_indices[k]].append(o[1:]-o[0]+tmp_offset_agg[self.device_indices[k]][-1][-1])
                tmp_indices_agg[self.device_indices[k]].append(i)
            for dev in range(ndevices):
                t_list[dev].append(torch.cat(tmp_offset_agg[dev], dim=0))
                i_list[dev].append(torch.cat(tmp_indices_agg[dev], dim=0))
        return t_list, i_list
                
        #for bid in range(dataset.num_batches):
        #    #indices_offset = [0]*ndevices
        #    for k in range(n_emb):
        #        o = dataset.offsets[(bid*dataset.batch_size*n_emb+k*dataset.batch_size):(bid*dataset.batch_size*n_emb+(k+1)*dataset.batch_size+1)]
        #        i = dataset.indices[(bid*dataset.batch_size*n_emb*L+k*dataset.batch_size*L):(bid*dataset.batch_size*n_emb*L+(k+1)*dataset.batch_size*L)]
        #        if not tmp_o[self.device_indices[k]]:
        #            if o[0] == 0:
        #                tmp_o[self.device_indices[k]].append(o)
        #            else:
        #                tmp_o[self.device_indices[k]].append(o[0:]-o[0]) 
        #        else:
        #            tmp_o[self.device_indices[k]].append(o[1:]-o[0]+tmp_o[self.device_indices[k]][-1][-1])
        #        tmp_i[self.device_indices[k]].append(i)
        #        #tmp_i[self.device_indices[k]].append(i+indices_offset[self.device_indices[k]])
        #        #indices_offset[self.device_indices[k]] += self.ln_emb[k]

        #for dev in range(ndevices):
        #    t_list.append(torch.cat(tmp_o[dev], dim=0))
        #    i_list.append(torch.cat(tmp_i[dev], dim=0))

        #return t_list, i_list

    def create_emb_batched(
        self, 
        learning_rate=0.1, 
        optimizer=Optimizer.SGD,
        managed=EmbeddingLocation.DEVICE):
        assert managed in (EmbeddingLocation.DEVICE, EmbeddingLocation.HOST_MAPPED)
        all_embedding_data = []
        all_talbe_offsets = []
        for dev in range(self.ndevices):
            Ext = np.sum(self.device_table_partition[dev])
            if managed == EmbeddingLocation.DEVICE:
                print("Allocating device embedding bag on device %d, size (%d, %d)\n" % (dev, Ext, self.m_spa))
                embedding_data = torch.randn(
                    size=(Ext, self.m_spa),
                    device=dev,
                    dtype=torch.float32
                )
                #all_embedding_data.append(nn.Parameter(embedding_data))
                all_embedding_data.append(embedding_data)
            elif managed == EmbeddingLocation.HOST_MAPPED:
                print("Allocating host-mapped embedding bag")
                embedding_data = torch.randn(
                    size=(Ext, self.m_spa),
                    dtype=torch.float32
                )
            all_talbe_offsets.append(torch.tensor(
                    [0]
                    + np.cumsum(self.device_table_partition[dev][:-1]).tolist()
                , dtype=torch.int32)
            )
        self.all_embedding_weights = all_embedding_data
        self.all_table_offsets = all_talbe_offsets
        self.optimizer = optimizer
        self.learning_rate = learning_rate
    
    def forward(self, sparse_indices, sparse_offsets):
        return LookupFunction.apply(
            self.all_embedding_weights,
            self.all_table_offsets,
            sparse_indices,
            sparse_offsets,
        )

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
    dataset = RandomDataset(dlrm.ln_emb, args.num_batches, args.batches_size, args.indices_per_lookup)
    dataset.print()

    (partitioned_offsets_batches, partitioned_indices_batches) = dlrm.distribute_emb_data(dataset)

    for bid in range(dataset.num_batches):
        merged_offsets = []
        merged_indices = []
        for dev_id in range(dlrm.ndevices):
            merged_offsets.append(partitioned_offsets_batches[dev_id][bid])
            merged_indices.append(partitioned_indices_batches[dev_id][bid])
        dlrm(merged_indices, merged_offsets)

    #dlrm(partitioned_indices, partitioned_offsets)


if __name__ == "__main__":
    run()