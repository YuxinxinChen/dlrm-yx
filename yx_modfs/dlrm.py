import argparse
import sys
import enum
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.profiler import record_function
import torch.cuda.nccl as nccl

sys.path.append('/home/yuxin420/dlrm-yx/yx_modfs/build/lib.linux-x86_64-cpython-310')

lookup_time = 0

class EmbeddingLocation(enum.Enum):
    DEVICE = 0
    HOST_MAPPED = 1

def get_table_batched_offsets_from_dense(merged_indices):
    (NB, T, B, L) = merged_indices.size()
    lengths = np.ones(shape=(NB, T, B), dtype=np.int32) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.contiguous().view(-1),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist()), dtype=torch.int32),
    )

class RandomDataset():
    def __init__(self,
        m_den, 
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
        np.random.seed(self.rand_seed)

        n_emb = len(ln_emb)
        tmp_offsets_list = []
        tmp_indices_list = []
        tmp_dens = []
        for _ in range(num_batches):
            xs = [torch.randint(low=0, high=ln_emb[k], size=(batch_size, num_indices_per_lookup), dtype=torch.int32) for k in range(n_emb)]
            x = torch.cat([x.view(1, batch_size, num_indices_per_lookup) for x in xs], dim=0)
            merged_indices = torch.cat([x.view(1, -1, batch_size, num_indices_per_lookup) for x in x], dim=0)
            (tmp_indices, tmp_offsets) = get_table_batched_offsets_from_dense(merged_indices)
            tmp_offsets_list.append(tmp_offsets)
            tmp_indices_list.append(tmp_indices)
            tmp_dens.append(torch.tensor(np.random.rand(batch_size, m_den).astype(np.float32)))
        self.offsets_list = tmp_offsets_list
        self.indices_list = tmp_indices_list
        self.dens_inputs = tmp_dens

    def print_sparse(self):
        print("Sparse Features")
        print("num of batches: ", self.num_batches)
        print("batch size: ", self.batch_size)
        print("num of indices per lookup: ", self.num_indices_per_lookup)
        for i in range(self.num_batches):
            print("batch ", i)
            print("offsets: ", self.offsets_list[i])
            print("indiceis: ", self.indices_list[i])

    def print_dens(self):
        print("Dense Features")
        print("num of batches: ", self.num_batches)
        print("batch size: ", self.batch_size)
        for i in range(self.num_batches):
            print("batch ", i)
            print(self.dens_inputs[i])

def get_split_len(items, persons):
    k, m = divmod(items, persons)
    splits = [(k+1) if i < m else k for i in range(persons)]
    return splits

def get_my_slice(items, persons, my_rank):
    k, m = divmod(items, persons)
    return slice(
        my_rank * k + min(my_rank, m), (my_rank + 1) * k + min(my_rank + 1, m), 1
    )

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
        L_max = int(embedding_dimension/4)*4
        BT_block_size = int(max(2048/(L_max/4), 1))

        import table_batched_embeddings_yx.batched_forward
        res = torch.ops.batched_forward.forward(
            weights,
            table_offsets,
            offsets,
            indices,
            L_max,
            BT_block_size,
            False
        )
        return res

class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, ln.size-1):
            n = ln[i]
            m = ln[i+1]
            LL = nn.Linear(int(n), int(m), bias=True)

            mean = 0.0
            std_dev = np.sqrt(2/(m+n))
            W = np.random.normal(mean, std_dev, size=(m,n)).astype(np.float32)
            std_dev = np.sqrt(1/m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            
            if i== sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        
        return torch.nn.Sequential(*layers)
    
    def create_emb(self, 
        managed=EmbeddingLocation.DEVICE
    ):
        assert managed in (EmbeddingLocation.DEVICE, EmbeddingLocation.HOST_MAPPED)
        numTables = len(self.ln_emb)
        self.emb_l = [torch.nn.EmbeddingBag(self.ln_emb[i], self.m_spa, mode="sum", sparse=True, dtype=torch.float32) for i in range(numTables)]
        if managed == EmbeddingLocation.DEVICE:
           for t in self.emb_l:
                t.cuda()


    def create_emb_batched(self, 
        managed=EmbeddingLocation.DEVICE):
        assert managed in (EmbeddingLocation.DEVICE, EmbeddingLocation.HOST_MAPPED)
        all_embedding_data = []
        all_table_offsets=[]

        self.create_emb(EmbeddingLocation.HOST_MAPPED)
        tmp_agg_by_device = []
        for dev in range(self.ndevices):
            tmp_agg_by_device.append([])
        
        for t in range(len(self.ln_emb)):
            tmp_agg_by_device[self.table_device_indices[t]].append(self.emb_l[t].weight.data)    
        
        for (list, dev) in zip(tmp_agg_by_device, range(self.ndevices)):
            if managed == EmbeddingLocation.DEVICE:
                all_embedding_data.append(torch.cat(list).to(dev))
                table_offsets_data = torch.tensor(
                    [0] + 
                    np.cumsum(self.device_table_indices[dev][:-1]).tolist(),
                    device=dev,
                    dtype=torch.int32
                )
                all_table_offsets.append(table_offsets_data)
            elif managed == EmbeddingLocation.HOST_MAPPED:
                all_embedding_data.append(torch.cat(list))
                table_offsets_data = torch.tensor(
                    [0] + 
                    np.cumsum(self.device_table_indices[dev][:-1]).tolist(),
                    dtype=torch.int32
                )
                all_table_offsets.append(table_offsets_data)

        self.all_embedding_weights = all_embedding_data
        self.all_table_offsets = all_table_offsets

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        ndevices=-1,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        rand_seed=12321,
    ):
        super(DLRM_Net, self).__init__()
        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
        ):
            self.m_spa = m_spa
            self.ln_emb = ln_emb
            self.ln_bot = ln_bot
            self.ln_top = ln_top
            self.ndevices = ndevices
            self.rand_seed = rand_seed
            torch.manual_seed(self.rand_seed)
            np.random.seed(self.rand_seed)

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            self.loss_fn = torch.nn.MSELoss(reduction="mean")

            if self.ndevices > 1:
                n_emb = len(ln_emb)
                if n_emb < self.ndevices:
                    sys.exit("%d embedding tables, %d GPUs, Not sufficient devices!" % (n_emb, self.ndevices))
                self.n_emb_partition_table=get_split_len(n_emb, ndevices)

                table_device_indices = [0]*n_emb
                device_table_indices = []
                for dev in range(ndevices):
                    local_emb_slice = get_my_slice(n_emb, ndevices, dev)
                    indices = range(local_emb_slice.start, local_emb_slice.stop, local_emb_slice.step)
                    device_table_indices.append(self.ln_emb[local_emb_slice])
                    for it in indices:
                        table_device_indices[it] = dev
                self.device_table_indices = device_table_indices
                self.table_device_indices = table_device_indices
                self.create_emb_batched(EmbeddingLocation.DEVICE)
                self.top_l = torch.nn.DataParallel(self.top_l, device_ids=range(ndevices)).cuda()
                self.bot_l = torch.nn.DataParallel(self.bot_l, device_ids=range(ndevices)).cuda()

            elif self.ndevices == 1:
                self.table_device_indices = [0]*len(ln_emb)
                self.create_emb(EmbeddingLocation.DEVICE)
                self.top_l.cuda()
                self.bot_l.cuda()
            else:
                self.create_emb(EmbeddingLocation.HOST_MAPPED)

    def apply_mlp(self, x, layers):
        return layers(x)
    
    def apply_emb(self, indices, offsets):
        return [self.emb_l[i](indices[i], offsets[i]) for i in range(len(self.ln_emb))]

    def interact_features(self, x, ly):
        if isinstance(ly, list):
            return torch.cat(([x]+ly), dim=1)
        elif torch.is_tensor(ly):
            return torch.cat((x, ly), dim=1)

    def sequential_forward(self, dense_input, indices, offsets):
        with record_function("module::forward_pass::bottom_mlp"):
            x = self.apply_mlp(dense_input, self.bot_l)
        with record_function("module::forward_pass::embedding_lookups"):
            start_time = time.time()
            ly = self.apply_emb(indices, offsets)
            past_time = time.time()-start_time
            global lookup_time
            lookup_time = lookup_time + past_time
            print("embedding lookup time (in ms): ", past_time*1000.0)
        with record_function("module::forward_pass::interaction"):
            z = self.interact_features(x, ly)
        with record_function("module::forward_pass::top_mlp"):
            p = self.apply_mlp(z, self.top_l)
        
        return p
    
    def parallel_forward(self, dense_input, indices, offsets):
        with record_function("module::forwaard_pass::bottom_mlp"):
            x = self.apply_mlp(dense_input, self.bot_l)
        with record_function("module::forward_pass::embedding_lookups"):
            start_time = time.time()
            ly = LookupFunction.apply(
                self.all_embedding_weights,
                self.all_table_offsets,
                indices,
                offsets
            )
            ly = ly.reshape(ly.shape[0], -1)
            past_time = time.time() - start_time
            global lookup_time
            lookup_time = lookup_time + past_time
            print("embedding lookup time (in ms): ", past_time*1000.0)
        with record_function("module::forward_pass::interaction"):
            z = self.interact_features(x, ly)
        with record_function("module::forward_pass::top_mlp"):
            p = self.apply_mlp(z, self.top_l)
        return p

    def forward(self, dense_input, indices,offsets):
        if self.ndevices <= 1:
            return self.sequential_forward(dense_input, indices, offsets)
        elif self.ndevices > 1:
            return self.parallel_forward(dense_input, indices, offsets)

    def distributed_emb_inputs(self, indices, offsets):
        n_emb = len(self.ln_emb)
        assert n_emb == len(indices)
        ndevices = self.ndevices
        tmp_offsets = []
        tmp_indices = []
        emb_input_offsets = []
        emb_input_indices = []
        for dev in range(ndevices):
            tmp_offsets.append([])
            tmp_indices.append([])
            emb_input_offsets.append([])
            emb_input_indices.append([])

        for i in range(len(indices)):
            tmp_indices[self.table_device_indices[i]].append(indices[i]) 
            offsets[i] = torch.cat((offsets[i], torch.tensor([len(indices[i])], dtype=offsets[i].dtype)), dim=0)
            if not tmp_offsets[self.table_device_indices[i]]:
                tmp_offsets[self.table_device_indices[i]].append(offsets[i])
            else:
                tmp_offsets[self.table_device_indices[i]].append(offsets[i][1:]+tmp_offsets[self.table_device_indices[i]][-1][-1])

        for i in range(ndevices):
            offset_agg = torch.cat(tmp_offsets[i])        
            emb_input_offsets[i] = offset_agg
            indices_agg = torch.cat(tmp_indices[i])
            emb_input_indices[i] = indices_agg
        return emb_input_indices, emb_input_offsets


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
        "--arch-embedding-size", type=dash_separated_ints, default="4-3"
    )
    parser.add_argument("--num-batches", type=int, default=2)    
    parser.add_argument("--batches-size", type=int, default=2)    
    parser.add_argument("--indices-per-lookup", type=int, default=3)    
    parser.add_argument("--rand-seed", type=int, default=12321) 

    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="2-1")

    global args
    args = parser.parse_args()

    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    ln_top = np.fromstring(args.arch_mlp_top, dtype=int, sep="-")
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    use_gpu = args.use_gpu and torch.cuda.is_available()
    ln_top = np.insert(ln_top, 0, args.sparse_feature_size*len(ln_emb)+ln_bot[-1])
    assert(ln_top[0] == args.sparse_feature_size*len(ln_emb)+ln_bot[-1])

    ndevices = -1
    if use_gpu:
        ngpus = torch.cuda.device_count()
        assert ngpus >= args.world_size
        ndevices = args.world_size
    
    dlrm = DLRM_Net(args.sparse_feature_size, ln_emb, ln_bot,ln_top, ndevices)
    #print(dlrm.top_l)
    #print(next(dlrm.top_l.parameters()).is_cuda)
    #print(dlrm.bot_l)
    #print(next(dlrm.bot_l.parameters()).is_cuda)
    #print(dlrm.emb_l)
    #print(next(dlrm.emb_l[0].parameters()).is_cuda)

    dataset = RandomDataset(ln_bot[0], dlrm.ln_emb, args.num_batches, args.batches_size, args.indices_per_lookup)

    forward_time = 0.0
    for bid in range(dataset.num_batches):
        print("-------------batch %d------------" % bid)
        indices = []
        offsets = []
        for i in range(len(ln_emb)):
            indices.append(dataset.indices_list[bid][args.batches_size*i*args.indices_per_lookup:args.batches_size*(i+1)*args.indices_per_lookup])
            offsets.append(dataset.offsets_list[bid][args.batches_size*i:args.batches_size*(i+1)]-dataset.offsets_list[bid][args.batches_size*i])
        dense_x = dataset.dens_inputs[bid]
        if use_gpu:
            dense_x = dense_x.cuda()
            if ndevices > 1:
                (indices, offsets) = dlrm.distributed_emb_inputs(indices, offsets)
                for i in range(len(indices)):
                    indices[i] = indices[i].cuda(i)
                    offsets[i] = offsets[i].cuda(i)
            elif ndevices == 1:
                for i in range(len(indices)):
                    indices[i] = indices[i].cuda()
                    offsets[i] = offsets[i].cuda()
        start_time = time.time()
        out = dlrm(dense_x, indices, offsets)
        end_time = time.time()
        forward_time = forward_time + (end_time-start_time)
        #filename = str(args.world_size)+"gpu.txt"
        #f = open(filename, "w")
        #for o in out:
        #    to = o.detach()
        #    f.write(str(to)+"\n")
        #f.close()
    print("forward time (in second): ", forward_time)
    print("embedding lookup time (in ms): ", lookup_time/args.num_batches*1000.0)

if __name__ == "__main__":
    run()