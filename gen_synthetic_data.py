"""Generate training data, indices and offsets, and embedding table config
"""

import argparse
import os
import json

import numpy as np
import torch

import dlrm_data_pytorch as dp


def gen_table_configs(args):
    rows = np.random.randint(args.row_range[0], args.row_range[1], args.T)
    pooling_factors = np.random.randint(args.pooling_factor_range[0], args.pooling_factor_range[1], args.T)
    dims = np.random.choice(args.dim_range, args.T)
    table_configs = {}
    table_configs["tables"] = []
    for i in range(args.T):
        table_config = {}
        table_config["index"] = i
        table_config["row"] = int(rows[i])
        table_config["dim"] = int(dims[i])
        table_config["pooling_factor"] = int(pooling_factors[i])
        table_configs["tables"].append(table_config)
    return table_configs


def generate_random_data(
    table_configs,
    m_den,
    nbatches,
    mini_batch_size,
    num_targets=1,
    round_targets=False
):

    ln_emb = [config["row"] for config in table_configs["tables"]]
    pooling_factors = [config["pooling_factor"] for config in table_configs["tables"]]
    data_size = nbatches * mini_batch_size

    lT = []
    lX = []
    lS_offsets = []
    lS_indices = []

    for j in range(0, nbatches):
        # number of data points in a batch
        n = min(mini_batch_size, data_size - (j * mini_batch_size))

        # generate a batch of dense and sparse features
        (Xt, lS_emb_offsets, lS_emb_indices) = generate_uniform_input_batch(
            m_den,
            ln_emb,
            n,
            pooling_factors,
        )

        # dense feature
        lX.append(Xt)
        # sparse feature (sparse indices)
        lS_offsets.append(lS_emb_offsets)
        lS_indices.append(lS_emb_indices)

        # generate a batch of target (probability of a click)
        P = dp.generate_random_output_batch(n, num_targets, round_targets)
        lT.append(P)

    return (nbatches, lX, lS_offsets, lS_indices, lT, ln_emb)

def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    pooling_factors,
):
    # dense feature
    Xt = torch.tensor(np.random.rand(n, m_den).astype(np.float32))
    Xt = torch.log(Xt + 1)

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i in range(len(ln_emb)):
        size = ln_emb[i]
        num_indices_per_lookup = pooling_factors[i]

        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n): 
            # num of sparse indices to be used per embedding
            while True:
                r = np.random.random(min(size, num_indices_per_lookup))
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
                if sparse_group.size == num_indices_per_lookup:
                    break
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)

def main():
    parser = argparse.ArgumentParser("Generate synthetic data")
    parser.add_argument('--T', type=int, default=12)
    parser.add_argument('--m-den', type=int, default=512)
    parser.add_argument('--num_batches', type=int, default=10)
    parser.add_argument('--mini-batch-size', type=int, default=2048)
    parser.add_argument('--row-range', type=str, default="500,10000") # Uniformly sample
    parser.add_argument('--dim-range', type=str, default="64,128,256,512") # Randomly select one of them
    parser.add_argument('--pooling-factor-range', type=str, default="10,500") # Uniformly sample
    parser.add_argument('--out-dir', type=str, default="synthetic")

    args = parser.parse_args()

    args.row_range = list(map(int, args.row_range.split(",")))
    args.dim_range = list(map(int, args.dim_range.split(",")))
    args.pooling_factor_range = list(map(int, args.pooling_factor_range.split(",")))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Generate table configs
    print("Generating table configs...")
    table_configs = gen_table_configs(args)
    with open(os.path.join(args.out_dir, "table_configs.json"), "w") as f:
        json.dump(table_configs, f)

    print("Generating data...")
    (nbatches, lX, lS_offsets, lS_indices, lT, ln_emb) = generate_random_data(
        table_configs,
        args.m_den,
        args.num_batches,
        args.mini_batch_size
    )

    data = {
        "nbatches": nbatches,
        "lX": lX,
        "lS_offsets": lS_offsets,
        "lS_indices": lS_indices,
        "lT": lT,
    }

    torch.save(data, os.path.join(args.out_dir, "data.pt"))

if __name__ == '__main__':
    main()
