# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i)  Criteo Kaggle Display Advertising Challenge Dataset
#    https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#    ii) Criteo Terabyte Dataset
#    https://labs.criteo.com/2013/12/download-terabyte-click-logs


from __future__ import absolute_import, division, print_function, unicode_literals

# others
import os
from os import path
import sys
import bisect
import collections
import h5py

from numpy.core.numeric import indices

import data_utils

# numpy
import numpy as np
from numpy import random as ra
from collections import deque


# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

import data_loader_terabyte
import mlperf_logger


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets
class CriteoDataset(Dataset):

    def __init__(
            self,
            dataset,
            max_ind_range,
            sub_sample_rate,
            randomize,
            split="train",
            raw_path="",
            pro_data="",
            memory_map=False,
            dataset_multiprocessing=False,
    ):
        # dataset
        # tar_fea = 1   # single target
        den_fea = 13  # 13 dense  features
        # spa_fea = 26  # 26 sparse features
        # tad_fea = tar_fea + den_fea
        # tot_fea = tad_fea + spa_fea
        if dataset == "kaggle":
            days = 7
            out_file = "kaggleAdDisplayChallenge_processed"
        elif dataset == "terabyte":
            days = 24
            out_file = "terabyte_processed"
        else:
            raise(ValueError("Data set option is not supported"))
        self.max_ind_range = max_ind_range
        self.memory_map = memory_map

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
        self.npzfile = self.d_path + (
            (self.d_file + "_day") if dataset == "kaggle" else self.d_file
        )
        self.trafile = self.d_path + (
            (self.d_file + "_fea") if dataset == "kaggle" else "fea"
        )

        # check if pre-processed data is available
        data_ready = True
        if memory_map:
            for i in range(days):
                reo_data = self.npzfile + "_{0}_reordered.npz".format(i)
                if not path.exists(str(reo_data)):
                    data_ready = False
        else:
            if not path.exists(str(pro_data)):
                data_ready = False

        # pre-process data if needed
        # WARNNING: when memory mapping is used we get a collection of files
        if data_ready:
            print("Reading pre-processed data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            print("Reading raw data=%s" % (str(raw_path)))
            file = data_utils.getCriteoAdData(
                raw_path,
                out_file,
                max_ind_range,
                sub_sample_rate,
                days,
                split,
                randomize,
                dataset == "kaggle",
                memory_map,
                dataset_multiprocessing,
            )

        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]
        # print(self.offset_per_file)

        # setup data
        if memory_map:
            # setup the training/testing split
            self.split = split
            if split == 'none' or split == 'train':
                self.day = 0
                self.max_day_range = days if split == 'none' else days - 1
            elif split == 'test' or split == 'val':
                self.day = days - 1
                num_samples = self.offset_per_file[days] - \
                              self.offset_per_file[days - 1]
                self.test_size = int(np.ceil(num_samples / 2.))
                self.val_size = num_samples - self.test_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")

            '''
            # text
            print("text")
            for i in range(days):
                fi = self.npzfile + "_{0}".format(i)
                with open(fi) as data:
                    ttt = 0; nnn = 0
                    for _j, line in enumerate(data):
                        ttt +=1
                        if np.int32(line[0]) > 0:
                            nnn +=1
                    print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                          + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            # processed
            print("processed")
            for i in range(days):
                fi = self.npzfile + "_{0}_processed.npz".format(i)
                with np.load(fi) as data:
                    yyy = data["y"]
                ttt = len(yyy)
                nnn = np.count_nonzero(yyy)
                print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                      + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            # reordered
            print("reordered")
            for i in range(days):
                fi = self.npzfile + "_{0}_reordered.npz".format(i)
                with np.load(fi) as data:
                    yyy = data["y"]
                ttt = len(yyy)
                nnn = np.count_nonzero(yyy)
                print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                      + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            '''

            # load unique counts
            with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                self.counts = data["counts"]
            self.m_den = den_fea  # X_int.shape[1]
            self.n_emb = len(self.counts)
            print("Sparse features= %d, Dense features= %d" % (self.n_emb, self.m_den))

            # Load the test data
            # Only a single day is used for testing
            if self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                fi = self.npzfile + "_{0}_reordered.npz".format(
                    self.day
                )
                with np.load(fi) as data:
                    self.X_int = data["X_int"]  # continuous  feature
                    self.X_cat = data["X_cat"]  # categorical feature
                    self.y = data["y"]          # target

        else:
            # load and preprocess data
            with np.load(file) as data:
                X_int = data["X_int"]  # continuous  feature
                X_cat = data["X_cat"]  # categorical feature
                y = data["y"]          # target
                self.counts = data["counts"]
            self.m_den = X_int.shape[1]  # den_fea
            self.n_emb = len(self.counts)
            print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

            # create reordering
            indices = np.arange(len(y))

            if split == "none":
                # randomize all data
                if randomize == "total":
                    indices = np.random.permutation(indices)
                    print("Randomized indices...")

                X_int[indices] = X_int
                X_cat[indices] = X_cat
                y[indices] = y

            else:
                indices = np.array_split(indices, self.offset_per_file[1:-1])

                # randomize train data (per day)
                if randomize == "day":  # or randomize == "total":
                    for i in range(len(indices) - 1):
                        indices[i] = np.random.permutation(indices[i])
                    print("Randomized indices per day ...")

                train_indices = np.concatenate(indices[:-1])
                test_indices = indices[-1]
                test_indices, val_indices = np.array_split(test_indices, 2)

                print("Defined %s indices..." % (split))

                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(train_indices)
                    print("Randomized indices across days ...")

                # create training, validation, and test sets
                if split == 'train':
                    self.X_int = [X_int[i] for i in train_indices]
                    self.X_cat = [X_cat[i] for i in train_indices]
                    self.y = [y[i] for i in train_indices]
                elif split == 'val':
                    self.X_int = [X_int[i] for i in val_indices]
                    self.X_cat = [X_cat[i] for i in val_indices]
                    self.y = [y[i] for i in val_indices]
                elif split == 'test':
                    self.X_int = [X_int[i] for i in test_indices]
                    self.X_cat = [X_cat[i] for i in test_indices]
                    self.y = [y[i] for i in test_indices]

            print("Split data according to indices...")

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        if self.memory_map:
            if self.split == 'none' or self.split == 'train':
                # check if need to swicth to next day and load data
                if index == self.offset_per_file[self.day]:
                    # print("day_boundary switch", index)
                    self.day_boundary = self.offset_per_file[self.day]
                    fi = self.npzfile + "_{0}_reordered.npz".format(
                        self.day
                    )
                    # print('Loading file: ', fi)
                    with np.load(fi) as data:
                        self.X_int = data["X_int"]  # continuous  feature
                        self.X_cat = data["X_cat"]  # categorical feature
                        self.y = data["y"]          # target
                    self.day = (self.day + 1) % self.max_day_range

                i = index - self.day_boundary
            elif self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                i = index + (0 if self.split == 'test' else self.test_size)
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")
        else:
            i = index

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    def _default_preprocess(self, X_int, X_cat, y):
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
        if self.max_ind_range > 0:
            X_cat = torch.tensor(X_cat % self.max_ind_range, dtype=torch.long)
        else:
            X_cat = torch.tensor(X_cat, dtype=torch.long)
        y = torch.tensor(y.astype(np.float32))

        return X_int, X_cat, y

    def __len__(self):
        if self.memory_map:
            if self.split == 'none':
                return self.offset_per_file[-1]
            elif self.split == 'train':
                return self.offset_per_file[-2]
            elif self.split == 'test':
                return self.test_size
            elif self.split == 'val':
                return self.val_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train nor test.")
        else:
            return len(self.y)


def collate_wrapper_criteo_offset(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


def ensure_dataset_preprocessed(args, d_path):
    need_dataset = True
    # if args.batched_emb:
    #     tmp_d_path = '_'.join(d_path.split('_')[:-1])
    #     train_file_no_batch = tmp_d_path + "_train.bin"
    #     test_file_no_batch = tmp_d_path + "_test.bin"
    #     # val_file_no_batch = tmp_d_path + "_val.bin"
    #     counts_file_no_batch = args.raw_data_file + '_fea_count.npz'
    #     if all(path.exists(p) for p in [train_file_no_batch,
    #                                     test_file_no_batch,
    #                                     counts_file_no_batch]):
    #         print("no need dataset")
    #         need_dataset = False # No need to get the datasets if all non-batched binary files exist
    
    if need_dataset:
        _ = CriteoDataset(
            args.data_set,
            args.max_ind_range,
            args.data_sub_sample_rate,
            args.data_randomize,
            "train",
            args.raw_data_file,
            args.processed_data_file,
            args.memory_map,
            args.dataset_multiprocessing
        )

        _ = CriteoDataset(
            args.data_set,
            args.max_ind_range,
            args.data_sub_sample_rate,
            args.data_randomize,
            "test",
            args.raw_data_file,
            args.processed_data_file,
            args.memory_map,
            args.dataset_multiprocessing
        )

    for split in ['train', 'val', 'test']:
        print('Running preprocessing for split =', split)

        train_files = ['{}_{}_reordered.npz'.format(args.raw_data_file, day)
                       for
                       day in range(0, 23)]

        test_valid_file = args.raw_data_file + '_23_reordered.npz'

        output_file = d_path + '_{}.bin'.format(split)

        input_files = train_files if split == 'train' else [test_valid_file]
        data_loader_terabyte.numpy_to_binary(input_files=input_files,
                                             output_file_path=output_file,
                                             split=split)


# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def collate_wrapper_criteo_length(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = torch.stack([X_cat[:, i] for i in range(featureCnt)])
    lS_o = torch.stack(
        [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    )

    lS_l = offset_to_length_converter(lS_o, lS_i)

    return X_int, lS_l, lS_i, T


def make_criteo_data_and_loaders(args, offset_to_length_converter=False):
    if args.mlperf_logging and args.memory_map and args.data_set == "terabyte":
        # more efficient for larger batches
        data_directory = path.dirname(args.raw_data_file)

        if args.mlperf_bin_loader:
            lstr = args.processed_data_file.split("/")
            d_path = "/".join(lstr[0:-1]) + "/" + lstr[-1].split(".")[0] # + ("_batched" if args.batched_emb else "")
            train_file = d_path + "_train.bin"
            test_file = d_path + "_test.bin"
            # val_file = d_path + "_val.bin"
            counts_file = args.raw_data_file + '_fea_count.npz'

            if any(not path.exists(p) for p in [train_file,
                                                test_file,
                                                counts_file]):
                ensure_dataset_preprocessed(args, d_path)

            train_data = data_loader_terabyte.CriteoBinDataset(
                data_file=train_file,
                counts_file=counts_file,
                batch_size=args.mini_batch_size,
                max_ind_range=args.max_ind_range,
                batched_or_fbgemm_emb=args.batched_emb or args.fbgemm_emb
            )

            mlperf_logger.log_event(key=mlperf_logger.constants.TRAIN_SAMPLES,
                                    value=train_data.num_entries)

            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
                pin_memory=args.pin_memory,
                drop_last=False,
                sampler=RandomSampler(train_data) if args.mlperf_bin_shuffle else None
            )

            test_data = data_loader_terabyte.CriteoBinDataset(
                data_file=test_file,
                counts_file=counts_file,
                batch_size=args.test_mini_batch_size,
                max_ind_range=args.max_ind_range,
                batched_or_fbgemm_emb=args.batched_emb or args.fbgemm_emb
            )

            mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_SAMPLES,
                                    value=test_data.num_entries)

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
                pin_memory=args.pin_memory,
                drop_last=False,
            )
        else:
            data_filename = args.raw_data_file.split("/")[-1]

            train_data = CriteoDataset(
                args.data_set,
                args.max_ind_range,
                args.data_sub_sample_rate,
                args.data_randomize,
                "train",
                args.raw_data_file,
                args.processed_data_file,
                args.memory_map,
                args.dataset_multiprocessing
            )

            test_data = CriteoDataset(
                args.data_set,
                args.max_ind_range,
                args.data_sub_sample_rate,
                args.data_randomize,
                "test",
                args.raw_data_file,
                args.processed_data_file,
                args.memory_map,
                args.dataset_multiprocessing
            )

            train_loader = data_loader_terabyte.DataLoader(
                data_directory=data_directory,
                data_filename=data_filename,
                days=list(range(23)),
                batch_size=args.mini_batch_size,
                max_ind_range=args.max_ind_range,
                split="train"
            )

            test_loader = data_loader_terabyte.DataLoader(
                data_directory=data_directory,
                data_filename=data_filename,
                days=[23],
                batch_size=args.test_mini_batch_size,
                max_ind_range=args.max_ind_range,
                split="test"
            )
    else:
        train_data = CriteoDataset(
            args.data_set,
            args.max_ind_range,
            args.data_sub_sample_rate,
            args.data_randomize,
            "train",
            args.raw_data_file,
            args.processed_data_file,
            args.memory_map,
            args.dataset_multiprocessing,
        )

        test_data = CriteoDataset(
            args.data_set,
            args.max_ind_range,
            args.data_sub_sample_rate,
            args.data_randomize,
            "test",
            args.raw_data_file,
            args.processed_data_file,
            args.memory_map,
            args.dataset_multiprocessing,
        )

        collate_wrapper_criteo = collate_wrapper_criteo_offset
        if offset_to_length_converter:
            collate_wrapper_criteo = collate_wrapper_criteo_length

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=args.pin_memory,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_criteo,
            pin_memory=args.pin_memory,
            drop_last=False,  # True
        )

    return train_data, train_loader, test_data, test_loader


# uniform ditribution (input data)
class RandomDataset(Dataset):
    def __init__(
            self,
            m_den,
            ln_emb,
            data_size,
            num_batches,
            mini_batch_size,
            num_indices_per_lookup,
            num_indices_per_lookup_fixed,
            batched_or_fbgemm_emb,
            num_targets=1,
            round_targets=False,
            data_generation="random",
            trace_file="",
            enable_padding=False,
            bin_loader=False,
            reset_seed_on_access=False,
            rand_seed=0,
            rand_data_dist="uniform",
            rand_data_min=1,
            rand_data_max=1,
            rand_data_mu=-1,
            rand_data_sigma=1,
            data_directory="",
            from_dataset=True
    ):
        # compute batch size
        nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
        if num_batches != 0:
            nbatches = num_batches
            data_size = nbatches * mini_batch_size
            # print("Total number of batches %d" % nbatches)

        # save args (recompute data_size if needed)
        self.total_data_batches = 10 # Hard-coded for now
        self.m_den = m_den
        self.ln_emb = ln_emb
        self.data_size = data_size
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.num_indices_per_lookup_fixed = num_indices_per_lookup_fixed
        self.batched_or_fbgemm_emb = batched_or_fbgemm_emb
        self.num_targets = num_targets
        self.round_targets = round_targets
        self.data_generation = data_generation
        self.trace_file = trace_file
        self.enable_padding = enable_padding
        self.reset_seed_on_access = reset_seed_on_access
        self.rand_seed = rand_seed
        self.rand_data_dist = rand_data_dist
        self.rand_data_min = rand_data_min
        self.rand_data_max = rand_data_max
        self.rand_data_mu = rand_data_mu
        self.rand_data_sigma = rand_data_sigma
        self.n_spa = len(self.ln_emb)
        self.data_directory = data_directory
        self.bin = bin_loader # Borrow the mlperf_bin_loader flag
        self.from_dataset = from_dataset if not self.bin or num_indices_per_lookup_fixed else False # Only make the dataset and load from it if the num of indices per lookup is fixed (so that the tensors can be stacked)

        # Generate dataset and store for future use, instead of generating on the fly
        if self.from_dataset:
            if self.bin:
                self.random_data_file_suffix = '{}_{}_{}_{}_{}_{}.bin'.format(self.total_data_batches,
                                                                                self.m_den,
                                                                                '-'.join([str(x) for x in self.ln_emb]),
                                                                                self.mini_batch_size,
                                                                                self.num_indices_per_lookup,
                                                                                self.num_indices_per_lookup_fixed)
            else:
                self.random_data_file_suffix = '{}_{}_{}_{}_{}'.format(self.m_den,
                                                                        '-'.join([str(x) for x in self.ln_emb]),
                                                                        self.mini_batch_size,
                                                                        self.num_indices_per_lookup,
                                                                        self.num_indices_per_lookup_fixed)

            if self.data_directory == "":
                self.data_directory = "./"
            data_file = self.data_directory + '/' + self.random_data_file_suffix

            if self.bin:
                dataset_exist = path.exists(data_file)
            else:
                import os
                dataset_exist = True
                if not any(fname.startswith(self.random_data_file_suffix) for fname in os.listdir(self.data_directory)):
                    dataset_exist = False

            if not dataset_exist:
                # nbatches: num of batches
                # lX: list of dense features
                # lS_offsets/indices: list of list of offsets/indices
                # lT: probability
                (nbatches, lX, lS_offsets, lS_indices, lT) = generate_random_data(  self.m_den,
                                                                                    self.ln_emb,
                                                                                    self.data_size,
                                                                                    self.total_data_batches,
                                                                                    self.mini_batch_size,
                                                                                    self.num_indices_per_lookup,
                                                                                    self.num_indices_per_lookup_fixed,
                                                                                    num_targets=self.num_targets,
                                                                                    round_targets=self.round_targets,
                                                                                    data_generation=self.data_generation,
                                                                                    trace_file=self.trace_file,
                                                                                    enable_padding=self.enable_padding)
                if self.bin:
                    np_data = []
                    for b in range(nbatches):
                        np_data.append(np.concatenate([
                            lT[b].reshape(-1, 1), # target comes first: (mini_batch_size, 1)
                            lX[b], # dense features: (mini_batch_size, dense_feature_count)
                            torch.stack(lS_offsets[b]).t().contiguous(), # offsets: (mini_batch_size, num_embedding_tables) -> [transpose]
                            torch.stack(lS_indices[b]).reshape(-1, self.mini_batch_size, self.num_indices_per_lookup).permute(1, 0, 2).reshape(self.mini_batch_size, -1).contiguous()
                        ], axis=1)) # indices: (num_embedding_tables, mini_batch_size * num_indices_per_lookup) -> (mini_batch_size, num_embedding_tables * num_indices_per_lookup), e.g. (8, 2048 * 100) -> (2048, 8 * 100)

                    with open(data_file, 'wb') as f:
                        f.write(np_data.tobytes())

                    # dataset
                    self.tot_fea = self.m_den + self.n_spa * (1 + self.num_indices_per_lookup) + 1
                    self.bytes_per_entry = (4 * self.tot_fea * self.mini_batch_size)
                    self.num_entries = self.num_batches

                    self.file = open(data_file, 'rb')
                else:
                    import extend_distributed as ext_dist
                    # Only rank 0 writes to avoid contention
                    if ext_dist.my_size <= 1 or ext_dist.my_local_rank == 0:
                        for b in range(nbatches):
                            fname = data_file + '_{}.hdf5'.format(b)
                            X = torch.log(lX[b].type(torch.float32) + 1)
                            lS_o = torch.stack(lS_offsets[b])
                            lS_i = torch.stack(lS_indices[b])
                            y = lT[b].view(-1, 1)

                            with h5py.File(fname, 'w') as hf:
                                _ = hf.create_dataset('X', data=X, shape=X.shape)
                                _ = hf.create_dataset('lS_o', data=lS_o, shape=lS_o.shape)
                                _ = hf.create_dataset('lS_i', data=lS_i, shape=lS_i.shape)
                                _ = hf.create_dataset('y', data=y, shape=y.shape)

    def reset_numpy_seed(self, numpy_rand_seed):
        np.random.seed(numpy_rand_seed)
        # torch.manual_seed(numpy_rand_seed)

    def _transform_features(self, X, offsets, indices, T):
        X_batch = torch.log(X.clone().detach().type(torch.float32) + 1)
        lS_o = offsets.clone().detach().type(torch.long).t() # transpose back to (mini_batch_size, num_embedding_tables)
        lS_i = indices.clone().detach().type(torch.long).reshape(self.mini_batch_size, -1, self.num_indices_per_lookup).permute(1, 0, 2).reshape(-1, self.mini_batch_size * self.num_indices_per_lookup) # e.g. (2048, 8 * 100) -> (8, 2048 * 100)
        T_batch = T.clone().detach().type(torch.float32).view(-1, 1)

        if self.batched_or_fbgemm_emb:
            indices = torch.cat([x.view(-1) for x in lS_i], dim=0).int()
            E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in lS_i]).tolist()
            offsets = torch.cat([x + y for x, y in zip(lS_o, E_offsets[:-1])] + [torch.tensor([E_offsets[-1]])], dim=0).int() # TODO: fix this
            lS_i = indices
            lS_o = offsets

        return X_batch, lS_o, lS_i, T_batch

    def __getitem__(self, index):
        with torch.autograd.profiler.record_function("module::get_batch_data"):
            if isinstance(index, slice):
                return [
                    self[idx] for idx in range(
                        index.start or 0, index.stop or len(self), index.step or 1
                    )
                ]

            if self.from_dataset:
                if self.bin:
                    self.file.seek((index % self.total_data_batches) * self.bytes_per_entry, 0)
                    raw_data = self.file.read(self.bytes_per_entry)
                    array = np.frombuffer(raw_data, dtype=np.float32)
                    tensor = torch.from_numpy(array).view((-1, self.tot_fea))

                    return self._transform_features(X=tensor[:, 1:(self.m_den+1)],
                                                    offsets=tensor[:, (self.m_den+1):(self.m_den+1+self.n_spa)],
                                                    indices=tensor[:, (self.m_den+1+self.n_spa):],
                                                    T=tensor[:, 0])
                else:
                    fname = self.data_directory + '/' + self.random_data_file_suffix + '_{}.hdf5'.format(index % self.total_data_batches)
                    with h5py.File(fname, 'r') as hf:
                        X = torch.tensor(np.array(hf['X']))
                        lS_o = torch.tensor(np.array(hf['lS_o']))
                        lS_i = torch.tensor(np.array(hf['lS_i']))
                        y = torch.tensor(np.array(hf['y']))

                    if self.batched_or_fbgemm_emb:
                        indices = torch.cat([x.view(-1) for x in lS_i], dim=0).int()
                        E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in lS_i]).tolist()
                        offsets = torch.cat([x + y for x, y in zip(lS_o, E_offsets[:-1])] + [torch.tensor([E_offsets[-1]])], dim=0).int() # TODO: fix this
                        lS_i = indices
                        lS_o = offsets

                    return X, lS_o, lS_i, y
            else:
                # WARNING: reset seed on access to first element
                # (e.g. if same random samples needed across epochs)
                if self.reset_seed_on_access and index == 0:
                    self.reset_numpy_seed(self.rand_seed)

                # number of data points in a batch
                n = min(self.mini_batch_size, self.data_size - (index * self.mini_batch_size))

                # generate a batch of dense and sparse features
                if self.data_generation == "random":
                    (X, lS_o, lS_i) = generate_dist_input_batch(
                        self.m_den,
                        self.ln_emb,
                        n,
                        self.num_indices_per_lookup,
                        self.num_indices_per_lookup_fixed,
                        rand_data_dist=self.rand_data_dist,
                        rand_data_min=self.rand_data_min,
                        rand_data_max=self.rand_data_max,
                        rand_data_mu=self.rand_data_mu,
                        rand_data_sigma=self.rand_data_sigma,
                    )
                elif self.data_generation == "synthetic":
                    (X, lS_o, lS_i) = generate_synthetic_input_batch(
                        self.m_den,
                        self.ln_emb,
                        n,
                        self.num_indices_per_lookup,
                        self.num_indices_per_lookup_fixed,
                        self.trace_file,
                        self.enable_padding
                    )
                else:
                    sys.exit(
                        "ERROR: --data-generation=" + self.data_generation + " is not supported"
                    )

                # generate a batch of target (probability of a click)
                T = generate_random_output_batch(n, self.num_targets, self.round_targets)

                if self.batched_or_fbgemm_emb:
                    # lS_i: List of T tensors with size <= B * L, each of which contains concatenated indices segments with variable lengths, as L is never fixed
                    # -> indices (T * B * L)
                    # lS_o: List of T tensors with size (B), each of which contains B offsets with variable difference between each consecutive pair of them, as L is never fixed
                    # -> offsets (T * B + 1)
                    indices = torch.cat([x.view(-1) for x in lS_i], dim=0).int()
                    E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in lS_i]).tolist()
                    offsets = torch.cat([x + y for x, y in zip(lS_o, E_offsets[:-1])] + [torch.tensor([E_offsets[-1]])], dim=0).int() # TODO: fix this
                    lS_i = indices
                    lS_o = offsets

        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


def collate_wrapper_random_offset(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o) if isinstance(lS_o, list) else lS_o,
            lS_i,
            T)


def collate_wrapper_random_length(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            offset_to_length_converter(torch.stack(lS_o), lS_i),
            lS_i,
            T)


def make_random_data_and_loader(args, ln_emb, m_den,
    offset_to_length_converter=False,
):
    if not path.exists(args.processed_data_file):
        os.makedirs(args.processed_data_file)

    train_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        args.batched_emb or args.fbgemm_emb,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        args.mlperf_bin_loader,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed,
        data_directory=args.processed_data_file
    )  # WARNING: generates a batch of lookups at once

    test_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        args.batched_emb or args.fbgemm_emb,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed,
        data_directory=args.processed_data_file
    )

    collate_wrapper_random = collate_wrapper_random_offset
    if offset_to_length_converter:
        collate_wrapper_random = collate_wrapper_random_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=args.pin_memory,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_data, train_loader, test_data, test_loader


class ProcessedDataset(Dataset):
    def __init__(
            self,
            processed_data_file,
            total_num_batches,
            batched_or_fbgemm_emb
    ):
        self.total_num_batches = total_num_batches
        self.batched_or_fbgemm_emb = batched_or_fbgemm_emb
        data = torch.load(os.path.join(processed_data_file, "data.pt"))
        self.nbatches = data["nbatches"]
        self.lX = data["lX"]
        self.lS_offsets = data["lS_offsets"]
        self.lS_indices = data["lS_indices"]
        self.lT = data["lT"]

    def __getitem__(self, index):
        with torch.autograd.profiler.record_function("module::get_batch_data"):
            if isinstance(index, slice):
                return [
                    self[idx] for idx in range(
                        index.start or 0, index.stop or len(self), index.step or 1
                    )
                ]

            X = self.lX[index % self.nbatches]
            lS_o = self.lS_offsets[index % self.nbatches]
            lS_i = self.lS_indices[index % self.nbatches]
            T = self.lT[index % self.nbatches]

            # if self.batched_emb:
            #     indices = torch.cat([x.view(-1) for x in lS_i], dim=0).int()
            #     E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in lS_i]).tolist()
            #     offsets = torch.cat([x + y for x, y in zip(lS_o, E_offsets[:-1])] + [torch.tensor([E_offsets[-1]])], dim=0).int() # TODO: fix this
            #     lS_i = indices
            #     lS_o = offsets

            return X, lS_o, lS_i, T

    def __len__(self):
        return self.total_num_batches

def make_processed_data_and_loader(args):
    train_data = ProcessedDataset(
        args.processed_data_file,
        args.num_batches,
        args.batched_emb or args.fbgemm_emb
    )

    test_data = ProcessedDataset(
        args.processed_data_file,
        args.num_batches,
        args.batched_emb or args.fbgemm_emb
    )

    collate_wrapper_random = collate_wrapper_random_offset

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=args.pin_memory,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_data, train_loader, test_data, test_loader


def generate_random_data(
    m_den,
    ln_emb,
    data_size,
    num_batches,
    mini_batch_size,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    num_targets=1,
    round_targets=False,
    data_generation="random",
    trace_file="",
    enable_padding=False,
    length=False, # length for caffe2 version (except dlrm_s_caffe2)
):
    nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
    if num_batches != 0:
        nbatches = num_batches
        data_size = nbatches * mini_batch_size
    # print("Total number of batches %d" % nbatches)

    # inputs
    lT = []
    lX = []
    lS_offsets = []
    lS_indices = []
    for j in range(0, nbatches):
        # number of data points in a batch
        n = min(mini_batch_size, data_size - (j * mini_batch_size))

        # generate a batch of dense and sparse features
        if data_generation == "random":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_uniform_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                length,
            )
        elif data_generation == "synthetic":
            (Xt, lS_emb_offsets, lS_emb_indices) = generate_synthetic_input_batch(
                m_den,
                ln_emb,
                n,
                num_indices_per_lookup,
                num_indices_per_lookup_fixed,
                trace_file,
                enable_padding
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + data_generation + " is not supported"
            )
        # dense feature
        lX.append(Xt)
        # sparse feature (sparse indices)
        lS_offsets.append(lS_emb_offsets)
        lS_indices.append(lS_emb_indices)

        # generate a batch of target (probability of a click)
        P = generate_random_output_batch(n, num_targets, round_targets)
        lT.append(P)

    return (nbatches, lX, lS_offsets, lS_indices, lT)


def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)


# uniform ditribution (input data)
def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    length,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
                # sparse indices to be used per embedding, loop until there are no duplicates so that the num of sparse indices is acutally FIXED
                while True:
                    r = ra.random(sparse_group_size)
                    sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
                    if sparse_group.size == num_indices_per_lookup:
                        break
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
                # sparse indices to be used per embedding
                r = ra.random(sparse_group_size)
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
                # reset sparse_group_size in case some index duplicates were removed
                sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            if length: # for caffe2 version
                lS_batch_offsets += [sparse_group_size]
            else:
                lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


# random data from uniform or gaussian ditribution (input data)
def generate_dist_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    rand_data_dist,
    rand_data_min,
    rand_data_max,
    rand_data_mu,
    rand_data_sigma,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            if rand_data_dist == "gaussian":
                if rand_data_mu == -1:
                    rand_data_mu = (rand_data_max + rand_data_min) / 2.0
                r = ra.normal(rand_data_mu, rand_data_sigma, sparse_group_size)
                sparse_group = np.clip(r, rand_data_min, rand_data_max)
                sparse_group = np.unique(sparse_group).astype(np.int64)
            elif rand_data_dist == "uniform":
                r = ra.random(sparse_group_size)
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            else:
                raise(rand_data_dist, "distribution is not supported. \
                     please select uniform or gaussian")

            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # print(len(lS_batch_offsets))
            # print(len(lS_batch_indices))
            # update offset for next iteration
            offset += sparse_group_size
        # print("-------")
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))
        # print(len(torch.tensor(lS_batch_offsets)))
        # print(len(torch.tensor(lS_batch_indices)))

    return (Xt, lS_emb_offsets, lS_emb_indices)


# synthetic distribution (input data)
def generate_synthetic_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    trace_file,
    enable_padding=False,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i, size in enumerate(ln_emb):
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            file_path = trace_file
            line_accesses, list_sd, cumm_sd = read_dist_from_file(
                file_path.replace("j", str(i))
            )
            # debug prints
            # print("input")
            # print(line_accesses); print(list_sd); print(cumm_sd);
            # print(sparse_group_size)
            # approach 1: rand
            # r = trace_generate_rand(
            #     line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            # )
            # approach 2: lru
            r = trace_generate_lru(
                line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            )
            # WARNING: if the distribution in the file is not consistent
            # with embedding table dimensions, below mod guards against out
            # of range access
            sparse_group = np.unique(r).astype(np.int64)
            minsg = np.min(sparse_group)
            maxsg = np.max(sparse_group)
            if (minsg < 0) or (size <= maxsg):
                print(
                    "WARNING: distribution is inconsistent with embedding "
                    + "table size (using mod to recover and continue)"
                )
                sparse_group = np.mod(sparse_group, size).astype(np.int64)
            # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
    u = ra.rand(1)
    if i < max_i:
        # only generate stack distances up to the number of new references seen so far
        j = bisect.bisect(cumm_val, i) - 1
        fi = cumm_dist[j]
        u *= fi  # shrink distribution support to exclude last values
    elif enable_padding:
        # WARNING: disable generation of new references (once all have been seen)
        fi = cumm_dist[0]
        u = (1.0 - fi) * u + fi  # remap distribution support to exclude first value

    for (j, f) in enumerate(cumm_dist):
        if u <= f:
            return cumm_val[j]


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1


def trace_generate_lru(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)
    i = 0
    ztrace = deque()
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0

        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses[0]
            del line_accesses[0]
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            del line_accesses[l - sd]
            line_accesses.append(line_ref)
        # save generated memory reference
        ztrace.append(mem_ref)

    return ztrace


def trace_generate_rand(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)  # !!!Unique,
    i = 0
    ztrace = []
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0
        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
        ztrace.append(mem_ref)

    return ztrace


def trace_profile(trace, enable_padding=False):
    # number of elements in the array (assuming 1D)
    # n = trace.size

    rstack = deque()  # S
    stack_distances = deque()  # SDS
    line_accesses = deque()  # L
    for x in trace:
        r = np.uint64(x / cache_line_size)
        l = len(rstack)
        try:  # found #
            i = rstack.index(r)
            # WARNING: I believe below is the correct depth in terms of meaning of the
            #          algorithm, but that is not what seems to be in the paper alg.
            #          -1 can be subtracted if we defined the distance between
            #          consecutive accesses (e.g. r, r) as 0 rather than 1.
            sd = l - i  # - 1
            # push r to the end of stack_distances
            stack_distances.appendleft(sd)
            # remove r from its position and insert to the top of stack
            del rstack[i]  # rstack.remove(r)
            rstack.append(r)
        except ValueError:  # not found #
            sd = 0  # -1
            # push r to the end of stack_distances/line_accesses
            stack_distances.appendleft(sd)
            line_accesses.appendleft(r)
            # push r to the top of stack
            rstack.append(r)

    if enable_padding:
        # WARNING: notice that as the ratio between the number of samples (l)
        # and cardinality (c) of a sample increases the probability of
        # generating a sample gets smaller and smaller because there are
        # few new samples compared to repeated samples. This means that for a
        # long trace with relatively small cardinality it will take longer to
        # generate all new samples and therefore obtain full distribution support
        # and hence it takes longer for distribution to resemble the original.
        # Therefore, we may pad the number of new samples to be on par with
        # average number of samples l/c artificially.
        l = len(stack_distances)
        c = max(stack_distances)
        padding = int(np.ceil(l / c))
        stack_distances = stack_distances + [0] * padding

    return (rstack, stack_distances, line_accesses)


# auxiliary read/write routines
def read_trace_from_file(file_path):
    try:
        with open(file_path) as f:
            if args.trace_file_binary_type:
                array = np.fromfile(f, dtype=np.uint64)
                trace = array.astype(np.uint64).tolist()
            else:
                line = f.readline()
                trace = list(map(lambda x: np.uint64(x), line.split(", ")))
            return trace
    except Exception:
        print(f"ERROR: trace file '{file_path}' is not available.")


def write_trace_to_file(file_path, trace):
    try:
        if args.trace_file_binary_type:
            with open(file_path, "wb+") as f:
                np.array(trace).astype(np.uint64).tofile(f)
        else:
            with open(file_path, "w+") as f:
                s = str(list(trace))
                f.write(s[1 : len(s) - 1])
    except Exception:
        print("ERROR: no output trace file has been provided")


def read_dist_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
    except Exception:
        print("{file_path} Wrong file or file path")
    # read unique accesses
    unique_accesses = [int(el) for el in lines[0].split(", ")]
    # read cumulative distribution (elements are passed as two separate lists)
    list_sd = [int(el) for el in lines[1].split(", ")]
    cumm_sd = [float(el) for el in lines[2].split(", ")]

    return unique_accesses, list_sd, cumm_sd


def write_dist_to_file(file_path, unique_accesses, list_sd, cumm_sd):
    try:
        with open(file_path, "w") as f:
            # unique_acesses
            s = str(list(unique_accesses))
            f.write(s[1 : len(s) - 1] + "\n")
            # list_sd
            s = str(list_sd)
            f.write(s[1 : len(s) - 1] + "\n")
            # cumm_sd
            s = str(list(cumm_sd))
            f.write(s[1 : len(s) - 1] + "\n")
    except Exception:
        print("Wrong file or file path")


if __name__ == "__main__":
    import operator
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Generate Synthetic Distributions")
    parser.add_argument("--trace-file", type=str, default="./input/trace.log")
    parser.add_argument("--trace-file-binary-type", type=bool, default=False)
    parser.add_argument("--trace-enable-padding", type=bool, default=False)
    parser.add_argument("--dist-file", type=str, default="./input/dist.log")
    parser.add_argument(
        "--synthetic-file", type=str, default="./input/trace_synthetic.log"
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--print-precision", type=int, default=5)
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    ### read trace ###
    trace = read_trace_from_file(args.trace_file)
    # print(trace)

    ### profile trace ###
    (_, stack_distances, line_accesses) = trace_profile(
        trace, args.trace_enable_padding
    )
    stack_distances.reverse()
    line_accesses.reverse()
    # print(line_accesses)
    # print(stack_distances)

    ### compute probability distribution ###
    # count items
    l = len(stack_distances)
    dc = sorted(
        collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
    )

    # create a distribution
    list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    dist_sd = list(
        map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
    )  # k = tuple_x_k[1]
    cumm_sd = deque()  # np.cumsum(dc).tolist() #prefixsum
    for i, (_, k) in enumerate(dc):
        if i == 0:
            cumm_sd.append(k / float(l))
        else:
            # add the 2nd element of the i-th tuple in the dist_sd list
            cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    ### write stack_distance and line_accesses to a file ###
    write_dist_to_file(args.dist_file, line_accesses, list_sd, cumm_sd)

    ### generate corresponding synthetic ###
    # line_accesses, list_sd, cumm_sd = read_dist_from_file(args.dist_file)
    synthetic_trace = trace_generate_lru(
        line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    )
    # synthetic_trace = trace_generate_rand(
    #     line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    # )
    write_trace_to_file(args.synthetic_file, synthetic_trace)
