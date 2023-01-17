#include <torch/extension.h>
#include <iostream>

std::vector<torch::Tensor> batched_embedding_forward_cuda(torch::TensorList weights,
                                    torch::TensorList table_offsets,
                                    torch::TensorList offsets, torch::TensorList indices,
                                    int64_t L_max,
                                    int64_t BT_block_size,
                                    bool shmem);

TORCH_LIBRARY(batched_forward, m) {
    m.def("forward", &batched_embedding_forward_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){}