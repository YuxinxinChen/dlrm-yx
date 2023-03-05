#include <torch/extension.h>
#include <iostream>

at::Tensor sparse_embedding_cuda_forward_offsets(
    // [E][T][D]
    at::Tensor weights,
    // [\sum_{0 <= b < B, 0 <= t < T} L_{b, t}]
    at::Tensor indices,
    // [B][T+1]
    at::Tensor offsets
);

void sparse_embedding_cuda_forward_all2all_nccl(at::Tensor embeddings);

TORCH_LIBRARY(sparse_offset_forward, m) {
    m.def("forward", &sparse_embedding_cuda_forward_offsets);
    m.def("all2all", &sparse_embedding_cuda_forward_all2all_nccl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){}