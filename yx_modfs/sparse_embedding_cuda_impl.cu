#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/cuda/device_set.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
//#include <ATen/cuda/Exceptions.h>
#include "mpi.h"

#include <torch/csrc/cuda/nccl.h>
#include <nccl.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

constexpr int L_max = 200;

static std::tuple<MPI_Comm*, int, int> mpi_comm() {
  static std::once_flag once;
  static MPI_Comm world_comm;
  static int world_size;
  static int world_rank;

  std::call_once(once, [&] {
    {
      auto op = MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
      AT_ASSERT(op == MPI_SUCCESS);
    }
    {
      auto op = MPI_Comm_size(world_comm, &world_size);
      AT_ASSERT(op == MPI_SUCCESS);
    }
    {
      auto op = MPI_Comm_rank(world_comm, &world_rank);
      AT_ASSERT(op == MPI_SUCCESS);
    }
  });
  return std::make_tuple(&world_comm, world_size, world_rank);
}

ncclDataType_t get_data_type(const at::Tensor& t) {
  if (t.type().backend() != at::Backend::CUDA) {
    throw std::runtime_error("Unconvertible NCCL type");
  }
  switch (t.scalar_type()) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclChar;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

static ncclComm_t nccl_comm() {
  //using namespace torch::cuda::nccl::detail;
  static std::once_flag once;
  static ncclComm_t world_comm;

  std::call_once(once, [&] {
    auto comm_size_rank = mpi_comm();
    int my_rank = std::get<2>(comm_size_rank);

    ncclUniqueId id;
    // generating NCCL unique ID at one process and broadcasting it to all
    if (my_rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&id));
    }
    {
      auto op = MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0,
                          *(std::get<0>(comm_size_rank)));
      AT_ASSERT(op == MPI_SUCCESS);
    }

    //ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    //config.blocking = 0;
    // Initialize the communicator for the current rank
    NCCLCHECK(ncclGroupStart()); // not sure if it is needed
    NCCLCHECK(
        ncclCommInitRank(&world_comm, std::get<1>(comm_size_rank), id, my_rank));
    
    //ncclCommInitRankConfig(&world_comm, std::get<1>(comm_size_rank), id, my_rank, &config);
    //ncclResult_t state;
    //do {
    //  NCCLCHECK(ncclCommGetAsyncError(world_comm, &state));
    //  // Handle outside events, timeouts, progress, ...
    //} while(state == ncclInProgress);
    NCCLCHECK(ncclGroupEnd());
  });
  return world_comm;
}

template <typename scalar_t, bool T_blocked>
__global__ void sparse_embedding_cuda_forward_offsets_kernel_impl(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits>
        indices, // [N = B x T total indices,]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits>
        offsets, // [B][T+1]
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output) {

  extern __shared__ int shmem_indices[];

  const int B = offsets.size(0);
  const int T_1 = offsets.size(1);
  const int T = T_1 - 1;
  const int D = weights.size(2);
  const int E = weights.size(0);
  //if(threadIdx.x ==0 && threadIdx.y == 0 && blockIdx.x==0 && blockIdx.y==0 ) {
  //  printf("thread (%d, %d), grid (%d, %d), T_block %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y, T_blocked);
  //  for(int i=0; i<B; i++) {
  //    for(int j=0; j<T; j++) {
  //      for(int k=0; k<3; k++)
  //        printf("(%d, %d, %d) -> %d\n",i,j,k, indices[i][j*3+k] );
  //    }
  //  }
  //}
  if (!T_blocked) {
    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;

    const int32_t indices_start = offsets[b][t];
    const int32_t indices_end = offsets[b][t+1];
    int L = indices_end - indices_start;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg(&(indices[b][indices_start + i]));
    }
    __syncthreads();

    at::acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
    for (int l = 0; l < L; ++l) {
      sum += __ldg((&weights[shmem_indices[l]][t][0]) + d);
    }
    output[b][t][d] = sum;
  } 
  else {
    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int L = 0;
    if (t < T) {
      const int32_t indices_start = offsets[b][t];
      const int32_t indices_end = offsets[b][t+1];
      L = indices_end - indices_start;
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        //printf("d %d, t_t %d, b %d, t_b %d, t %d, L %d, B %d, T %d, D %d, E %d, indices_start %d, indices_end %d, indices+i %d, value %d, write to %d\n", d, t_t, b, t_b, t, L,  B, T, D, E, indices_start, indices_end, indices_start+i, indices[b][indices_start + i], t_t*L_max+i);
        shmem_indices[t_t * L_max + i] = __ldg(&(indices[b][indices_start + i]));
      }
    }
    __syncthreads();
    if (t < T) {
      at::acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
      for (int l = 0; l < L; ++l) {
        sum += __ldg((&weights[shmem_indices[t_t * L_max + l]][t][0]) + d);
      }
      output[b][t][d] = sum;
    }
  }
}

torch::Tensor sparse_embedding_cuda_forward_offsets_kernel(
  // [E][T][D]
  torch::Tensor weights,
  // [N = \sum_B \sum_T L_{b, t}]
  torch::Tensor indices, 
  // [B][T+1]
  torch::Tensor offsets) {
  const auto B = offsets.size(0);
  const auto T_1 = offsets.size(1);
  const auto T = T_1 - 1;
  const auto D = weights.size(2);
  auto output = at::empty({B, T, D}, weights.options());
  //std::cout << offsets.options() << std::endl;
  if (D < 64) {
    const int T_t = std::min<int>(64 / D, 4);
    const int T_b = (T + T_t - 1) / T_t;
    const dim3 threads(D, T_t);
    const dim3 blocks(B, T_b);
    //printf("threads %d, %d, block %d, %d\n", threads.x, threads.y, blocks.x, blocks.y);

    AT_DISPATCH_FLOATING_TYPES(
        weights.type(), "sparse_embedding_cuda_forward_offsets", ([&] {
          sparse_embedding_cuda_forward_offsets_kernel_impl<
              scalar_t, true><<<blocks, threads, T_t * L_max * sizeof(int),
                                at::cuda::getCurrentCUDAStream()>>>(
              weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
              output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    const int threads = D;
    const dim3 blocks(B, T);
    //printf("threads %d, block %d, %d\n", threads, blocks.x, blocks.y);

    AT_DISPATCH_FLOATING_TYPES(
        weights.type(), "sparse_embedding_cuda_forward_offsets", ([&] {
          sparse_embedding_cuda_forward_offsets_kernel_impl<scalar_t, false><<<
              blocks, threads, L_max * sizeof(int), at::cuda::getCurrentCUDAStream()>>>(
              weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
              output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  return output;
}

at::Tensor sparse_embedding_cuda_forward_offsets(
    // [E][T][D]
    at::Tensor weights,
    // [\sum_{0 <= b < B, 0 <= t < T} L_{b, t}]
    at::Tensor indices,
    // [B][T+1]
    at::Tensor offsets
) {
  // -> [B][T][D]
  std::tuple<MPI_Comm*, int, int> meta = mpi_comm();
  const int rank = std::get<2>(meta);
  const int world_size = std::get<1>(meta);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  //printf("I am here Rank %d, world %d, device %d\n", rank, world_size, int(weights.get_device()));
  AT_CUDA_CHECK(cudaSetDevice(weights.get_device()));
  return sparse_embedding_cuda_forward_offsets_kernel(weights, indices, offsets);
}


at::Tensor sparse_embedding_cuda_forward_all2all_nccl(
  // [B][T // devices][D])
  at::Tensor embeddings) {

  // input: [B][T // devices][D]
  // reinterpret_input: [devices][B // devices][T // devices][D]
  // output: [B // devices][T][D]

  // at step w:
  // send input[w][B // devices][T // devices][D] to rank $w$
  // recv input[w][B // devices][T // devices][D] from rank $w$

  // now, after all-to-all, 
  // output is size [devices][B // devices][T // devices][D]
  // now, transpose to [B // devices][devices][T // devices][D]
  // now, view as [B // devices][devices * T // devices][D]
  // then, make contiguous.

  std::tuple<MPI_Comm*, int, int> meta = mpi_comm();
  const int rank = std::get<2>(meta);
  const int world_size = std::get<1>(meta);
  if(world_size == 1)
    return embeddings;
    //return;
  using namespace torch::cuda::nccl::detail;

  const auto B = embeddings.size(0);
  const auto T = embeddings.size(1) * world_size;
  const auto D = embeddings.size(2);
  at::cuda::CUDAGuard device_guard(embeddings.get_device());
  AT_ASSERT(B % world_size == 0);

  auto all_to_all_output = at::empty({world_size, B / world_size, T / world_size, D}, embeddings.options());

  AT_ASSERT(embeddings.is_contiguous());
  AT_ASSERT(all_to_all_output.is_contiguous());

  AT_ASSERT(embeddings.numel() == all_to_all_output.numel());
  AT_ASSERT(embeddings.scalar_type() == all_to_all_output.scalar_type());


  ncclDataType_t data_type = get_data_type(embeddings);
  int64_t count = embeddings.numel() / world_size;
  const auto rank_offset = count * ncclTypeSize(data_type);
  auto comm = nccl_comm();

  AT_ASSERT(count == B * T * D / world_size / world_size);
  check_inputs({embeddings}, {all_to_all_output}, 1, 1);
  cudaStream_t streams[world_size*2];
  for(int i=0; i<world_size*2; i++) {
    AT_CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
  }
  //auto stream =
  //    at::cuda::getCurrentCUDAStream(embeddings.get_device()).stream();
  {
    //pybind11::gil_scoped_release no_gil;
    //torch::cuda::nccl::AutoNcclGroup nccl_group_guard;
    ncclGroupStart();
    for (int r = 0; r < world_size; r++) {
      // send all tables $t$ from rank $i$ to global batch chunk $j$.
      // recieve all tables $t$ from rank $j$ for global batch chunk $i$.
      if (r == rank) 
        AT_CUDA_CHECK(cudaMemcpyAsync((uint8_t *)all_to_all_output.data_ptr()+r*rank_offset, (uint8_t *)embeddings.data_ptr()+r*rank_offset, rank_offset, cudaMemcpyDeviceToDevice, streams[r*2]));
      else {
        NCCLCHECK(ncclSend(((uint8_t*)embeddings.data_ptr()) + r * rank_offset, count, data_type, r, comm, streams[r*2]));
        NCCLCHECK(ncclRecv(((uint8_t*)all_to_all_output.data_ptr()) + r * rank_offset, count, data_type, r, comm, streams[r*2+1]));
      }
    }
    ncclGroupEnd();
  }
  for (int i=0; i<world_size*2; i++)
    AT_CUDA_CHECK(cudaStreamSynchronize(streams[i]));

  auto transposed = all_to_all_output.transpose(1, 0);
  return transposed.contiguous().view({B / world_size, T, D});
}