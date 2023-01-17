#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/record_function.h>
#include <ATen/SequenceNumber.h>
#include <c10/cuda/CUDAGuard.h>

#include "cub/device/device_radix_sort.cuh"
#include <pybind11/pybind11.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>
#include <vector>

using namespace at;
using namespace torch;

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))

//namespace {

static constexpr int32_t kWarpSize = 32;
static constexpr int32_t kMaxThreads = 1024;

template <typename T> struct Vec4T {};

struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(Half *p) {
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(p), "r"(__HALF2_TO_UI(a)), "r"(__HALF2_TO_UI(b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(a.x), "r"(b.x));
#endif
  }
};

template <> struct Vec4T<Half> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const Half *p) {
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(p));
#endif

    float2 a = __half22float2(out.a);
    float2 b = __half22float2(out.b);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
  }

  DEVICE_INLINE void store(Half *p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }
};

template <> struct Vec4T<float> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const float *p) { acc = *((const float4 *)p); }

  DEVICE_INLINE void store(float *p) { *((float4 *)p) = acc; }
};

template <> struct Vec4T<double> {
  double4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const double *p) { acc = *((const double4 *)p); }

  DEVICE_INLINE void store(double *p) { *((double4 *)p) = acc; }
};

template <typename T>
DEVICE_INLINE T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask);
  }
  return val;
}

static DEVICE_INLINE double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if ((CUDA_VERSION < 10000) ||                                                 \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  unsigned int *address_as_ui =
      (unsigned int *)((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  at::Half hsum;
  do {
    assumed = old;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return hsum;
#else
  return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}

template <typename scalar_t> struct UnweightedForward {
  DEVICE_INLINE void accumulate(Vec4T<scalar_t> &sum, Vec4T<scalar_t> weight,
                                int32_t indices_offset) {
    sum.acc.x += weight.acc.x;
    sum.acc.y += weight.acc.y;
    sum.acc.z += weight.acc.z;
    sum.acc.w += weight.acc.w;
  }
};

template <typename scalar_t, bool shared_indices, typename F>
__global__ void batched_embedding_forward_kernel_1(
    // [\sum_t E_t][D]
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [T][B][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [T x B + 1]
    // offsets = cumsum([0] + lengths.contiguous()), where lengths L is [T][.
    PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits>
        output, // [B][T][D],
    int32_t L_max,
    F f) {

  extern __shared__ int32_t shmem_indices[];

  const int32_t B = output.size(0);
  const int32_t T = output.size(1);
  const int32_t D = output.size(2);

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_indices) {
    int32_t shmem_offset = threadIdx.y * L_max;
    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        Vec4T<scalar_t> weight((&weights[table_offset + idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = __ldg(&indices[indices_start + l]);
        Vec4T<scalar_t> weight((&weights[table_offset + idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  }
}

template<typename scalar_t, bool shared_indices, typename F>
__global__ void batched_embedding_forward_kernel_2(
  const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
  PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> output,
  int32_t L_max,
  F f) {
  if(threadIdx.x + blockIdx.x*blockDim.x== 0) {
    printf("hello from tensor:\n");
    printf("%f, %d, %d, %d, %f\n", 
      weights[0][0], table_offsets[0], indices[0], offsets[0], output[0][0][0]);
    printf("finish!\n");
  }
}

std::vector<Tensor> batched_embedding_forward_cuda(TensorList weights, 
                                    TensorList table_offsets,
                                    TensorList offsets,
                                    TensorList indices,
                                    int64_t L_max,
                                    int64_t BT_block_size,
                                    bool shmem) {
  int num_devices = weights.size();
  auto output_vec = std::vector<Tensor>();                     
  for(int dev_id = 0; dev_id < num_devices; dev_id++)
  {
    int reside_device = weights[dev_id].get_device();
    const auto T = table_offsets[dev_id].size(0);
    const auto D = weights[dev_id].size(1);
    AT_ASSERT(dev_id == reside_device);
    AT_ASSERT(T > 0);
    AT_ASSERT(D > 0);
    const auto B = (offsets[dev_id].size(0)-1)/T;
    AT_ASSERT(B > 0);
    printf("weights %d, reside on device %d\nT=%ld, D=%ld,B=%ld\n", dev_id, reside_device, T, D, B);

    AT_ASSERT(BT_block_size != 0);
    if ((B * T) % BT_block_size != 0) {
      BT_block_size = 1;
    }
    AT_ASSERT((B * T) % BT_block_size == 0);
    AT_ASSERT(D % 4 == 0);
    const dim3 threads(std::min(D/4, kMaxThreads/BT_block_size), BT_block_size);
    const dim3 blocks((B*T)/BT_block_size);
    
    cudaSetDevice(reside_device);
    auto tmp_output = empty({B, T, D}, weights[dev_id].options());
    output_vec.push_back(tmp_output);
    Device device(kCUDA, reside_device);
    Tensor input_table_offsets = table_offsets[dev_id].to(device);
    Tensor input_indices = indices[dev_id].to(device);
    Tensor input_offsets = offsets[dev_id].to(device);
    std::cout << weights[dev_id].options() << std::endl;
    std::cout << input_table_offsets.options() << std::endl;
    std::cout << input_indices.options() << std::endl;
    std::cout << input_offsets.options() << std::endl;
    std::cout << output_vec[dev_id].options() << std::endl;
    AT_DISPATCH_FLOATING_TYPES(weights[dev_id].type(), "hello_cuda", 
     ([&] {
        //batched_embedding_forward_kernel_1<scalar_t, false><<<blocks, threads, 0>>>
	       //(weights[dev_id]).packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
        // (table_offsets[dev_id]).packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
        // (indices[dev_id]).packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
        // (offsets[dev_id]).packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
        // output.packed_accessor32<scalar_t, 3, UnweightedForward>(),
        // static_cast<int32_t>(L_max), UnweightedForward<scalar_t>() 
        //);
        batched_embedding_forward_kernel_1<scalar_t, false><<<blocks, threads>>>
	      ((weights[dev_id]).packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
         input_table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
         input_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
         input_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
         output_vec[dev_id].packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
         static_cast<int32_t>(L_max), UnweightedForward<scalar_t>()
        );
    }));
    //AT_DISPATCH_FLOATING_TYPES(weights[dev_id].type(), 
    //  'kernel',
    //  ([&] {
    //    batched_embedding_forward_kernel_1<scalar_t, shmem><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>
    //    (weights[dev_id].packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
    //    table_offsets[dev_id].packed_accessor32<int, 1, RestrictPtrTraits>(),
    //    indices[dev_id].packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
    //    offsets[dev_id].packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
    //    static_cast<int32_t>(L_max), UnweightedForward<scalar_t>());
    //  })
    //);
    //AT_CUDA_CHECK(cudaGetLastError());
  }
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  cudaSetDevice(1);
  cudaDeviceSynchronize();
  return output_vec;
}

/*
static void CUDAManagedDeleter(void *ptr) { AT_CUDA_CHECK(cudaFree(ptr)); }

struct CUDAManagedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    AT_CUDA_CHECK(cudaMallocManaged(&ptr, size));
    AT_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
                                cudaCpuDeviceId));
    AT_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy,
                                at::cuda::current_device()));
    return {ptr,
            ptr,
            &CUDAManagedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override { return &CUDAManagedDeleter; }
};

static CUDAManagedAllocator g_managed_allocator;

static void CUDAHostMappedDeleter(void *ptr) {
  AT_CUDA_CHECK(cudaFreeHost(ptr));
}

struct CUDAHostMappedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    AT_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined |
                                                cudaHostAllocMapped));

    void *dev_ptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, ptr, 0));
    return {dev_ptr,
            ptr,
            &CUDAHostMappedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &CUDAHostMappedDeleter;
  }
};

static CUDAHostMappedAllocator g_host_mapped_allocator;

std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }
  return strides;
}

int64_t computeStorageSize(IntArrayRef sizes, IntArrayRef strides) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return size;
}

Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  auto strides = defaultStrides(sizes);
  auto storage = Storage(Storage::use_byte_size_t(), computeStorageSize(sizes, strides),
                         &g_managed_allocator,
                         resizable=false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

Tensor new_host_mapped_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  auto strides = defaultStrides(sizes);
  auto storage = Storage(Storage::use_byte_size_t(), computeStorageSize(sizes, strides),
                         &g_host_mapped_allocator,
                         resizable=false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

__global__ void construct_offsets_kernel(
    const PackedTensorAccessor32<int32_t, 2, RestrictPtrTraits>
        batch_offsets_per_table,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        total_indices_per_table,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> output) {
  const int32_t T = batch_offsets_per_table.size(0);
  const int32_t B = batch_offsets_per_table.size(1);

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b = b_t % B;
  int32_t t = b_t / B;
  if (t < T) {
    int32_t upper = 0;
    if (b != B - 1) {
      upper = batch_offsets_per_table[t][b + 1];
    } else {
      upper = total_indices_per_table[t];
    }
    int32_t lower = batch_offsets_per_table[t][b];
    output[1 + t * B + b] = upper - lower;
  }
}

Tensor construct_offsets(Tensor batch_offsets_per_table, // [T][B]
                         Tensor total_indices_per_table  // [T]
                         ) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(batch_offsets_per_table.get_device());

  const auto T = batch_offsets_per_table.size(0);
  const auto B = batch_offsets_per_table.size(1);
  const dim3 threads(kMaxThreads);
  const dim3 blocks((B * T + 1 + kMaxThreads - 1) / kMaxThreads);

  Tensor output =
      at::zeros({1 + B * T}, batch_offsets_per_table.options().dtype(kInt));

  construct_offsets_kernel<<<blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      batch_offsets_per_table
          .packed_accessor32<int32_t, 2, RestrictPtrTraits>(),
      total_indices_per_table
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      output.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

struct IndexInfo {
  int32_t b_t;
  int32_t indices_idx;
};
*/