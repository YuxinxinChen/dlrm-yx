import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

NCCL_HOME = os.getenv('NCCL_HOME')
nccl_include_path = NCCL_HOME+'/include'
nccl_lib_path = NCCL_HOME+'/lib'
MPI_HOME = os.getenv('MPI_HOME')
mpi_include_path = MPI_HOME+'/include'
print(mpi_include_path)
ext_modules = []

if torch.cuda.is_available():
    extension = CUDAExtension(
        'sparse_embedding.forward',
        [ 'sparse_embedding_cuda.cpp','sparse_embedding_cuda_impl.cu'],
        include_dirs=[nccl_include_path, mpi_include_path],
        libraries=['nccl'],
        library_dirs=[nccl_lib_path],
        extra_compile_args={'nvcc': ['-O2']}
    )
    ext_modules.append(extension)

    extension = CUDAExtension(
        'sparse_embedding.all2all',
        [ 'sparse_embedding_cuda.cpp','sparse_embedding_cuda_impl.cu'],
        include_dirs=[nccl_include_path, mpi_include_path],
        libraries=['nccl'],
        library_dirs=[nccl_lib_path],
        extra_compile_args={'nvcc': ['-O2']}
    )
    ext_modules.append(extension)


setup(
    name='sparse_embedding',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=1)}
)
