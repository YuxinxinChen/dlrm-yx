import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

NCCL_HOME = os.getenv('NCCL_HOME')
nccl_include_path = NCCL_HOME+'/include'
ext_modules = []

if torch.cuda.is_available():
    extension = CUDAExtension(
        'table_batched_embeddings_yx.batched_forward', 
        [ 'batched_forward.cpp', 'table_batched_embeddings_cuda_yx.cu'],
        #include_dirs=['/home/yuxin420/pkg/nccl/build/include'],
        include_dirs=[nccl_include_path],
        extra_compile_args={'nvcc': ['-O2']}
    )
    ext_modules.append(extension)

setup(
    name='table_batched_embeddings_yx',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=1)}
)