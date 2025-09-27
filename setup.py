# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import platform

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This package requires CUDA.")

# Platform-specific compiler arguments
def get_compiler_args():
    system = platform.system()
    
    if system == "Windows":
        cxx_args = ['/MD', '/std:c++17', '/DWITH_CUDA']
        nvcc_args = ['-O3', '--use_fast_math', '-lineinfo', '--std=c++17']
    elif system == "Linux":
        cxx_args = ['-O3', '-std=c++17', '-DWITH_CUDA']
        nvcc_args = ['-O3', '--use_fast_math', '-lineinfo', '--std=c++17']
    elif system == "Darwin":  # macOS
        cxx_args = ['-O3', '-std=c++17', '-DWITH_CUDA']
        nvcc_args = ['-O3', '--use_fast_math', '-lineinfo', '--std=c++17']
    else:
        # Default fallback
        cxx_args = ['-O3', '-std=c++17', '-DWITH_CUDA']
        nvcc_args = ['-O3', '--use_fast_math', '-lineinfo', '--std=c++17']
    
    return {'cxx': cxx_args, 'nvcc': nvcc_args}

# Optionally set TORCH_CUDA_ARCH_LIST to restrict architectures, e.g. "8.6;8.0"
# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"  # adjust for your GPU if desired

setup(
    name='emd_ext',
    version='1.0.0',
    description='Earth Mover Distance (EMD) CUDA extension for PyTorch',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Haoqiang Fan, Kaichun Mo, Jiayuan Gu',
    maintainer='hieulhaiwork',
    url='https://github.com/hieulhaiwork/EMD-Pytorch',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'numpy',
    ],
    python_requires='>=3.7',
    ext_modules=[
        CUDAExtension(
            name='emd_cuda',
            sources=[
                'emd/cuda/emd.cpp',
                'emd/cuda/emd_kernel.cu',
            ],
            extra_compile_args=get_compiler_args()
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
    ],
)