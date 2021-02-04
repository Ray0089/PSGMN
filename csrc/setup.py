from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='node_index_projection',
    ext_modules=[
        CUDAExtension('node_index_projection', [
            './src/node_index_project.cpp',
            './src/node_index_project_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
