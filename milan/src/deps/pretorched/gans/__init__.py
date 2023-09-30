from . import dcgan, sagan, sngan, utils
from .biggan import BigGAN
from .biggan_deep import BigGANDeep

# from .stylegan import stylegan
from .proggan import proggan

# try:
#     from .stylegan2 import stylegan2
# except EnvironmentError:
#     stylegan2 = None
#     print('Warning: could not compile cuda code for stylegan2')
#     print('Ensure $CUDA_HOME/bin/nvcc exists!')
stylegan = None
stylegan2 = None

__all__ = [
    "BigGAN",
    "BigGANDeep",
    "dcgan",
    "sngan",
    "sagan",
    "proggan",
    "stylegan",
    "stylegan2",
    "utils",
]
