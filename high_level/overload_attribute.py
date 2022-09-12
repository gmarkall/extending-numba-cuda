from numba import cuda, config, types
from numba.extending import overload_attribute
import numpy as np

config.CUDA_WARN_ON_IMPLICIT_COPY = False
config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@overload_attribute(types.Array, 'nbytes', target='cuda')
def array_nbytes(arr):
    def nbytes_impl(arr):
        return arr.size * arr.itemsize
    return nbytes_impl


@cuda.jit
def f(arr):
    print("Nbytes is", arr.nbytes)


f[1, 1](np.arange(5))
f[1, 1](np.arange(10))
cuda.synchronize()
