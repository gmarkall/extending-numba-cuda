from numba import cuda, config, types
from numba.extending import overload_method
import numpy as np

config.CUDA_WARN_ON_IMPLICIT_COPY = False
config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@overload_method(types.Array, 'sum', target='cuda')
def array_sum(arr):
    if arr.ndim != 1:
        # Only implement 1D for this quick example
        return None

    def sum_impl(arr):
        res = 0
        for i in range(len(arr)):
            res += arr[i]
        return res
    return sum_impl


@cuda.jit
def f(arr):
    print("Sum is", arr.sum())


f[1, 1](np.arange(5))
f[1, 1](np.arange(10))
cuda.synchronize()
