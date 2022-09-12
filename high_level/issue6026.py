from numba import cuda, config
import numpy as np

config.CUDA_WARN_ON_IMPLICIT_COPY = False
config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@cuda.jit(device=True)
def foo(val):
    return np.float64(val).view(np.int64)


@cuda.jit
def bar(v):
    print(foo(v))


bar[1, 1](10.)
cuda.synchronize()
