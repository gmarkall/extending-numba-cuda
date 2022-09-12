from numba import cuda, config, types
import numpy as np

config.CUDA_WARN_ON_IMPLICIT_COPY = False
config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@cuda.jit
def use_view_scalar(x):
    print("As integer is ", x.view(np.int64))


#use_view_scalar[1, 1](1.5)
cuda.synchronize()


@cuda.jit
def use_view_array(x):
    v = x.view(np.int64)
    for i in range(len(v)):
        print("As integer value is", v[i])


use_view_array[1, 1](np.ones(2))
cuda.synchronize()
