from numba import cuda, config
from numba.extending import overload

config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@overload(len, target='cuda')
def grid_group_len(seq):
    if isinstance(seq, cuda.types.GridGroup):
        def len_impl(seq):
            n = cuda.gridsize(1)
            print("Length of group is", n)
            return n
        return len_impl


@cuda.jit
def f():
    if cuda.grid(1) == 0:
        len(cuda.cg.this_grid())


f[1, 1]()
f[1, 2]()
f[1, 3]()
cuda.synchronize()
