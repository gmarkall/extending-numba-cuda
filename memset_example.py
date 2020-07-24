# From Interval example docs - typing
from numba import types
from numba.core.extending import typeof_impl, type_callable

# Data model

from numba.core.extending import models, register_model

# Lowering

from numba.core.extending import lower_builtin
from numba.core import cgutils


from llvmlite import ir

# User CUDA + test code imports

from numba import cuda


# Tutorial code

class Interval(object):
    def __init__(self):
        self.lo = 0.0
        self.hi = 0.0


class IntervalType(types.Type):
    def __init__(self):
        super().__init__(name='Interval')


interval_type = IntervalType()


@typeof_impl.register(Interval)
def typeof_interval(val, c):
    return interval_type


@type_callable(Interval)
def type_interval(context):
    def typer():
        return interval_type
    return typer


@register_model(IntervalType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('lo', types.float64),
            ('hi', types.float64),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@lower_builtin(Interval)
def impl_interval(context, builder, sig, args):
    dstty = ir.LiteralStructType([ir.DoubleType(), ir.DoubleType()])
    dst_ptr = cgutils.alloca_once(builder, dstty)
    dst_ptr.align = 4
    cgutils.memset(builder, dst_ptr, context.get_constant(types.uint32, 8), 0)
    return builder.load(dst_ptr)


# User code

@cuda.jit
def kernel():
    Interval()

kernel[1, 1]()

