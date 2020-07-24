# From Interval example docs - typing
from numba import types
from numba.core.extending import typeof_impl, type_callable

# Data model

from numba.core.extending import models, register_model

# Lowering

from numba.core.extending import lower_builtin, make_attribute_wrapper
from numba.core import cgutils
from numba.extending import unbox, NativeValue

# Specific to CUDA extension
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr
from numba.core.typing.templates import AttributeTemplate

# User CUDA + test code imports

from numba import cuda, jit
import numpy as np


# Tutorial code

class Interval(object):
    """
    A half-open interval on the real number line.
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


class IntervalType(types.Type):
    def __init__(self):
        super().__init__(name='Interval')


interval_type = IntervalType()


@typeof_impl.register(Interval)
def typeof_interval(val, c):
    return interval_type


@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
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


@lower_builtin(Interval, types.Float, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    dstty = context.get_value_type(types.uint8)
    dst_ptr = cgutils.alloca_once(builder, dstty)
    dst = builder.bitcast(dst_ptr, dstty.as_pointer())
    dst.align = 4
    cgutils.memset(builder, dst, context.get_constant(types.uint8, 8), 0)
    return interval._getvalue()

# User code

@cuda.jit
def kernel():
    x = Interval(1.0, 3.0)



kernel[1, 1]()

