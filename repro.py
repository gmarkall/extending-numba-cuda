# From Interval example docs - typing
from numba import types
from numba.core.extending import typeof_impl, type_callable

# Data model

from numba.core.extending import models, register_model

# Lowering

from numba.core.extending import lower_builtin, make_attribute_wrapper
from numba.core import cgutils

# Specific to CUDA extension
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr
from numba.core.typing.templates import AttributeTemplate, signature

# User CUDA + test code imports

from numba import cuda
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
    return interval._getvalue()


make_attribute_wrapper(IntervalType, 'lo', 'lo')
make_attribute_wrapper(IntervalType, 'hi', 'hi')


# From the tutorial - doesn't work due to:
#
# No definition for lowering <built-in method getter of _dynfunc._Closure
# object at 0x7ff6d564c4c0>(Interval,) -> float64

# @overload_attribute(IntervalType, "width")
# def get_width(interval):
#     def getter(interval):
#         return interval.hi - interval.lo
#     return getter

# Alternative:

@cuda_registry.register_attr
class Interval_attrs(AttributeTemplate):
    key = IntervalType

    def resolve_width(self, mod):
        return types.float64


@cuda_lower_attr(IntervalType, 'width')
def cuda_Interval_width(context, builder, sig, arg):
    lo = builder.extract_value(arg, 0)
    hi = builder.extract_value(arg, 1)
    return builder.fsub(hi, lo)


# User code

@cuda.jit
def width(arr):
    x = Interval(-2.0, 3.0)
    arr[0] = x.hi - x.lo
    arr[1] = x.width


out = np.zeros(2)

width[1, 1](out)

print(out)
