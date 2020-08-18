# From Interval example docs - typing
from numba import types
from numba.core.extending import typeof_impl, type_callable

from numba.core import typing

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

import math


# Tutorial code

class Quaternion(object):
    """
    A quaternion. Not to be taken as an exemplar API for a quaternion!
    For Numba extension example purposes only.

    A quaternion is: a + bi + cj + dk
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @property
    def phi(self):
        """The angle between the x-axis and the N-axis"""
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        return math.atan(2 * (a * b + c * d) / (a * a - b * b - c * c + d * d))

    @property
    def theta(self):
        """The angle between the z-axis and the Z-axis"""
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        return -math.asin(2 * (b * d - a * c))

    @property
    def psi(self):
        """The angle between the N-axis and the X-axis"""
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        return math.atan(2 * (a * d + b * c) / (a * a + b * b - c * c - d * d))


class QuaternionType(types.Type):
    def __init__(self):
        super().__init__(name='Quaternion')


quaternion_type = QuaternionType()


@typeof_impl.register(Quaternion)
def typeof_interval(val, c):
    return quaternion_type


@type_callable(Quaternion)
def type_interval(context):
    def typer(a, b, c, d):
        if (isinstance(a, types.Float) and isinstance(b, types.Float)
                and isinstance(c, types.Float) and isinstance(d, types.Float)):
            return quaternion_type
    return typer


@register_model(QuaternionType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('a', types.float64),
            ('b', types.float64),
            ('c', types.float64),
            ('d', types.float64),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@lower_builtin(Quaternion, types.Float, types.Float, types.Float, types.Float)
def impl_quaternion(context, builder, sig, args):
    typ = sig.return_type
    a, b, c, d = args
    quaternion = cgutils.create_struct_proxy(typ)(context, builder)
    quaternion.a = a
    quaternion.b = b
    quaternion.c = c
    quaternion.d = d
    return quaternion._getvalue()


make_attribute_wrapper(QuaternionType, 'a', 'a')
make_attribute_wrapper(QuaternionType, 'b', 'b')
make_attribute_wrapper(QuaternionType, 'c', 'c')
make_attribute_wrapper(QuaternionType, 'd', 'd')


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
    key = QuaternionType

    def resolve_phi(self, mod):
        return types.float64

    def resolve_theta(self, mod):
        return types.float64

    def resolve_psi(self, mod):
        return types.float64


@cuda_lower_attr(QuaternionType, 'phi')
def cuda_quaternion_phi(context, builder, sig, arg):
    # Computes math.atan(2 * (a * b + c * d) / (a * a - b * b - c * c + d * d))
    a = builder.extract_value(arg, 0)
    b = builder.extract_value(arg, 1)
    c = builder.extract_value(arg, 2)
    d = builder.extract_value(arg, 3)
    atan_sig = typing.signature(types.float64, types.float64)
    atan_impl = context.get_function(math.atan, atan_sig)

    a2 = builder.fmul(a, a)
    b2 = builder.fmul(b, b)
    c2 = builder.fmul(c, c)
    d2 = builder.fmul(d, d)

    numerator = builder.fadd(builder.fmul(a, b), builder.fmul(c, d))
    denominator = builder.fadd(builder.fsub(builder.fsub(a2, b2), c2), d2)
    atan_arg = builder.fmul(context.get_constant(types.float64, 2),
                            builder.fdiv(numerator, denominator))
    return atan_impl(builder, [atan_arg])


# Examples from the tutorial

@jit(nopython=True)
def inside_interval(interval, x):
    return interval.lo <= x < interval.hi


@jit(nopython=True)
def interval_width(interval):
    return interval.width


@jit(nopython=True)
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)



# User code

@cuda.jit
def kernel(arr):
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    arr[0] = q.phi

out = np.zeros(7)

kernel[1, 1](out)

# prints: [ 4.   2.   1.   0.   2.   8.5 12. ]
print(out)
print(Quaternion(1,2,3,4).phi)
