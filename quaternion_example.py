# Typing

from numba import types
from numba.core.extending import typeof_impl, type_callable
from numba.core import typing

# Data model

from numba.core.extending import models, register_model

# Lowering

from numba.core.extending import lower_builtin, make_attribute_wrapper
from numba.core import cgutils

# Specific to CUDA extension
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr
from numba.core.typing.templates import AttributeTemplate

# For user CUDA + test code

from numba import cuda
import numpy as np

import math


# Python class to support in Numba

class Quaternion(object):
    """
    A quaternion. Not to be taken as an exemplar API or implementation of a
    quaternion!  For Numba extension example purposes only.

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


# Typing

class QuaternionType(types.Type):
    def __init__(self):
        super().__init__(name='Quaternion')


quaternion_type = QuaternionType()


@typeof_impl.register(Quaternion)
def typeof_quaternion(val, c):
    return quaternion_type


@type_callable(Quaternion)
def type_quaternion(context):
    def typer(a, b, c, d):
        if (isinstance(a, types.Float) and isinstance(b, types.Float)
                and isinstance(c, types.Float) and isinstance(d, types.Float)):
            return quaternion_type
    return typer


# Data model

@register_model(QuaternionType)
class QuaternionModel(models.StructModel):
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


@cuda_registry.register_attr
class Quaternion_attrs(AttributeTemplate):
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

    a2 = builder.fmul(a, a)
    b2 = builder.fmul(b, b)
    c2 = builder.fmul(c, c)
    d2 = builder.fmul(d, d)

    numerator = builder.fadd(builder.fmul(a, b), builder.fmul(c, d))
    denominator = builder.fadd(builder.fsub(builder.fsub(a2, b2), c2), d2)

    atan_sig = typing.signature(types.float64, types.float64)
    atan_impl = context.get_function(math.atan, atan_sig)
    atan_arg = builder.fmul(context.get_constant(types.float64, 2),
                            builder.fdiv(numerator, denominator))
    return atan_impl(builder, [atan_arg])


@cuda_lower_attr(QuaternionType, 'theta')
def cuda_quaternion_theta(context, builder, sig, arg):
    # Computes -math.asin(2 * (b * d - a * c))
    a = builder.extract_value(arg, 0)
    b = builder.extract_value(arg, 1)
    c = builder.extract_value(arg, 2)
    d = builder.extract_value(arg, 3)
    asin_sig = typing.signature(types.float64, types.float64)
    asin_impl = context.get_function(math.asin, asin_sig)

    x = builder.fsub(builder.fmul(b, d), builder.fmul(a, c))
    asin_arg = builder.fmul(context.get_constant(types.float64, 2), x)
    asin_res = asin_impl(builder, [asin_arg])
    return builder.fsub(context.get_constant(types.float64, -0.0), asin_res)


@cuda_lower_attr(QuaternionType, 'psi')
def cuda_quaternion_psi(context, builder, sig, arg):
    # Computes math.atan(2 * (a * d + b * c) / (a * a + b * b - c * c - d * d))
    a = builder.extract_value(arg, 0)
    b = builder.extract_value(arg, 1)
    c = builder.extract_value(arg, 2)
    d = builder.extract_value(arg, 3)

    a2 = builder.fmul(a, a)
    b2 = builder.fmul(b, b)
    c2 = builder.fmul(c, c)
    d2 = builder.fmul(d, d)

    numerator = builder.fadd(builder.fmul(a, d), builder.fmul(b, c))
    denominator = builder.fsub(builder.fsub(builder.fadd(a2, b2), c2), d2)

    atan_sig = typing.signature(types.float64, types.float64)
    atan_impl = context.get_function(math.atan, atan_sig)
    atan_arg = builder.fmul(context.get_constant(types.float64, 2),
                            builder.fdiv(numerator, denominator))
    return atan_impl(builder, [atan_arg])


@cuda.jit
def kernel(arr):
    q = Quaternion(1.0, 2.0, 9.75, 5.0)
    arr[0] = q.phi
    arr[1] = q.theta
    arr[2] = q.psi


numba_res = np.zeros(3)

kernel[1, 1](numba_res)

q = Quaternion(1, 2, 9.75, 5)
python_res = np.array([q.phi, q.theta, q.psi])

print("Computed with Numba-JITted code:")
print(numba_res)
print("Computed in Python:")
print(python_res)

# Sanity check
np.testing.assert_allclose(numba_res, python_res)
print("Sanity check passed!")
