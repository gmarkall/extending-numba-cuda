# From Interval example docs - typing
from numba import types
from numba.core.extending import typeof_impl, type_callable

# Data model

from numba.core.extending import models, register_model

# Lowering

from numba.extending import lower_builtin
from numba.core import cgutils


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


#### User code

from numba import cuda


@cuda.jit
def f():
    x = Interval(1.0, 3.0)

f[1, 1]()
