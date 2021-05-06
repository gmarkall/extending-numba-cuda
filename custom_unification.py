"""
This demonstrates customised unification for an extension type, with the
unify() method. The unify() method handles the typing - a successful typing
will require lowering of a cast to the extension type from other types.
"""

import sys

from numba.extending import (
    types,
    register_model,
    models,
)

from numba.core import cgutils
from numba.core.errors import TypingError
from numba.cuda.cudaimpl import registry as cuda_registry
from numba import cuda

from colorama import init, Fore, Style
init()


class ExtensionType(types.Type):
    """An extension type parameterized by an underlying value type"""
    def __init__(self, value):
        super().__init__(name="Extension")
        self.value = value

    # Defined to make debugging a little easier
    def __repr__(self):
        return f'Extension({self.value})'

    def unify(self, context, other):
        """Custom unification for the extension type"""
        # The aim here is to mostly re-use Numba's unification machinery for
        # the types to figure out what we should unify to. For example, it's
        # straightforward to see that we can do:
        #
        #     {ExtensionType(int64), int64} -> ExtensionType(int64)
        #
        # but for something like:
        #
        #     {ExtensionType(uint32), float32}
        #
        # the correct unification is less obvious. So we reuse Numba logic that
        # would do (probably, I haven't checked):
        #
        #     {uint32, float32} -> {float64}
        #
        # to do:
        #
        #     {ExtensionType(uint32), float32} -> ExtensionType(float64)
        #
        # without any complicated / duplicate logic of our own.

        # Try to unify our value type and the other type
        unified = context.unify_pairs(self.value, other)

        # If that unification failed, then there's no way we can unify
        if unified is None:
            return None

        # Otherwise, the unified type is an extension parameterised by the
        # unified type
        return ExtensionType(unified)


@register_model(ExtensionType)
class ExtensionModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("value", fe_type.value)]
        models.StructModel.__init__(self, dmm, fe_type, members)


# To handle the unification, we need to support casting from any type to an
# extension type. The cast implementation takes the value passed in and returns
# an extension struct wrapping that value.
@cuda_registry.lower_cast(types.Any, ExtensionType)
def cast_primitive_to_extension(context, builder, fromty, toty, val):
    casted = context.cast(builder, val, fromty, toty.value)
    ext = cgutils.create_struct_proxy(toty)(context, builder)
    ext.value = casted
    return ext._getvalue()


# Compilation of this succeeds, because we can unify an int64 and another int64
def func1(x, y):
    if y > 5:
        return x
    else:
        return 2


signature = (ExtensionType(types.int64), types.int64)

cuda.compile_ptx_for_current_device(func1, signature, device=True)


# Compilation of this fails, because an int64 and a tuple don't unify
def func2(x, y):
    if y > 5:
        return x
    else:
        return (2, 2)


try:
    cuda.compile_ptx_for_current_device(func2, signature, device=True)
except TypingError as e:
    msg = "Compilation failed with a TypeError as expected:\n"
    print(Fore.GREEN + Style.BRIGHT + "SUCCESS: " + Style.RESET_ALL + msg)
    [print(arg) for arg in e.args]
    sys.exit(0)
else:
    print(Fore.RED + Style.BRIGHT + "FAIL: " + Style.RESET_ALL +
          "Expected a TypeError compiling func2")
    sys.exit(1)
