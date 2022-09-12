from numba import cuda, config, types
from numba.cuda.extending import intrinsic

config.CUDA_WARN_ON_IMPLICIT_COPY = False
config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@intrinsic
def cast_int_to_byte_ptr(typingctx, src):
    # check for accepted types
    if isinstance(src, types.Integer):
        # create the expected type signature
        result_type = types.CPointer(types.uint8)
        sig = result_type(types.uintp)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)
        return sig, codegen


@cuda.jit('void(int64, int)')
def f(arr):
    print("Sum is", arr.sum())

