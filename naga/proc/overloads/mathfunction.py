from naga import ir
from .overload_set import OverloadSet
from .any_overload_set import AnyOverloadSet
from .regular import Regular, ConclusionRule
from .list import ListOverloadSet
from .constructor_set import ConstructorSet
from .scalar_set import ScalarSet
from .utils import (
    float_scalars, float_scalars_unimplemented_abstract, list_set, pairs, rule, 
    scalar_or_vecn, triples, concrete_int_scalars, vector_sizes
)

def get_math_function_overloads(fun: ir.MathFunction) -> OverloadSet:
    """Return the overload set for a given math function."""
    from naga.ir.operators import MathFunction as Mf
    
    # Component-wise unary numeric operations
    if fun in (Mf.ABS, Mf.SIGN):
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.NUMERIC, ConclusionRule.ArgumentType)

    # Component-wise binary numeric operations
    if fun in (Mf.MIN, Mf.MAX):
        return Regular(2, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.NUMERIC, ConclusionRule.ArgumentType)

    # Component-wise ternary numeric operations
    if fun == Mf.CLAMP:
        return Regular(3, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.NUMERIC, ConclusionRule.ArgumentType)

    # Component-wise unary floating-point operations
    if fun in (
        Mf.SIN, Mf.COS, Mf.TAN, Mf.ASIN, Mf.ACOS, Mf.ATAN, Mf.SINH, Mf.COSH, Mf.TANH, 
        Mf.ASINH, Mf.ACOSH, Mf.ATANH, Mf.SATURATE, Mf.RADIANS, Mf.DEGREES, Mf.CEIL, 
        Mf.FLOOR, Mf.ROUND, Mf.FRACT, Mf.TRUNC, Mf.EXP, Mf.EXP2, Mf.LOG, Mf.LOG2, 
        Mf.SQRT, Mf.INVERSE_SQRT
    ):
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT, ConclusionRule.ArgumentType)

    # Component-wise binary floating-point operations
    if fun in (Mf.ATAN2, Mf.POW, Mf.STEP):
        return Regular(2, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT, ConclusionRule.ArgumentType)

    # Component-wise ternary floating-point operations
    if fun in (Mf.FMA, Mf.SMOOTH_STEP):
        return Regular(3, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT, ConclusionRule.ArgumentType)

    # Component-wise unary concrete integer operations
    if fun in (
        Mf.COUNT_TRAILING_ZEROS, Mf.COUNT_LEADING_ZEROS, Mf.COUNT_ONE_BITS, 
        Mf.REVERSE_BITS, Mf.FIRST_TRAILING_BIT, Mf.FIRST_LEADING_BIT
    ):
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.CONCRETE_INTEGER, ConclusionRule.ArgumentType)

    # Packing functions
    if fun in (Mf.PACK_4X8_SNORM, Mf.PACK_4X8_UNORM):
        return Regular(1, ConstructorSet.VEC4, ScalarSet.F32, ConclusionRule.U32)
    if fun in (Mf.PACK_2X16_SNORM, Mf.PACK_2X16_UNORM, Mf.PACK_2X16_FLOAT):
        return Regular(1, ConstructorSet.VEC2, ScalarSet.F32, ConclusionRule.U32)
    if fun == Mf.PACK_4X_I8:
        return Regular(1, ConstructorSet.VEC4, ScalarSet.I32, ConclusionRule.U32)
    if fun == Mf.PACK_4X_U8:
        return Regular(1, ConstructorSet.VEC4, ScalarSet.U32, ConclusionRule.U32)
    if fun == Mf.PACK_4X_I8_CLAMP:
        return Regular(1, ConstructorSet.VEC4, ScalarSet.I32, ConclusionRule.U32)
    if fun == Mf.PACK_4X_U8_CLAMP:
        return Regular(1, ConstructorSet.VEC4, ScalarSet.U32, ConclusionRule.U32)

    # Unpacking functions
    if fun in (Mf.UNPACK_4X8_SNORM, Mf.UNPACK_4X8_UNORM):
        return Regular(1, ConstructorSet.SCALAR, ScalarSet.U32, ConclusionRule.Vec4F)
    if fun in (Mf.UNPACK_2X16_SNORM, Mf.UNPACK_2X16_UNORM, Mf.UNPACK_2X16_FLOAT):
        return Regular(1, ConstructorSet.SCALAR, ScalarSet.U32, ConclusionRule.Vec2F)
    if fun == Mf.UNPACK_4X_I8:
        return Regular(1, ConstructorSet.SCALAR, ScalarSet.U32, ConclusionRule.Vec4I)
    if fun == Mf.UNPACK_4X_U8:
        return Regular(1, ConstructorSet.SCALAR, ScalarSet.U32, ConclusionRule.Vec4U)
    if fun == Mf.DOT4_I8_PACKED:
        return Regular(2, ConstructorSet.SCALAR, ScalarSet.U32, ConclusionRule.I32)
    if fun == Mf.DOT4_U8_PACKED:
        return Regular(2, ConstructorSet.SCALAR, ScalarSet.U32, ConclusionRule.U32)

    # One-off operations
    if fun == Mf.DOT:
        return Regular(2, ConstructorSet.VECN, ScalarSet.NUMERIC, ConclusionRule.Scalar)
    if fun == Mf.MODF:
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.Modf)
    if fun == Mf.FREXP:
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.Frexp)
    if fun == Mf.LDEXP:
        return ldexp()
    if fun == Mf.OUTER:
        return outer()
    if fun == Mf.CROSS:
        return Regular(2, ConstructorSet.VEC3, ScalarSet.FLOAT, ConclusionRule.ArgumentType)
    if fun == Mf.DISTANCE:
        return Regular(2, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.Scalar)
    if fun == Mf.LENGTH:
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.Scalar)
    if fun == Mf.NORMALIZE:
        return Regular(1, ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.ArgumentType)
    if fun == Mf.FACE_FORWARD:
        return Regular(3, ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.ArgumentType)
    if fun == Mf.REFLECT:
        return Regular(2, ConstructorSet.VECN, ScalarSet.FLOAT_ABSTRACT_UNIMPLEMENTED, ConclusionRule.ArgumentType)
    if fun == Mf.REFRACT:
        return refract()
    if fun == Mf.MIX:
        return mix()
    if fun in (Mf.INVERSE, Mf.DETERMINANT):
        conclude = ConclusionRule.ArgumentType if fun == Mf.INVERSE else ConclusionRule.Scalar
        return Regular(1, ConstructorSet.MAT2X2 | ConstructorSet.MAT3X3 | ConstructorSet.MAT4X4, ScalarSet.FLOAT, conclude)
    if fun == Mf.TRANSPOSE:
        return transpose()
    if fun == Mf.QUANTIZE_TO_F16:
        return Regular(1, ConstructorSet.SCALAR | ConstructorSet.VECN, ScalarSet.F32, ConclusionRule.ArgumentType)
    if fun == Mf.EXTRACT_BITS:
        return extract_bits()
    if fun == Mf.INSERT_BITS:
        return insert_bits()

    raise NotImplementedError(f"No overload set for math function {fun}")

def ldexp() -> ListOverloadSet:
    from naga.ir.type import ScalarKind, Scalar as IrScalar
    def exponent_from_mantissa(mantissa: ir.Scalar) -> ir.Scalar:
        if mantissa.kind == ScalarKind.ABSTRACT_FLOAT:
            return IrScalar(ScalarKind.ABSTRACT_INT, 0)
        elif mantissa.kind == ScalarKind.FLOAT:
            return IrScalar(ScalarKind.SINT, 4)
        raise ValueError("not a float scalar")

    rules = []
    for mantissa_scalar in float_scalars_unimplemented_abstract():
        exponent_scalar = exponent_from_mantissa(mantissa_scalar)
        for mantissa, exponent in zip(scalar_or_vecn(mantissa_scalar), scalar_or_vecn(exponent_scalar)):
            rules.append(rule([mantissa, exponent], mantissa))
    return list_set(rules)

def outer() -> ListOverloadSet:
    rules = []
    for cols, rows, scalar in triples(vector_sizes(), vector_sizes(), float_scalars_unimplemented_abstract()):
        left = ir.TypeInner.vector(cols, scalar)
        right = ir.TypeInner.vector(rows, scalar)
        res = ir.TypeInner.matrix(cols, rows, scalar)
        rules.append(rule([left, right], res))
    return list_set(rules)

def refract() -> ListOverloadSet:
    rules = []
    for size, scalar in pairs(vector_sizes(), float_scalars_unimplemented_abstract()):
        incident = ir.TypeInner.vector(size, scalar)
        normal = incident
        ratio = ir.TypeInner.scalar(scalar)
        rules.append(rule([incident, normal, ratio], incident))
    return list_set(rules)

def transpose() -> ListOverloadSet:
    rules = []
    for a, b, scalar in triples(vector_sizes(), vector_sizes(), float_scalars()):
        input_ty = ir.TypeInner.matrix(a, b, scalar)
        output_ty = ir.TypeInner.matrix(b, a, scalar)
        rules.append(rule([input_ty], output_ty))
    return list_set(rules)

def extract_bits() -> ListOverloadSet:
    from naga.ir.type import ScalarKind, Scalar as IrScalar
    rules = []
    for s in concrete_int_scalars():
        for input_ty in scalar_or_vecn(s):
            offset = ir.TypeInner.scalar(IrScalar(ScalarKind.UINT, 4))
            count = ir.TypeInner.scalar(IrScalar(ScalarKind.UINT, 4))
            rules.append(rule([input_ty, offset, count], input_ty))
    return list_set(rules)

def insert_bits() -> ListOverloadSet:
    from naga.ir.type import ScalarKind, Scalar as IrScalar
    rules = []
    for s in concrete_int_scalars():
        for input_ty in scalar_or_vecn(s):
            newbits = input_ty
            offset = ir.TypeInner.scalar(IrScalar(ScalarKind.UINT, 4))
            count = ir.TypeInner.scalar(IrScalar(ScalarKind.UINT, 4))
            rules.append(rule([input_ty, newbits, offset, count], input_ty))
    return list_set(rules)

def mix() -> ListOverloadSet:
    rules = []
    for s in float_scalars():
        for input_ty in scalar_or_vecn(s):
            scalar_ratio = ir.TypeInner.scalar(s)
            rules.append(rule([input_ty, input_ty, input_ty], input_ty))
            rules.append(rule([input_ty, input_ty, scalar_ratio], input_ty))
    return list_set(rules)
