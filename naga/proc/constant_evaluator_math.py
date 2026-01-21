"""Math function implementations for constant evaluation.

This module provides implementations of math functions that can be
evaluated at compile time in constant expressions.
Mirrors the math() method implementation in constant_evaluator.rs
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from naga import Handle, Expression, Span, MathFunction


if TYPE_CHECKING:
    from .constant_evaluator import ConstantEvaluator


# ============================================================================
# Math function implementations
# ============================================================================

class MathFunctionEvaluator:
    """Helper class for evaluating math functions."""

    @staticmethod
    def evaluate(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        arg2: Handle[Expression] | None,
        arg3: Handle[Expression] | None,
        fun: MathFunction,
        span: Span,
    ) -> Handle[Expression]:
        """Evaluate a math function.

        Args:
            eval: Constant evaluator instance
            arg: First argument
            arg1: Optional second argument
            arg2: Optional third argument
            arg3: Optional fourth argument
            fun: Math function to evaluate
            span: Span for error reporting

        Returns:
            Handle to the evaluated result

        Raises:
            ConstantEvaluatorError: If function cannot be evaluated
        """
        # Determine expected argument count
        expected = MathFunctionEvaluator._arg_count(fun)
        given = sum(1 for x in [arg, arg1, arg2, arg3] if x is not None)
        if expected != given:
            from .constant_evaluator import ConstantEvaluatorError
            raise ConstantEvaluatorError(
                "InvalidMathArgCount",
                {"function": fun, "expected": expected, "given": given},
            )

        # Dispatch to specific function implementation
        match fun:
            # Comparison functions
            case MathFunction.Abs:
                return MathFunctionEvaluator._abs(eval, arg, span)
            case MathFunction.Min:
                return MathFunctionEvaluator._min(eval, arg, arg1, span)
            case MathFunction.Max:
                return MathFunctionEvaluator._max(eval, arg, arg1, span)
            case MathFunction.Clamp:
                return MathFunctionEvaluator._clamp(eval, arg, arg1, arg2, span)
            case MathFunction.Saturate:
                return MathFunctionEvaluator._saturate(eval, arg, span)

            # Trigonometry
            case MathFunction.Cos:
                return MathFunctionEvaluator._cos(eval, arg, span)
            case MathFunction.Cosh:
                return MathFunctionEvaluator._cosh(eval, arg, span)
            case MathFunction.Sin:
                return MathFunctionEvaluator._sin(eval, arg, span)
            case MathFunction.Sinh:
                return MathFunctionEvaluator._sinh(eval, arg, span)
            case MathFunction.Tan:
                return MathFunctionEvaluator._tan(eval, arg, span)
            case MathFunction.Tanh:
                return MathFunctionEvaluator._tanh(eval, arg, span)
            case MathFunction.Acos:
                return MathFunctionEvaluator._acos(eval, arg, span)
            case MathFunction.Asin:
                return MathFunctionEvaluator._asin(eval, arg, span)
            case MathFunction.Atan:
                return MathFunctionEvaluator._atan(eval, arg, span)
            case MathFunction.Atan2:
                return MathFunctionEvaluator._atan2(eval, arg, arg1, span)
            case MathFunction.Asinh:
                return MathFunctionEvaluator._asinh(eval, arg, span)
            case MathFunction.Acosh:
                return MathFunctionEvaluator._acosh(eval, arg, span)
            case MathFunction.Atanh:
                return MathFunctionEvaluator._atanh(eval, arg, span)
            case MathFunction.Radians:
                return MathFunctionEvaluator._radians(eval, arg, span)
            case MathFunction.Degrees:
                return MathFunctionEvaluator._degrees(eval, arg, span)

            # Decomposition
            case MathFunction.Ceil:
                return MathFunctionEvaluator._ceil(eval, arg, span)
            case MathFunction.Floor:
                return MathFunctionEvaluator._floor(eval, arg, span)
            case MathFunction.Round:
                return MathFunctionEvaluator._round(eval, arg, span)
            case MathFunction.Fract:
                return MathFunctionEvaluator._fract(eval, arg, span)
            case MathFunction.Trunc:
                return MathFunctionEvaluator._trunc(eval, arg, span)

            # Exponent
            case MathFunction.Exp:
                return MathFunctionEvaluator._exp(eval, arg, span)
            case MathFunction.Exp2:
                return MathFunctionEvaluator._exp2(eval, arg, span)
            case MathFunction.Log:
                return MathFunctionEvaluator._log(eval, arg, span)
            case MathFunction.Log2:
                return MathFunctionEvaluator._log2(eval, arg, span)
            case MathFunction.Pow:
                return MathFunctionEvaluator._pow(eval, arg, arg1, span)

            # Computational
            case MathFunction.Sign:
                return MathFunctionEvaluator._sign(eval, arg, span)
            case MathFunction.Fma:
                return MathFunctionEvaluator._fma(eval, arg, arg1, arg2, span)
            case MathFunction.Step:
                return MathFunctionEvaluator._step(eval, arg, arg1, span)
            case MathFunction.Sqrt:
                return MathFunctionEvaluator._sqrt(eval, arg, span)
            case MathFunction.InverseSqrt:
                return MathFunctionEvaluator._inverse_sqrt(eval, arg, span)

            # Bit operations
            case MathFunction.CountTrailingZeros:
                return MathFunctionEvaluator._count_trailing_zeros(eval, arg, span)
            case MathFunction.CountLeadingZeros:
                return MathFunctionEvaluator._count_leading_zeros(eval, arg, span)
            case MathFunction.CountOneBits:
                return MathFunctionEvaluator._count_one_bits(eval, arg, span)
            case MathFunction.ReverseBits:
                return MathFunctionEvaluator._reverse_bits(eval, arg, span)
            case MathFunction.FirstTrailingBit:
                return MathFunctionEvaluator._first_trailing_bit(eval, arg, span)
            case MathFunction.FirstLeadingBit:
                return MathFunctionEvaluator._first_leading_bit(eval, arg, span)

            # Vector operations
            case MathFunction.Dot4I8Packed:
                return MathFunctionEvaluator._dot4_i8_packed(eval, arg, arg1, span)
            case MathFunction.Dot4U8Packed:
                return MathFunctionEvaluator._dot4_u8_packed(eval, arg, arg1, span)
            case MathFunction.Cross:
                return MathFunctionEvaluator._cross(eval, arg, arg1, span)
            case MathFunction.Dot:
                return MathFunctionEvaluator._dot(eval, arg, arg1, span)
            case MathFunction.Length:
                return MathFunctionEvaluator._length(eval, arg, span)
            case MathFunction.Distance:
                return MathFunctionEvaluator._distance(eval, arg, arg1, span)
            case MathFunction.Normalize:
                return MathFunctionEvaluator._normalize(eval, arg, span)

            # Not implemented functions
            case _:
                from .constant_evaluator import ConstantEvaluatorError
                raise ConstantEvaluatorError(
                    "NotImplemented",
                    {"message": f"Math function {fun} not implemented"},
                )

    @staticmethod
    def _arg_count(fun: MathFunction) -> int:
        """Return expected argument count for a math function."""
        # Single argument functions
        single_arg = {
            MathFunction.Abs,
            MathFunction.Acos,
            MathFunction.Asin,
            MathFunction.Atan,
            MathFunction.Ceiling,
            MathFunction.Cos,
            MathFunction.Cosh,
            MathFunction.Exp,
            MathFunction.Exp2,
            MathFunction.Floor,
            MathFunction.Fract,
            MathFunction.Length,
            MathFunction.Log,
            MathFunction.Log2,
            MathFunction.Normalize,
            MathFunction.Round,
            MathFunction.Sign,
            MathFunction.Sin,
            MathFunction.Sinh,
            MathFunction.Sqrt,
            MathFunction.Sqr,
            MathFunction.Tan,
            MathFunction.Tanh,
            MathFunction.Trunc,
            MathFunction.CountLeadingZeros,
            MathFunction.CountOneBits,
            MathFunction.CountTrailingZeros,
            MathFunction.FirstLeadingBit,
            MathFunction.FirstTrailingBit,
            MathFunction.InverseSqrt,
            MathFunction.Saturate,
        }

        # Two argument functions
        two_arg = {
            MathFunction.Atan2,
            MathFunction.Distance,
            MathFunction.Max,
            MathFunction.Min,
            MathFunction.Pow,
            MathFunction.Step,
            MathFunction.Dot4I8Packed,
            MathFunction.Dot4U8Packed,
            MathFunction.Cross,
            MathFunction.Dot,
        }

        # Three argument functions
        three_arg = {
            MathFunction.Clamp,
            MathFunction.Fma,
            MathFunction.Mix,
            MathFunction.SmoothStep,
        }

        if fun in single_arg:
            return 1
        elif fun in two_arg:
            return 2
        elif fun in three_arg:
            return 3
        else:
            return 0

    # ========================================================================
    # Comparison functions
    # ========================================================================

    @staticmethod
    def _abs(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Absolute value."""
        from .component_wise import component_wise_scalar, Scalar
        
        def handler(args: Scalar) -> Scalar:
            match args:
                case Scalar.AbstractFloat(values=[e]):
                    return Scalar.AbstractFloat([abs(e)])
                case Scalar.F32(values=[e]):
                    return Scalar.F32([abs(e)])
                case Scalar.F16(values=[e]):
                    return Scalar.F16([abs(e)])
                case Scalar.AbstractInt(values=[e]):
                    # wrapping_abs for integers
                    return Scalar.AbstractInt([abs(e)])
                case Scalar.I32(values=[e]):
                    return Scalar.I32([abs(e)])
                case Scalar.U32(values=[e]):
                    return Scalar.U32([e])  # Already positive
                case Scalar.I64(values=[e]):
                    return Scalar.I64([abs(e)])
                case Scalar.U64(values=[e]):
                    return Scalar.U64([e])  # Already positive
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_scalar(eval, span, [arg], handler)

    @staticmethod
    def _min(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Minimum of two values."""
        from .component_wise import component_wise_scalar, Scalar
        
        def handler(args: Scalar) -> Scalar:
            match args:
                case Scalar.AbstractFloat(values=[e1, e2]):
                    return Scalar.AbstractFloat([min(e1, e2)])
                case Scalar.F32(values=[e1, e2]):
                    return Scalar.F32([min(e1, e2)])
                case Scalar.F16(values=[e1, e2]):
                    return Scalar.F16([min(e1, e2)])
                case Scalar.AbstractInt(values=[e1, e2]):
                    return Scalar.AbstractInt([min(e1, e2)])
                case Scalar.I32(values=[e1, e2]):
                    return Scalar.I32([min(e1, e2)])
                case Scalar.U32(values=[e1, e2]):
                    return Scalar.U32([min(e1, e2)])
                case Scalar.I64(values=[e1, e2]):
                    return Scalar.I64([min(e1, e2)])
                case Scalar.U64(values=[e1, e2]):
                    return Scalar.U64([min(e1, e2)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_scalar(eval, span, [arg, arg1], handler)

    @staticmethod
    def _max(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Maximum of two values."""
        from .component_wise import component_wise_scalar, Scalar
        
        def handler(args: Scalar) -> Scalar:
            match args:
                case Scalar.AbstractFloat(values=[e1, e2]):
                    return Scalar.AbstractFloat([max(e1, e2)])
                case Scalar.F32(values=[e1, e2]):
                    return Scalar.F32([max(e1, e2)])
                case Scalar.F16(values=[e1, e2]):
                    return Scalar.F16([max(e1, e2)])
                case Scalar.AbstractInt(values=[e1, e2]):
                    return Scalar.AbstractInt([max(e1, e2)])
                case Scalar.I32(values=[e1, e2]):
                    return Scalar.I32([max(e1, e2)])
                case Scalar.U32(values=[e1, e2]):
                    return Scalar.U32([max(e1, e2)])
                case Scalar.I64(values=[e1, e2]):
                    return Scalar.I64([max(e1, e2)])
                case Scalar.U64(values=[e1, e2]):
                    return Scalar.U64([max(e1, e2)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_scalar(eval, span, [arg, arg1], handler)

    @staticmethod
    def _clamp(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        arg2: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Clamp value between low and high."""
        from .component_wise import component_wise_scalar, Scalar
        
        def handler(args: Scalar) -> Scalar:
            match args:
                case Scalar.AbstractFloat(values=[e, low, high]):
                    return Scalar.AbstractFloat([max(low, min(e, high))])
                case Scalar.F32(values=[e, low, high]):
                    return Scalar.F32([max(low, min(e, high))])
                case Scalar.F16(values=[e, low, high]):
                    return Scalar.F16([max(low, min(e, high))])
                case Scalar.AbstractInt(values=[e, low, high]):
                    return Scalar.AbstractInt([max(low, min(e, high))])
                case Scalar.I32(values=[e, low, high]):
                    return Scalar.I32([max(low, min(e, high))])
                case Scalar.U32(values=[e, low, high]):
                    return Scalar.U32([max(low, min(e, high))])
                case Scalar.I64(values=[e, low, high]):
                    return Scalar.I64([max(low, min(e, high))])
                case Scalar.U64(values=[e, low, high]):
                    return Scalar.U64([max(low, min(e, high))])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_scalar(eval, span, [arg, arg1, arg2], handler)

    @staticmethod
    def _saturate(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Saturate value between 0 and 1."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([max(0.0, min(e, 1.0))])
                case Float.F32(values=[e]):
                    return Float.F32([max(0.0, min(e, 1.0))])
                case Float.F16(values=[e]):
                    return Float.F16([max(0.0, min(e, 1.0))])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    # ========================================================================
    # Trigonometry
    # ========================================================================

    @staticmethod
    def _cos(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Cosine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.cos(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.cos(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.cos(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _sin(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Sine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.sin(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.sin(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.sin(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _tan(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Tangent."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.tan(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.tan(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.tan(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _acos(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Arc cosine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.acos(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.acos(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.acos(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _asin(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Arc sine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.asin(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.asin(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.asin(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _atan(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Arc tangent."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.atan(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.atan(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.atan(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _atan2(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Arc tangent of y/x."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[y, x]):
                    return Float.Abstract([math.atan2(y, x)])
                case Float.F32(values=[y, x]):
                    return Float.F32([math.atan2(y, x)])
                case Float.F16(values=[y, x]):
                    return Float.F16([math.atan2(y, x)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg, arg1], handler)

    @staticmethod
    def _sinh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Hyperbolic sine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.sinh(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.sinh(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.sinh(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _cosh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Hyperbolic cosine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.cosh(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.cosh(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.cosh(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _tanh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Hyperbolic tangent."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.tanh(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.tanh(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.tanh(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _asinh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Inverse hyperbolic sine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.asinh(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.asinh(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.asinh(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _acosh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Inverse hyperbolic cosine."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.acosh(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.acosh(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.acosh(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _atanh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Inverse hyperbolic tangent."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.atanh(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.atanh(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.atanh(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _radians(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Convert degrees to radians."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.radians(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.radians(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.radians(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _degrees(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Convert radians to degrees."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.degrees(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.degrees(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.degrees(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    # ========================================================================
    # Decomposition
    # ========================================================================

    @staticmethod
    def _ceil(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Ceiling (round up)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.ceil(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.ceil(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.ceil(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _floor(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Floor (round down)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.floor(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.floor(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.floor(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _round(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Round to nearest integer (ties to even)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([round(e)])
                case Float.F32(values=[e]):
                    return Float.F32([round(e)])
                case Float.F16(values=[e]):
                    return Float.F16([round(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _fract(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Fractional part (x - floor(x))."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([e - math.floor(e)])
                case Float.F32(values=[e]):
                    return Float.F32([e - math.floor(e)])
                case Float.F16(values=[e]):
                    return Float.F16([e - math.floor(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _trunc(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Truncate (round toward zero)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.trunc(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.trunc(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.trunc(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    # ========================================================================
    # Exponent
    # ========================================================================

    @staticmethod
    def _exp(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Exponential (e^x)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.exp(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.exp(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.exp(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _exp2(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Base-2 exponential (2^x)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.pow(2.0, e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.pow(2.0, e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.pow(2.0, e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _log(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Natural logarithm (ln)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.log(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.log(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.log(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _log2(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Base-2 logarithm."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.log2(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.log2(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.log2(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _pow(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Power (x^y)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[x, y]):
                    return Float.Abstract([math.pow(x, y)])
                case Float.F32(values=[x, y]):
                    return Float.F32([math.pow(x, y)])
                case Float.F16(values=[x, y]):
                    return Float.F16([math.pow(x, y)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg, arg1], handler)

    # ========================================================================
    # Computational
    # ========================================================================

    @staticmethod
    def _sign(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Sign of a value (-1, 0, or 1)."""
        from .component_wise import component_wise_signed, Signed
        
        def handler(args: Signed) -> Signed:
            def sign_value(e):
                if e > 0:
                    return type(e)(1)
                elif e < 0:
                    return type(e)(-1)
                else:
                    return type(e)(0)
            
            match args:
                case Signed.AbstractFloat(values=[e]):
                    return Signed.AbstractFloat([sign_value(e)])
                case Signed.AbstractInt(values=[e]):
                    return Signed.AbstractInt([sign_value(e)])
                case Signed.F32(values=[e]):
                    return Signed.F32([sign_value(e)])
                case Signed.F16(values=[e]):
                    return Signed.F16([sign_value(e)])
                case Signed.I32(values=[e]):
                    return Signed.I32([sign_value(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_signed(eval, span, [arg], handler)

    @staticmethod
    def _fma(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        arg2: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Fused multiply-add (a * b + c)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[a, b, c]):
                    return Float.Abstract([a * b + c])
                case Float.F32(values=[a, b, c]):
                    return Float.F32([a * b + c])
                case Float.F16(values=[a, b, c]):
                    return Float.F16([a * b + c])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg, arg1, arg2], handler)

    @staticmethod
    def _step(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Step function (0 if x < edge, 1 otherwise)."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[edge, x]):
                    return Float.Abstract([0.0 if x < edge else 1.0])
                case Float.F32(values=[edge, x]):
                    return Float.F32([0.0 if x < edge else 1.0])
                case Float.F16(values=[edge, x]):
                    return Float.F16([0.0 if x < edge else 1.0])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg, arg1], handler)

    @staticmethod
    def _sqrt(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Square root."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([math.sqrt(e)])
                case Float.F32(values=[e]):
                    return Float.F32([math.sqrt(e)])
                case Float.F16(values=[e]):
                    return Float.F16([math.sqrt(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    @staticmethod
    def _inverse_sqrt(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Inverse square root (1 / sqrt(x))."""
        from .component_wise import component_wise_float, Float
        
        def handler(args: Float) -> Float:
            match args:
                case Float.Abstract(values=[e]):
                    return Float.Abstract([1.0 / math.sqrt(e)])
                case Float.F32(values=[e]):
                    return Float.F32([1.0 / math.sqrt(e)])
                case Float.F16(values=[e]):
                    return Float.F16([1.0 / math.sqrt(e)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_float(eval, span, [arg], handler)

    # ========================================================================
    # Bit operations
    # ========================================================================

    @staticmethod
    def _count_trailing_zeros(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Count trailing zero bits."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            def count_tz(value: int, bits: int = 32) -> int:
                if value == 0:
                    return bits
                count = 0
                while (value & 1) == 0:
                    value >>= 1
                    count += 1
                return count
            
            match args:
                case ConcreteInt.U32(values=[e]):
                    return ConcreteInt.U32([count_tz(e & 0xFFFFFFFF, 32)])
                case ConcreteInt.I32(values=[e]):
                    return ConcreteInt.I32([count_tz(e & 0xFFFFFFFF, 32)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg], handler)

    @staticmethod
    def _count_leading_zeros(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Count leading zero bits."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            def count_lz(value: int, bits: int = 32) -> int:
                if value == 0:
                    return bits
                count = 0
                mask = 1 << (bits - 1)
                while (value & mask) == 0:
                    count += 1
                    mask >>= 1
                return count
            
            match args:
                case ConcreteInt.U32(values=[e]):
                    return ConcreteInt.U32([count_lz(e & 0xFFFFFFFF, 32)])
                case ConcreteInt.I32(values=[e]):
                    return ConcreteInt.I32([count_lz(e & 0xFFFFFFFF, 32)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg], handler)

    @staticmethod
    def _count_one_bits(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Count set bits."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            def popcount(value: int) -> int:
                count = 0
                while value:
                    count += value & 1
                    value >>= 1
                return count
            
            match args:
                case ConcreteInt.U32(values=[e]):
                    return ConcreteInt.U32([popcount(e & 0xFFFFFFFF)])
                case ConcreteInt.I32(values=[e]):
                    return ConcreteInt.I32([popcount(e & 0xFFFFFFFF)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg], handler)

    @staticmethod
    def _reverse_bits(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Reverse bits."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            def reverse(value: int, bits: int = 32) -> int:
                result = 0
                for _ in range(bits):
                    result = (result << 1) | (value & 1)
                    value >>= 1
                return result
            
            match args:
                case ConcreteInt.U32(values=[e]):
                    return ConcreteInt.U32([reverse(e & 0xFFFFFFFF, 32)])
                case ConcreteInt.I32(values=[e]):
                    return ConcreteInt.I32([reverse(e & 0xFFFFFFFF, 32)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg], handler)

    @staticmethod
    def _first_trailing_bit(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Find index of first trailing set bit."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        from .type_methods import first_trailing_bit
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            match args:
                case ConcreteInt.U32(values=[e]):
                    return ConcreteInt.U32([first_trailing_bit(e, signed=False)])
                case ConcreteInt.I32(values=[e]):
                    return ConcreteInt.I32([first_trailing_bit(e, signed=True)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg], handler)

    @staticmethod
    def _first_leading_bit(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Find index of first leading set bit."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        from .type_methods import first_leading_bit
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            match args:
                case ConcreteInt.U32(values=[e]):
                    return ConcreteInt.U32([first_leading_bit(e, signed=False)])
                case ConcreteInt.I32(values=[e]):
                    return ConcreteInt.I32([first_leading_bit(e, signed=True)])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg], handler)

    # ========================================================================
    # Vector operations
    # ========================================================================

    @staticmethod
    def _dot4_i8_packed(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Dot product of two packed i8 vectors."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        
        def unpack_i8x4(packed: int) -> list[int]:
            """Unpack 4 signed bytes from u32."""
            bytes_list = []
            for i in range(4):
                byte = (packed >> (i * 8)) & 0xFF
                # Sign extend from 8 bits to 32 bits
                if byte & 0x80:
                    byte = byte - 256
                bytes_list.append(byte)
            return bytes_list
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            match args:
                case ConcreteInt.U32(values=[a, b]) | ConcreteInt.I32(values=[a, b]):
                    # Unpack both values
                    a_bytes = unpack_i8x4(a & 0xFFFFFFFF)
                    b_bytes = unpack_i8x4(b & 0xFFFFFFFF)
                    
                    # Compute dot product
                    result = sum(x * y for x, y in zip(a_bytes, b_bytes))
                    
                    # Return as I32 (signed result)
                    return ConcreteInt.I32([result])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg, arg1], handler)

    @staticmethod
    def _dot4_u8_packed(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Dot product of two packed u8 vectors."""
        from .component_wise import component_wise_concrete_int, ConcreteInt
        
        def unpack_u8x4(packed: int) -> list[int]:
            """Unpack 4 unsigned bytes from u32."""
            return [
                (packed >> (i * 8)) & 0xFF
                for i in range(4)
            ]
        
        def handler(args: ConcreteInt) -> ConcreteInt:
            match args:
                case ConcreteInt.U32(values=[a, b]) | ConcreteInt.I32(values=[a, b]):
                    # Unpack both values
                    a_bytes = unpack_u8x4(a & 0xFFFFFFFF)
                    b_bytes = unpack_u8x4(b & 0xFFFFFFFF)
                    
                    # Compute dot product
                    result = sum(x * y for x, y in zip(a_bytes, b_bytes))
                    
                    # Return as U32 (unsigned result)
                    return ConcreteInt.U32([result])
                case _:
                    from .constant_evaluator import ConstantEvaluatorError
                    raise ConstantEvaluatorError("InvalidMathArg", {})
        
        return component_wise_concrete_int(eval, span, [arg, arg1], handler)

    @staticmethod
    def _cross(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Cross product of two 3D vectors."""
        from .vector_helpers import extract_vector_float_values, cross_product
        from naga import Expression, ExpressionType, Literal
        
        # Extract vector values
        vec1 = extract_vector_float_values(eval, arg, span)
        vec2 = extract_vector_float_values(eval, arg1, span)
        
        # Compute cross product
        result = cross_product(vec1, vec2)
        
        # Create vector literal components
        components = []
        for value in result:
            lit_expr = Expression(
                type=ExpressionType.LITERAL,
                literal=Literal.F32(value),
            )
            components.append(eval.register_evaluated_expr(lit_expr, span))
        
        # Get the type from the first argument
        arg_handle = eval.eval_zero_value_and_splat(arg, span)
        arg_expr = eval.expressions[arg_handle]
        
        if arg_expr.type == ExpressionType.COMPOSE:
            result_ty = arg_expr.compose_ty
        else:
            from .constant_evaluator import ConstantEvaluatorError
            raise ConstantEvaluatorError("Cross product requires vector arguments", {})
        
        # Create compose expression
        new_expr = Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=result_ty,
            compose_components=components,
        )
        
        return eval.register_evaluated_expr(new_expr, span)

    @staticmethod
    def _dot(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Dot product of two vectors."""
        from .vector_helpers import extract_vector_float_values, dot_product, create_float_literal
        
        # Extract vector values
        vec1 = extract_vector_float_values(eval, arg, span)
        vec2 = extract_vector_float_values(eval, arg1, span)
        
        # Compute dot product
        result = dot_product(vec1, vec2)
        
        # Create result literal
        return create_float_literal(eval, result, span)

    @staticmethod
    def _length(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Length of a vector."""
        from .vector_helpers import extract_vector_float_values, vector_length, create_float_literal
        
        # Extract vector values
        vec = extract_vector_float_values(eval, arg, span)
        
        # Compute length
        result = vector_length(vec)
        
        # Create result literal
        return create_float_literal(eval, result, span)

    @staticmethod
    def _distance(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Distance between two points."""
        from .vector_helpers import extract_vector_float_values, vector_distance, create_float_literal
        
        # Extract vector values
        vec1 = extract_vector_float_values(eval, arg, span)
        vec2 = extract_vector_float_values(eval, arg1, span)
        
        # Compute distance
        result = vector_distance(vec1, vec2)
        
        # Create result literal
        return create_float_literal(eval, result, span)

    @staticmethod
    def _normalize(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Normalize a vector."""
        from .vector_helpers import extract_vector_float_values, vector_normalize
        from naga import Expression, ExpressionType, Literal
        
        # Extract vector values
        vec = extract_vector_float_values(eval, arg, span)
        
        # Normalize
        result = vector_normalize(vec)
        
        # Create vector literal components
        components = []
        for value in result:
            lit_expr = Expression(
                type=ExpressionType.LITERAL,
                literal=Literal.F32(value),
            )
            components.append(eval.register_evaluated_expr(lit_expr, span))
        
        # Get the type from the argument
        arg_handle = eval.eval_zero_value_and_splat(arg, span)
        arg_expr = eval.expressions[arg_handle]
        
        if arg_expr.type == ExpressionType.COMPOSE:
            result_ty = arg_expr.compose_ty
        else:
            from .constant_evaluator import ConstantEvaluatorError
            raise ConstantEvaluatorError("Normalize requires vector argument", {})
        
        # Create compose expression
        new_expr = Expression(
            type=ExpressionType.COMPOSE,
            compose_ty=result_ty,
            compose_components=components,
        )
        
        return eval.register_evaluated_expr(new_expr, span)
