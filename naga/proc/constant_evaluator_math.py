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
        # TODO: Implement with component_wise_scalar
        raise NotImplementedError("Abs not implemented")

    @staticmethod
    def _min(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Minimum of two values."""
        # TODO: Implement with component_wise_scalar
        raise NotImplementedError("Min not implemented")

    @staticmethod
    def _max(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Maximum of two values."""
        # TODO: Implement with component_wise_scalar
        raise NotImplementedError("Max not implemented")

    @staticmethod
    def _clamp(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        arg2: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Clamp value between low and high."""
        # TODO: Implement with component_wise_scalar
        raise NotImplementedError("Clamp not implemented")

    @staticmethod
    def _saturate(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Saturate value between 0 and 1."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Saturate not implemented")

    # ========================================================================
    # Trigonometry
    # ========================================================================

    @staticmethod
    def _cos(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Cosine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Cos not implemented")

    @staticmethod
    def _sin(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Sine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Sin not implemented")

    @staticmethod
    def _tan(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Tangent."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Tan not implemented")

    @staticmethod
    def _acos(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Arc cosine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Acos not implemented")

    @staticmethod
    def _asin(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Arc sine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Asin not implemented")

    @staticmethod
    def _atan(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Arc tangent."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Atan not implemented")

    @staticmethod
    def _atan2(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Arc tangent of y/x."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Atan2 not implemented")

    @staticmethod
    def _sinh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Hyperbolic sine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Sinh not implemented")

    @staticmethod
    def _cosh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Hyperbolic cosine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Cosh not implemented")

    @staticmethod
    def _tanh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Hyperbolic tangent."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Tanh not implemented")

    @staticmethod
    def _asinh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Inverse hyperbolic sine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Asinh not implemented")

    @staticmethod
    def _acosh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Inverse hyperbolic cosine."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Acosh not implemented")

    @staticmethod
    def _atanh(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Inverse hyperbolic tangent."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Atanh not implemented")

    @staticmethod
    def _radians(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Convert degrees to radians."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Radians not implemented")

    @staticmethod
    def _degrees(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Convert radians to degrees."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Degrees not implemented")

    # ========================================================================
    # Decomposition
    # ========================================================================

    @staticmethod
    def _ceil(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Ceiling (round up)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Ceil not implemented")

    @staticmethod
    def _floor(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Floor (round down)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Floor not implemented")

    @staticmethod
    def _round(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Round to nearest integer (ties to even)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Round not implemented")

    @staticmethod
    def _fract(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Fractional part (x - floor(x))."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Fract not implemented")

    @staticmethod
    def _trunc(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Truncate (round toward zero)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Trunc not implemented")

    # ========================================================================
    # Exponent
    # ========================================================================

    @staticmethod
    def _exp(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Exponential (e^x)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Exp not implemented")

    @staticmethod
    def _exp2(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Base-2 exponential (2^x)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Exp2 not implemented")

    @staticmethod
    def _log(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Natural logarithm (ln)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Log not implemented")

    @staticmethod
    def _log2(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Base-2 logarithm."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Log2 not implemented")

    @staticmethod
    def _pow(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Power (x^y)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Pow not implemented")

    # ========================================================================
    # Computational
    # ========================================================================

    @staticmethod
    def _sign(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Sign of a value (-1, 0, or 1)."""
        # TODO: Implement with component_wise_signed
        raise NotImplementedError("Sign not implemented")

    @staticmethod
    def _fma(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        arg2: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Fused multiply-add (a * b + c)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Fma not implemented")

    @staticmethod
    def _step(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Step function (0 if x < edge, 1 otherwise)."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Step not implemented")

    @staticmethod
    def _sqrt(eval: ConstantEvaluator, arg: Handle[Expression], span: Span) -> Handle[Expression]:
        """Square root."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("Sqrt not implemented")

    @staticmethod
    def _inverse_sqrt(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Inverse square root (1 / sqrt(x))."""
        # TODO: Implement with component_wise_float
        raise NotImplementedError("InverseSqrt not implemented")

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
        # TODO: Implement with component_wise_concrete_int
        raise NotImplementedError("CountTrailingZeros not implemented")

    @staticmethod
    def _count_leading_zeros(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Count leading zero bits."""
        # TODO: Implement with component_wise_concrete_int
        raise NotImplementedError("CountLeadingZeros not implemented")

    @staticmethod
    def _count_one_bits(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Count set bits."""
        # TODO: Implement with component_wise_concrete_int
        raise NotImplementedError("CountOneBits not implemented")

    @staticmethod
    def _reverse_bits(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Reverse bits."""
        # TODO: Implement with component_wise_concrete_int
        raise NotImplementedError("ReverseBits not implemented")

    @staticmethod
    def _first_trailing_bit(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Find index of first trailing set bit."""
        # TODO: Implement with component_wise_concrete_int
        raise NotImplementedError("FirstTrailingBit not implemented")

    @staticmethod
    def _first_leading_bit(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Find index of first leading set bit."""
        # TODO: Implement with component_wise_concrete_int
        raise NotImplementedError("FirstLeadingBit not implemented")

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
        # TODO: Implement packed dot product
        raise NotImplementedError("Dot4I8Packed not implemented")

    @staticmethod
    def _dot4_u8_packed(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Dot product of two packed u8 vectors."""
        # TODO: Implement packed dot product
        raise NotImplementedError("Dot4U8Packed not implemented")

    @staticmethod
    def _cross(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Cross product of two 3D vectors."""
        # TODO: Implement cross product
        raise NotImplementedError("Cross not implemented")

    @staticmethod
    def _dot(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Dot product of two vectors."""
        # TODO: Implement dot product
        raise NotImplementedError("Dot not implemented")

    @staticmethod
    def _length(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Length of a vector."""
        # TODO: Implement length calculation
        raise NotImplementedError("Length not implemented")

    @staticmethod
    def _distance(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        span: Span,
    ) -> Handle[Expression]:
        """Distance between two points."""
        # TODO: Implement distance calculation
        raise NotImplementedError("Distance not implemented")

    @staticmethod
    def _normalize(
        eval: ConstantEvaluator,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Normalize a vector."""
        # TODO: Implement normalization
        raise NotImplementedError("Normalize not implemented")
