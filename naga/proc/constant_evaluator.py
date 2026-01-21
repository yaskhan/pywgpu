"""Constant expression evaluator for Naga IR (Fixed to match actual IR structure).

This module provides functionality for evaluating constant expressions at compile time,
mirroring naga/src/proc/constant_evaluator.rs implementation from wgpu crate.

Note: This is an updated version that matches the actual Expression dataclass structure
defined in naga/ir/expression.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntFlag
from typing import TYPE_CHECKING, Any, Iterable, Literal as L, TypeVar, Union

from naga import Expression, Handle, ExpressionType, Span
from naga.ir import (
    Type, TypeInner, ScalarKind, Scalar, VectorSize, ArraySize,
    BinaryOperator, Constant, MathFunction, RelationalFunction, UnaryOperator,
    Literal, LiteralType,
)

if TYPE_CHECKING:
    from naga import Arena, UniqueArena, Module, Override
    from naga.arena import HandleVec

# Import arena types at runtime
from naga import Arena, UniqueArena, Module, Override
from naga.arena import HandleVec


# ============================================================================
# Expression Kind Tracking
# ============================================================================

class ExpressionKind(IntFlag):
    """Kind of expression: Const, Override, or Runtime."""

    CONST = 1
    OVERRIDE = 2
    RUNTIME = 3


class ConstantEvaluatorError(Exception):
    """Errors that can occur during constant expression evaluation."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ExpressionKindTracker:
    """Tracks the constness of expressions residing in an arena."""

    def __init__(self) -> None:
        self.inner: dict[int, ExpressionKind] = {}

    def force_non_const(self, handle: Handle[Expression]) -> None:
        """Force the expression to not be const."""
        self.inner[handle] = ExpressionKind.RUNTIME

    def insert(self, handle: Handle[Expression], kind: ExpressionKind) -> None:
        """Insert a new expression with its kind."""
        self.inner[handle] = kind

    def is_const(self, handle: Handle[Expression]) -> bool:
        """Check if expression is const."""
        return self.type_of(handle) == ExpressionKind.CONST

    def is_const_or_override(self, handle: Handle[Expression]) -> bool:
        """Check if expression is const or override."""
        kind = self.type_of(handle)
        return kind in (ExpressionKind.CONST, ExpressionKind.OVERRIDE)

    def type_of(self, handle: Handle[Expression]) -> ExpressionKind:
        """Get the kind of an expression."""
        return self.inner[handle]

    @classmethod
    def from_arena(cls, arena: Arena[Expression]) -> ExpressionKindTracker:
        """Create a tracker from an existing expression arena."""
        tracker = cls()
        for handle, expr in arena.iter():
            kind = tracker.type_of_with_expr(expr)
            tracker.insert(handle, kind)
        return tracker

    def type_of_with_expr(self, expr: Expression) -> ExpressionKind:
        """Determine the kind of an expression from its structure."""
        match expr.type:
            case ExpressionType.LITERAL | ExpressionType.ZERO_VALUE | ExpressionType.CONSTANT:
                return ExpressionKind.CONST
            case ExpressionType.OVERRIDE:
                return ExpressionKind.OVERRIDE
            case ExpressionType.COMPOSE:
                if expr.compose_components:
                    result = ExpressionKind.CONST
                    for component in expr.compose_components:
                        result = max(result, self.type_of(component))
                    return result
                return ExpressionKind.CONST
            case ExpressionType.SPLAT:
                return self.type_of(expr.splat_value)
            case ExpressionType.ACCESS_INDEX:
                return self.type_of(expr.access_base)
            case ExpressionType.ACCESS:
                result = self.type_of(expr.access_base)
                if expr.access_index is not None:
                    result = max(result, self.type_of(expr.access_index))
                return result
            case ExpressionType.SWIZZLE:
                return self.type_of(expr.swizzle_vector)
            case ExpressionType.UNARY:
                return self.type_of(expr.unary_expr)
            case ExpressionType.BINARY:
                left_kind = self.type_of(expr.binary_left)
                right_kind = self.type_of(expr.binary_right)
                return max(left_kind, right_kind)
            case ExpressionType.MATH:
                result = self.type_of(expr.math_arg)
                for arg in [expr.math_arg1, expr.math_arg2, expr.math_arg3]:
                    if arg is not None:
                        result = max(result, self.type_of(arg))
                return result
            case ExpressionType.AS:
                return self.type_of(expr.as_expr)
            case ExpressionType.SELECT:
                condition_kind = self.type_of(expr.select_condition)
                accept_kind = self.type_of(expr.select_accept)
                reject_kind = self.type_of(expr.select_reject)
                return max(condition_kind, accept_kind, reject_kind)
            case ExpressionType.RELATIONAL:
                return self.type_of(expr.relational_argument)
            case ExpressionType.ARRAY_LENGTH:
                return self.type_of(expr.array_length)
            case _:
                return ExpressionKind.RUNTIME


# ============================================================================
# Wgsl and Glsl Restrictions
# ============================================================================

@dataclass
class WgslRestrictions:
    """WGSL-specific restrictions for constant evaluation."""
    kind: L["Const", "Override", "Runtime"]
    local_data: FunctionLocalData | None = None


@dataclass
class GlslRestrictions:
    """GLSL-specific restrictions for constant evaluation."""
    kind: L["Const", "Runtime"]
    local_data: FunctionLocalData | None = None


@dataclass
class Behavior:
    """Language-specific evaluation behavior."""
    wgsl: WgslRestrictions | None = None
    glsl: GlslRestrictions | None = None

    def has_runtime_restrictions(self) -> bool:
        """Returns True if this behavior has runtime restrictions."""
        if self.wgsl is not None and self.wgsl.kind == "Runtime":
            return True
        if self.glsl is not None and self.glsl.kind == "Runtime":
            return True
        return False


@dataclass
class FunctionLocalData:
    """Data specific to function-local constant evaluation."""
    global_expressions: Arena[Expression]
    emitter: Any  # Emitter - will be defined later
    block: Any  # Block - will be defined later


# ============================================================================
# Constant Evaluator
# ============================================================================

class ConstantEvaluator:
    """Evaluates constant expressions at compile time.

    This class provides methods to try to evaluate expressions and append
    them to an expression arena. If an expression's value can be determined
    at compile time, it's reduced to a tree of Literal, Compose, ZeroValue,
    and Swizzle expressions.
    """

    def __init__(
        self,
        behavior: Behavior,
        types: UniqueArena[Type],
        constants: Arena[Constant],
        overrides: Arena[Override],
        expressions: Arena[Expression],
        expression_kind_tracker: ExpressionKindTracker,
        layouter: Any,  # Layouter - will be defined later
    ) -> None:
        self.behavior = behavior
        self.types = types
        self.constants = constants
        self.overrides = overrides
        self.expressions = expressions
        self.expression_kind_tracker = expression_kind_tracker
        self.layouter = layouter

    @classmethod
    def for_wgsl_module(
        cls,
        module: Module,
        global_expression_kind_tracker: ExpressionKindTracker,
        layouter: Any,
        in_override_ctx: bool,
    ) -> ConstantEvaluator:
        """Create a ConstantEvaluator for WGSL module constant expressions."""
        if in_override_ctx:
            behavior = Behavior(wgsl=WgslRestrictions("Override"))
        else:
            behavior = Behavior(wgsl=WgslRestrictions("Const"))

        return cls(
            behavior=behavior,
            types=module.types,
            constants=module.constants,
            overrides=module.overrides,
            expressions=module.global_expressions,
            expression_kind_tracker=global_expression_kind_tracker,
            layouter=layouter,
        )

    @classmethod
    def for_glsl_module(
        cls,
        module: Module,
        global_expression_kind_tracker: ExpressionKindTracker,
        layouter: Any,
    ) -> ConstantEvaluator:
        """Create a ConstantEvaluator for GLSL module constant expressions."""
        behavior = Behavior(glsl=GlslRestrictions("Const"))

        return cls(
            behavior=behavior,
            types=module.types,
            constants=module.constants,
            overrides=module.overrides,
            expressions=module.global_expressions,
            expression_kind_tracker=global_expression_kind_tracker,
            layouter=layouter,
        )

    def to_ctx(self) -> dict[str, Any]:
        """Return global context for this evaluator."""
        return {
            "types": self.types,
            "constants": self.constants,
            "overrides": self.overrides,
            "global_expressions": self.expressions,
        }

    def check(self, expr: Handle[Expression]) -> None:
        """Check that an expression is const."""
        if not self.expression_kind_tracker.is_const(expr):
            raise ConstantEvaluatorError("SubexpressionsAreNotConstant")

    def try_eval_and_append(
        self,
        expr: Expression,
        span: Span,
    ) -> Handle[Expression]:
        """Try to evaluate an expression and append it to the arena.

        If the expression can be evaluated at compile time, append the
        evaluated form. If not, and we're in a runtime context,
        append the unevaluated expression. Otherwise, raise an error.
        """
        kind = self.expression_kind_tracker.type_of_with_expr(expr)

        match kind:
            case ExpressionKind.CONST:
                eval_result = self.try_eval_and_append_impl(expr, span)
                # If we're in a runtime context and evaluation failed due to
                # unimplemented features, just emit as runtime expression
                if self.behavior.has_runtime_restrictions() and isinstance(
                    eval_result, ConstantEvaluatorError
                ):
                    return self.append_expr(expr, span, ExpressionKind.RUNTIME)
                return eval_result

            case ExpressionKind.OVERRIDE:
                match self.behavior:
                    case Behavior(wgsl=WgslRestrictions(kind=("Override" | "Runtime"))):
                        return self.append_expr(expr, span, ExpressionKind.OVERRIDE)
                    case Behavior(glsl=GlslRestrictions(kind="Runtime")):
                        return self.append_expr(expr, span, ExpressionKind.OVERRIDE)
                    case _:
                        raise ConstantEvaluatorError("OverrideExpr")

            case ExpressionKind.RUNTIME:
                if self.behavior.has_runtime_restrictions():
                    return self.append_expr(expr, span, ExpressionKind.RUNTIME)
                raise ConstantEvaluatorError("RuntimeExpr")

        raise ConstantEvaluatorError(f"Unexpected expression kind: {kind}")

    def try_eval_and_append_impl(
        self,
        expr: Expression,
        span: Span,
    ) -> Handle[Expression]:
        """Implementation of expression evaluation."""
        match expr.type:
            case ExpressionType.CONSTANT:
                if self.is_global_arena():
                    # "See through" constant and use its initializer
                    constant = self.constants[expr.constant]
                    return constant.init  # type: ignore
                # Otherwise, just pass through
                return self.register_evaluated_expr(expr, span)

            case ExpressionType.OVERRIDE:
                raise ConstantEvaluatorError("Override")

            case ExpressionType.LITERAL | ExpressionType.ZERO_VALUE:
                return self.register_evaluated_expr(expr, span)

            case ExpressionType.COMPOSE:
                evaluated_components = [
                    self.check_and_get(comp) for comp in (expr.compose_components or [])
                ]
                new_expr = Expression(
                    type=expr.type,
                    compose_ty=expr.compose_ty,
                    compose_components=evaluated_components,
                )
                return self.register_evaluated_expr(new_expr, span)

            case ExpressionType.SPLAT:
                evaluated_value = self.check_and_get(expr.splat_value)
                new_expr = Expression(
                    type=expr.type,
                    splat_size=expr.splat_size,
                    splat_value=evaluated_value,
                )
                return self.register_evaluated_expr(new_expr, span)

            case ExpressionType.ACCESS_INDEX:
                evaluated_base = self.check_and_get(expr.access_base)
                # For AccessIndex, index is a u32 value
                index = expr.access_index_value or 0
                return self.access(evaluated_base, index, span)

            case ExpressionType.ACCESS:
                evaluated_base = self.check_and_get(expr.access_base)
                if expr.access_index is not None:
                    evaluated_index = self.check_and_get(expr.access_index)
                    # Get the constant value from the index
                    index_expr = self.expressions[evaluated_index]
                    if index_expr.type == ExpressionType.LITERAL:
                        index = index_expr.literal  # type: ignore
                        if isinstance(index, int):
                            return self.access(evaluated_base, index, span)
                raise ConstantEvaluatorError("InvalidAccessIndexTy")

            case ExpressionType.SWIZZLE:
                evaluated_vector = self.check_and_get(expr.swizzle_vector)
                pattern = expr.swizzle_pattern or []
                return self.swizzle(
                    expr.swizzle_size,
                    span,
                    evaluated_vector,
                    pattern,
                )

            case ExpressionType.UNARY:
                evaluated_expr = self.check_and_get(expr.unary_expr)
                return self.unary_op(expr.unary_op, evaluated_expr, span)

            case ExpressionType.BINARY:
                evaluated_left = self.check_and_get(expr.binary_left)
                evaluated_right = self.check_and_get(expr.binary_right)
                return self.binary_op(
                    expr.binary_op, evaluated_left, evaluated_right, span
                )

            case ExpressionType.MATH:
                evaluated_arg = self.check_and_get(expr.math_arg)
                evaluated_arg1 = (
                    self.check_and_get(expr.math_arg1)
                    if expr.math_arg1 is not None
                    else None
                )
                evaluated_arg2 = (
                    self.check_and_get(expr.math_arg2)
                    if expr.math_arg2 is not None
                    else None
                )
                evaluated_arg3 = (
                    self.check_and_get(expr.math_arg3)
                    if expr.math_arg3 is not None
                    else None
                )
                return self.math(
                    evaluated_arg,
                    evaluated_arg1,
                    evaluated_arg2,
                    evaluated_arg3,
                    expr.math_fun,
                    span,
                )

            case ExpressionType.AS:
                evaluated_expr = self.check_and_get(expr.as_expr)
                if expr.as_convert is not None:
                    # TODO: Implement cast with Scalar construction
                    raise NotImplementedError("Cast with convert not implemented")
                raise NotImplementedError("Bitcast not implemented")

            case ExpressionType.SELECT:
                evaluated_reject = self.check_and_get(expr.select_reject)
                evaluated_accept = self.check_and_get(expr.select_accept)
                evaluated_condition = self.check_and_get(expr.select_condition)
                return self.select(
                    evaluated_reject,
                    evaluated_accept,
                    evaluated_condition,
                    span,
                )

            case ExpressionType.RELATIONAL:
                evaluated_argument = self.check_and_get(expr.relational_argument)
                return self.relational(expr.relational_fun, evaluated_argument, span)

            case ExpressionType.ARRAY_LENGTH:
                match self.behavior:
                    case Behavior(wgsl=_):
                        raise ConstantEvaluatorError("ArrayLength not supported in WGSL")
                    case Behavior(glsl=_):
                        evaluated_expr = self.check_and_get(expr.array_length)
                        return self.array_length(evaluated_expr, span)

            case _:
                # Not supported expression types
                raise NotImplementedError(
                    f"Expression type {expr.type} not implemented for constant evaluation"
                )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def is_global_arena(self) -> bool:
        """Check if we're evaluating in the global arena."""
        match self.behavior:
            case Behavior(wgsl=WgslRestrictions(kind=("Const" | "Override"), local_data=None)):
                return True
            case Behavior(glsl=GlslRestrictions(kind="Const", local_data=None)):
                return True
            case _:
                return False

    def function_local_data(self) -> FunctionLocalData | None:
        """Get function-local data if available."""
        match self.behavior:
            case Behavior(wgsl=WgslRestrictions(local_data=data)):
                return data
            case Behavior(glsl=GlslRestrictions(local_data=data)):
                return data
            case _:
                return None

    def append_expr(
        self,
        expr: Expression,
        span: Span,
        expr_kind: ExpressionKind,
    ) -> Handle[Expression]:
        """Append an expression to the arena."""
        handle = self.expressions.append(expr, span)
        self.expression_kind_tracker.insert(handle, expr_kind)
        return handle

    def register_evaluated_expr(
        self,
        expr: Expression,
        span: Span,
    ) -> Handle[Expression]:
        """Register an evaluated expression."""
        # Validate literal values if applicable
        if expr.type == ExpressionType.LITERAL:
            # TODO: Implement check_literal_value
            pass

        # Validate vector compose lengths
        if expr.type == ExpressionType.COMPOSE:
            if expr.compose_ty is not None:
                ty_inner = self.types[expr.compose_ty].inner
                if isinstance(ty_inner, TypeInner.Vector):
                    expected = ty_inner.size.value  # type: ignore
                    actual = self.vector_compose_flattened_size(expr.compose_components or [])
                    if expected != actual:
                        raise ConstantEvaluatorError(
                            "InvalidVectorComposeLength",
                            {"expected": expected, "actual": actual},
                        )

        return self.append_expr(expr, span, ExpressionKind.CONST)

    def vector_compose_flattened_size(
        self, components: list[Handle[Expression]]
    ) -> int:
        """Calculate total number of components in a vector compose."""
        total = 0
        for comp in components:
            comp_expr = self.expressions[comp]
            match comp_expr.type:
                case ExpressionType.LITERAL | ExpressionType.ZERO_VALUE:
                    total += 1
                case ExpressionType.COMPOSE:
                    if comp_expr.compose_ty is not None:
                        inner = self.types[comp_expr.compose_ty].inner
                        if isinstance(inner, TypeInner.Vector):
                            total += inner.size.value  # type: ignore
                        elif isinstance(inner, TypeInner.Scalar):
                            total += 1
                case _:
                    raise ConstantEvaluatorError("InvalidVectorComposeComponent")
        return total

    # ========================================================================
    # Evaluation Method Stubs (TODO implementations)
    # ========================================================================

    def check_and_get(self, expr: Handle[Expression]) -> Handle[Expression]:
        """Check an expression and return its evaluated form."""
        self.check(expr)
        return expr

    def unary_op(
        self,
        op: UnaryOperator,
        expr: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Evaluate a unary operation."""
        # TODO: Implement unary operation evaluation
        raise NotImplementedError(f"Unary operation {op} not implemented")

    def binary_op(
        self,
        op: BinaryOperator,
        left: Handle[Expression],
        right: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Evaluate a binary operation."""
        # TODO: Implement binary operation evaluation
        raise NotImplementedError(f"Binary operation {op} not implemented")

    def math(
        self,
        arg: Handle[Expression],
        arg1: Handle[Expression] | None,
        arg2: Handle[Expression] | None,
        arg3: Handle[Expression] | None,
        fun: MathFunction,
        span: Span,
    ) -> Handle[Expression]:
        """Evaluate a math function."""
        # TODO: Implement math function evaluation
        raise NotImplementedError(f"Math function {fun} not implemented")

    def relational(
        self,
        fun: RelationalFunction,
        arg: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Evaluate a relational function."""
        # TODO: Implement relational function evaluation
        raise NotImplementedError(f"Relational function {fun} not implemented")

    def select(
        self,
        reject: Handle[Expression],
        accept: Handle[Expression],
        condition: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Evaluate a select expression."""
        # TODO: Implement select evaluation
        raise NotImplementedError("Select not implemented")

    def access(
        self,
        base: Handle[Expression],
        index: int,
        span: Span,
    ) -> Handle[Expression]:
        """Access a component of a composite value."""
        # TODO: Implement access evaluation
        raise NotImplementedError("Access not implemented")

    def swizzle(
        self,
        size: VectorSize,
        span: Span,
        src_constant: Handle[Expression],
        pattern: list[Any],
    ) -> Handle[Expression]:
        """Swizzle components of a vector."""
        # TODO: Implement swizzle evaluation
        raise NotImplementedError("Swizzle not implemented")

    def array_length(
        self,
        array: Handle[Expression],
        span: Span,
    ) -> Handle[Expression]:
        """Get the length of an array."""
        # TODO: Implement array_length evaluation
        raise NotImplementedError("ArrayLength not implemented")
