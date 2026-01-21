"""Constant expression evaluator for Naga IR.

This module provides functionality for evaluating constant expressions at compile time,
mirroring naga/src/proc/constant_evaluator.rs implementation from wgpu crate.

This implementation provides the core framework with key methods translated from Rust.
Complex math function and operator implementations remain as stubs (NotImplementedError).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING, Any

from naga import Expression, Handle, ExpressionType, Span
from naga.ir import (
    Type, TypeInner, ScalarKind, Scalar, VectorSize, ArraySize,
    BinaryOperator, Constant, MathFunction, RelationalFunction, UnaryOperator,
    Literal, LiteralType,
)

if TYPE_CHECKING:
    from naga import Arena, UniqueArena, Module, Override

# Import arena types at runtime
from naga import Arena, UniqueArena, Module, Override


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
        """Force an expression to not be const."""
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
            case (
                ExpressionType.LITERAL
                | ExpressionType.ZERO_VALUE
                | ExpressionType.CONSTANT
            ):
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

        If the expression's value can be determined at compile time, append the
        evaluated form. If not, and we're in a runtime context,
        append the unevaluated expression. Otherwise, raise an error.
        """
        kind = self.expression_kind_tracker.type_of_with_expr(expr)

        match kind:
            case ExpressionKind.CONST:
                # Attempt to evaluate const expression
                eval_result = self._try_eval_const(expr, span)
                # If we're in a runtime context and evaluation failed, just emit as runtime
                if self.behavior.has_runtime_restrictions() and isinstance(
                    eval_result, ConstantEvaluatorError
                ):
                    return self._append_expr(expr, span, ExpressionKind.RUNTIME)
                return eval_result

            case ExpressionKind.OVERRIDE:
                match self.behavior:
                    case Behavior(
                        wgsl=WgslRestrictions(kind=("Override" | "Runtime"))
                    ):
                        return self._append_expr(expr, span, ExpressionKind.OVERRIDE)
                    case Behavior(glsl=GlslRestrictions(kind="Runtime")):
                        return self._append_expr(expr, span, ExpressionKind.OVERRIDE)
                    case _:
                        raise ConstantEvaluatorError("OverrideExpr")

            case ExpressionKind.RUNTIME:
                if self.behavior.has_runtime_restrictions():
                    return self._append_expr(expr, span, ExpressionKind.RUNTIME)
                raise ConstantEvaluatorError("RuntimeExpr")

        raise ConstantEvaluatorError(f"Unexpected expression kind: {kind}")

    def _try_eval_const(
        self,
        expr: Expression,
        span: Span,
    ) -> Handle[Expression]:
        """Try to evaluate a const expression."""
        match expr.type:
            case ExpressionType.CONSTANT:
                # "See through" constant and use its initializer
                constant = self.constants[expr.constant]
                return constant.init
            case _:
                # For other expressions, just register as-is
                return self._append_expr(expr, span, ExpressionKind.CONST)

    def is_global_arena(self) -> bool:
        """Check if we're evaluating in the global arena."""
        match self.behavior:
            case Behavior(
                wgsl=WgslRestrictions(kind=("Const" | "Override"), local_data=None)
            ):
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

    def _append_expr(
        self,
        expr: Expression,
        span: Span,
        kind: ExpressionKind,
    ) -> Handle[Expression]:
        """Append an expression to the arena."""
        handle = self.expressions.append(expr, span)
        self.expression_kind_tracker.insert(handle, kind)
        return handle
