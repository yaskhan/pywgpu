"""Expression validation for NAGA validator.

This module provides comprehensive expression validation including:
- Type checking
- Constant expression validation
- Expression kind tracking
- Integration with constant evaluator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..arena import Handle, Arena
from ..ir import Expression, ExpressionType, Literal, Type, TypeInner
from ..proc import ExpressionKind, ExpressionKindTracker
from .errors import ValidationError
from .module_info import ModuleInfo

if TYPE_CHECKING:
    from ..ir.module import Module


class ExpressionValidator:
    """Validates expressions in a function or module context."""
    
    def __init__(
        self,
        module: Module,
        mod_info: ModuleInfo,
        expression_arena: Arena[Expression],
    ) -> None:
        """Initialize expression validator.
        
        Args:
            module: The module being validated
            mod_info: Module validation info
            expression_arena: Arena containing expressions to validate
        """
        self.module = module
        self.mod_info = mod_info
        self.expressions = expression_arena
        self.kind_tracker = ExpressionKindTracker()
    
    def validate_expression(
        self,
        expr_handle: Handle[Expression],
    ) -> None:
        """Validate a single expression.
        
        Args:
            expr_handle: Handle to the expression to validate
            
        Raises:
            ValidationError: If the expression is invalid
        """
        if expr_handle < 0 or expr_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid expression handle {expr_handle} "
                f"(expressions={len(self.expressions)})"
            )
        
        expr = self.expressions[expr_handle]
        
        # Validate based on expression type
        match expr.type:
            case ExpressionType.LITERAL:
                self._validate_literal(expr)
            case ExpressionType.CONSTANT:
                self._validate_constant_ref(expr)
            case ExpressionType.ZERO_VALUE:
                self._validate_zero_value(expr)
            case ExpressionType.COMPOSE:
                self._validate_compose(expr)
            case ExpressionType.SPLAT:
                self._validate_splat(expr)
            case ExpressionType.SWIZZLE:
                self._validate_swizzle(expr)
            case ExpressionType.ACCESS:
                self._validate_access(expr)
            case ExpressionType.ACCESS_INDEX:
                self._validate_access_index(expr)
            case ExpressionType.UNARY:
                self._validate_unary(expr)
            case ExpressionType.BINARY:
                self._validate_binary(expr)
            case ExpressionType.SELECT:
                self._validate_select(expr)
            case ExpressionType.MATH:
                self._validate_math(expr)
            case ExpressionType.RELATIONAL:
                self._validate_relational(expr)
            case _:
                # Other expression types - basic validation
                pass
        
        # Track expression kind
        kind = self.kind_tracker.type_of_with_expr(expr)
        self.kind_tracker.insert(expr_handle, kind)
    
    def _validate_literal(self, expr: Expression) -> None:
        """Validate a literal expression."""
        if not hasattr(expr, 'literal') or expr.literal is None:
            raise ValidationError("Literal expression missing literal value")
        
        # Literal is always valid if it exists
        # Type checking happens at construction time
    
    def _validate_constant_ref(self, expr: Expression) -> None:
        """Validate a constant reference expression."""
        if not hasattr(expr, 'constant'):
            raise ValidationError("Constant expression missing constant handle")
        
        const_handle = expr.constant
        if not isinstance(const_handle, int):
            raise ValidationError(
                f"Constant handle must be int, got {type(const_handle)}"
            )
        
        if const_handle < 0 or const_handle >= len(self.module.constants):
            raise ValidationError(
                f"Invalid constant handle {const_handle} "
                f"(constants={len(self.module.constants)})"
            )
    
    def _validate_zero_value(self, expr: Expression) -> None:
        """Validate a zero value expression."""
        if not hasattr(expr, 'zero_value_ty'):
            raise ValidationError("ZeroValue expression missing type handle")
        
        ty_handle = expr.zero_value_ty
        if not isinstance(ty_handle, int):
            raise ValidationError(
                f"ZeroValue type handle must be int, got {type(ty_handle)}"
            )
        
        if ty_handle < 0 or ty_handle >= len(self.module.types):
            raise ValidationError(
                f"Invalid type handle {ty_handle} "
                f"(types={len(self.module.types)})"
            )
        
        # Check if type is constructible
        if self.mod_info.type_flags:
            from .flags import TypeFlags
            flags = self.mod_info.type_flags[ty_handle]
            if not (flags & TypeFlags.CONSTRUCTIBLE):
                raise ValidationError(
                    f"ZeroValue type {ty_handle} is not constructible"
                )
    
    def _validate_compose(self, expr: Expression) -> None:
        """Validate a compose expression."""
        if not hasattr(expr, 'compose_ty'):
            raise ValidationError("Compose expression missing type handle")
        
        ty_handle = expr.compose_ty
        if not isinstance(ty_handle, int):
            raise ValidationError(
                f"Compose type handle must be int, got {type(ty_handle)}"
            )
        
        if ty_handle < 0 or ty_handle >= len(self.module.types):
            raise ValidationError(
                f"Invalid type handle {ty_handle} "
                f"(types={len(self.module.types)})"
            )
        
        # Validate components
        if not hasattr(expr, 'compose_components'):
            raise ValidationError("Compose expression missing components")
        
        components = expr.compose_components
        if not isinstance(components, list):
            raise ValidationError(
                f"Compose components must be list, got {type(components)}"
            )
        
        # Validate each component handle
        for i, comp in enumerate(components):
            if not isinstance(comp, int):
                raise ValidationError(
                    f"Compose component {i} must be int, got {type(comp)}"
                )
            if comp < 0 or comp >= len(self.expressions):
                raise ValidationError(
                    f"Compose component {i} has invalid handle {comp}"
                )
    
    def _validate_splat(self, expr: Expression) -> None:
        """Validate a splat expression."""
        if not hasattr(expr, 'splat_value'):
            raise ValidationError("Splat expression missing value handle")
        
        value_handle = expr.splat_value
        if not isinstance(value_handle, int):
            raise ValidationError(
                f"Splat value handle must be int, got {type(value_handle)}"
            )
        
        if value_handle < 0 or value_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid splat value handle {value_handle}"
            )
    
    def _validate_swizzle(self, expr: Expression) -> None:
        """Validate a swizzle expression."""
        if not hasattr(expr, 'swizzle_vector'):
            raise ValidationError("Swizzle expression missing vector handle")
        
        vector_handle = expr.swizzle_vector
        if not isinstance(vector_handle, int):
            raise ValidationError(
                f"Swizzle vector handle must be int, got {type(vector_handle)}"
            )
        
        if vector_handle < 0 or vector_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid swizzle vector handle {vector_handle}"
            )
    
    def _validate_access(self, expr: Expression) -> None:
        """Validate a dynamic access expression."""
        if not hasattr(expr, 'access_base'):
            raise ValidationError("Access expression missing base handle")
        
        base_handle = expr.access_base
        if not isinstance(base_handle, int):
            raise ValidationError(
                f"Access base handle must be int, got {type(base_handle)}"
            )
        
        if base_handle < 0 or base_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid access base handle {base_handle}"
            )
        
        # Validate index if present
        if hasattr(expr, 'access_index') and expr.access_index is not None:
            index_handle = expr.access_index
            if not isinstance(index_handle, int):
                raise ValidationError(
                    f"Access index handle must be int, got {type(index_handle)}"
                )
            
            if index_handle < 0 or index_handle >= len(self.expressions):
                raise ValidationError(
                    f"Invalid access index handle {index_handle}"
                )
    
    def _validate_access_index(self, expr: Expression) -> None:
        """Validate a constant index access expression."""
        if not hasattr(expr, 'access_base'):
            raise ValidationError("AccessIndex expression missing base handle")
        
        base_handle = expr.access_base
        if not isinstance(base_handle, int):
            raise ValidationError(
                f"AccessIndex base handle must be int, got {type(base_handle)}"
            )
        
        if base_handle < 0 or base_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid AccessIndex base handle {base_handle}"
            )
        
        # Validate index value
        if not hasattr(expr, 'access_index'):
            raise ValidationError("AccessIndex expression missing index value")
        
        index = expr.access_index
        if not isinstance(index, int):
            raise ValidationError(
                f"AccessIndex index must be int, got {type(index)}"
            )
        
        if index < 0:
            raise ValidationError(
                f"AccessIndex index must be non-negative, got {index}"
            )
    
    def _validate_unary(self, expr: Expression) -> None:
        """Validate a unary operation expression."""
        if not hasattr(expr, 'unary_expr'):
            raise ValidationError("Unary expression missing operand handle")
        
        operand_handle = expr.unary_expr
        if not isinstance(operand_handle, int):
            raise ValidationError(
                f"Unary operand handle must be int, got {type(operand_handle)}"
            )
        
        if operand_handle < 0 or operand_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid unary operand handle {operand_handle}"
            )
    
    def _validate_binary(self, expr: Expression) -> None:
        """Validate a binary operation expression."""
        if not hasattr(expr, 'binary_left'):
            raise ValidationError("Binary expression missing left operand")
        
        if not hasattr(expr, 'binary_right'):
            raise ValidationError("Binary expression missing right operand")
        
        left_handle = expr.binary_left
        right_handle = expr.binary_right
        
        if not isinstance(left_handle, int):
            raise ValidationError(
                f"Binary left handle must be int, got {type(left_handle)}"
            )
        
        if not isinstance(right_handle, int):
            raise ValidationError(
                f"Binary right handle must be int, got {type(right_handle)}"
            )
        
        if left_handle < 0 or left_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid binary left handle {left_handle}"
            )
        
        if right_handle < 0 or right_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid binary right handle {right_handle}"
            )
    
    def _validate_select(self, expr: Expression) -> None:
        """Validate a select (ternary) expression."""
        if not hasattr(expr, 'select_condition'):
            raise ValidationError("Select expression missing condition")
        
        if not hasattr(expr, 'select_accept'):
            raise ValidationError("Select expression missing accept value")
        
        if not hasattr(expr, 'select_reject'):
            raise ValidationError("Select expression missing reject value")
        
        cond_handle = expr.select_condition
        accept_handle = expr.select_accept
        reject_handle = expr.select_reject
        
        for name, handle in [
            ('condition', cond_handle),
            ('accept', accept_handle),
            ('reject', reject_handle)
        ]:
            if not isinstance(handle, int):
                raise ValidationError(
                    f"Select {name} handle must be int, got {type(handle)}"
                )
            
            if handle < 0 or handle >= len(self.expressions):
                raise ValidationError(
                    f"Invalid select {name} handle {handle}"
                )
    
    def _validate_math(self, expr: Expression) -> None:
        """Validate a math function expression."""
        if not hasattr(expr, 'math_arg'):
            raise ValidationError("Math expression missing first argument")
        
        arg_handle = expr.math_arg
        if not isinstance(arg_handle, int):
            raise ValidationError(
                f"Math arg handle must be int, got {type(arg_handle)}"
            )
        
        if arg_handle < 0 or arg_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid math arg handle {arg_handle}"
            )
        
        # Validate additional arguments if present
        for i, attr in enumerate(['math_arg1', 'math_arg2', 'math_arg3'], 1):
            if hasattr(expr, attr):
                arg = getattr(expr, attr)
                if arg is not None:
                    if not isinstance(arg, int):
                        raise ValidationError(
                            f"Math arg{i} handle must be int, got {type(arg)}"
                        )
                    
                    if arg < 0 or arg >= len(self.expressions):
                        raise ValidationError(
                            f"Invalid math arg{i} handle {arg}"
                        )
    
    def _validate_relational(self, expr: Expression) -> None:
        """Validate a relational function expression."""
        if not hasattr(expr, 'relational_argument'):
            raise ValidationError("Relational expression missing argument")
        
        arg_handle = expr.relational_argument
        if not isinstance(arg_handle, int):
            raise ValidationError(
                f"Relational arg handle must be int, got {type(arg_handle)}"
            )
        
        if arg_handle < 0 or arg_handle >= len(self.expressions):
            raise ValidationError(
                f"Invalid relational arg handle {arg_handle}"
            )


__all__ = ['ExpressionValidator']
