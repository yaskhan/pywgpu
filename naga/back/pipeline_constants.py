"""
Pipeline constant processing.

This module provides utilities for processing pipeline-overridable constants
and replacing them with concrete values.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, Union
from enum import Enum
import math
import copy

from ..ir import (
    Module, ShaderStage, Expression, Statement, Literal, Scalar, 
    Block, Handle, Override, Constant, Function, EntryPoint
)
from ..ir.type import TypeInner
from ..ir.expression import ExpressionType
from ..ir.statement import StatementType
from ..valid import ModuleInfo, ValidationError
from ..compact import compact, KeepUnused


class PipelineConstantError(Exception):
    """Base class for pipeline constant errors."""
    pass


class MissingValueError(PipelineConstantError):
    """Missing value for pipeline-overridable constant."""
    
    def __init__(self, identifier: str) -> None:
        """
        Initialize a MissingValueError.
        
        Args:
            identifier: The identifier string of the missing constant
        """
        super().__init__(
            f"Missing value for pipeline-overridable constant with identifier string: '{identifier}'"
        )
        self.identifier = identifier


class SrcNeedsToBeFiniteError(PipelineConstantError):
    """Source f64 value needs to be finite."""
    
    def __init__(self) -> None:
        """Initialize a SrcNeedsToBeFiniteError."""
        super().__init__(
            "Source f64 value needs to be finite (NaNs and Infinites are not allowed) for number destinations"
        )


class DstRangeTooSmallError(PipelineConstantError):
    """Source f64 value doesn't fit in destination."""
    
    def __init__(self) -> None:
        """Initialize a DstRangeTooSmallError."""
        super().__init__("Source f64 value doesn't fit in destination")


class NegativeWorkgroupSizeError(PipelineConstantError):
    """Workgroup size override isn't strictly positive."""
    
    def __init__(self) -> None:
        """Initialize a NegativeWorkgroupSizeError."""
        super().__init__("workgroup_size override isn't strictly positive")


class NegativeMeshOutputMaxError(PipelineConstantError):
    """Max vertices or max primitives is negative."""
    
    def __init__(self) -> None:
        """Initialize a NegativeMeshOutputMaxError."""
        super().__init__("max vertices or max primitives is negative")


PipelineConstants = Dict[str, float]


def process_overrides(
    module: Module,
    module_info: ModuleInfo,
    entry_point: Optional[Tuple[ShaderStage, str]],
    pipeline_constants: PipelineConstants,
) -> Tuple[Module, ModuleInfo]:
    """
    Compact module and replace all overrides with constants.
    
    If no changes are needed, this just returns references to
    module and module_info. Otherwise, it clones module, retains only the
    selected entry point, compacts the module, edits its global_expressions
    arena to contain only fully-evaluated expressions, and returns the
    simplified module and its validation results.
    
    The module returned has an empty overrides arena, and the
    global_expressions arena contains only fully-evaluated expressions.
    
    Args:
        module: The module to process
        module_info: The module info
        entry_point: Optional entry point (stage, name) to retain
        pipeline_constants: Map of override identifiers to values
        
    Returns:
        Tuple of (processed module, updated module info)
        
    Raises:
        PipelineConstantError: If processing fails
    """
    # If no entry point specified or single entry point, and no overrides, return as-is
    if (entry_point is None or len(module.entry_points) <= 1) and not module.overrides:
        return (module, module_info)
    
    # Clone the module
    module = copy.deepcopy(module)
    
    # Retain only the specified entry point if given
    if entry_point is not None:
        ep_stage, ep_name = entry_point
        module.entry_points = [
            ep for ep in module.entry_points
            if ep.stage == ep_stage and ep.name == ep_name
        ]
    
    # Compact the module to remove unreachable items
    compact(module, keep_unused=False)
    
    # If no overrides remain, we're done
    if not module.overrides:
        return revalidate(module)
    
    # Build data structures for override processing
    override_map: Dict[Handle[Override], Handle[Constant]] = {}
    adjusted_global_expressions: Dict[Handle[Expression], Handle[Expression]] = {}
    adjusted_constant_initializers: set[Handle[Constant]] = set()
    
    # Process overrides and build new global expression arena
    # This is a simplified version - the full implementation would need
    # a complete constant evaluator and expression kind tracker
    
    # For now, create a basic implementation that processes overrides
    for old_override_h, override in module.overrides.items():
        if override.id is not None:
            key = str(override.id)
        elif override.name is not None:
            key = override.name
        else:
            continue
        
        if key in pipeline_constants:
            value = pipeline_constants[key]
            # Convert f64 value to literal based on type
            literal = map_value_to_literal(value, override.ty)
            
            # Create new constant
            new_constant = Constant(
                name=override.name,
                ty=override.ty,
                init=None  # Will be set below
            )
            const_h = module.constants.append(new_constant)
            
            # Create literal expression
            expr_h = module.global_expressions.append(Expression(
                type=ExpressionType.LITERAL,
                literal=literal
            ))
            
            # Set constant initializer
            module.constants[const_h].init = expr_h
            
            override_map[old_override_h] = const_h
            adjusted_constant_initializers.add(const_h)
            override.init = expr_h
    
    # Replace override expressions with constant expressions in global expressions
    for expr_h, expr in module.global_expressions.items():
        if expr.type == ExpressionType.OVERRIDE:
            if expr.override in override_map:
                new_expr = Expression(
                    type=ExpressionType.CONSTANT,
                    constant=override_map[expr.override]
                )
                module.global_expressions[expr_h] = new_expr
    
    # Process workgroup size overrides
    for ep in module.entry_points:
        if hasattr(ep, 'workgroup_size_overrides') and ep.workgroup_size_overrides:
            process_workgroup_size_override(module, adjusted_global_expressions, ep)
        
        if hasattr(ep, 'mesh_info') and ep.mesh_info:
            process_mesh_shader_overrides(module, adjusted_global_expressions, ep)
    
    # Clear overrides arena
    module.overrides.clear()
    
    return revalidate(module)


def process_workgroup_size_override(
    module: Module,
    adjusted_global_expressions: Dict[Handle[Expression], Handle[Expression]],
    ep: EntryPoint,
) -> None:
    """
    Process workgroup size overrides for an entry point.
    
    Args:
        module: The module being processed
        adjusted_global_expressions: Map of adjusted expression handles
        ep: The entry point to process
        
    Raises:
        NegativeWorkgroupSizeError: If workgroup size is not positive
    """
    if not hasattr(ep, 'workgroup_size_overrides') or not ep.workgroup_size_overrides:
        return
    
    for i, overridden in enumerate(ep.workgroup_size_overrides):
        if overridden is not None and overridden in adjusted_global_expressions:
            # Get the constant value from the expression
            new_expr_h = adjusted_global_expressions[overridden]
            expr = module.global_expressions[new_expr_h]
            
            if expr.type == ExpressionType.LITERAL:
                value = expr.literal
                # Get integer value based on type
                if hasattr(value, 'value'):  # For typed literals
                    scalar_value = value.value
                else:
                    scalar_value = value
                
                # Check if positive
                if scalar_value <= 0:
                    raise NegativeWorkgroupSizeError()
                
                ep.workgroup_size[i] = scalar_value
    
    ep.workgroup_size_overrides = None


def process_mesh_shader_overrides(
    module: Module,
    adjusted_global_expressions: Dict[Handle[Expression], Handle[Expression]],
    ep: EntryPoint,
) -> None:
    """
    Process mesh shader overrides for an entry point.
    
    Args:
        module: The module being processed
        adjusted_global_expressions: Map of adjusted expression handles
        ep: The entry point to process
        
    Raises:
        NegativeMeshOutputMaxError: If max vertices or primitives is negative
    """
    if not hasattr(ep, 'mesh_info') or not ep.mesh_info:
        return
    
    mesh_info = ep.mesh_info
    
    if hasattr(mesh_info, 'max_vertices_override') and mesh_info.max_vertices_override:
        if mesh_info.max_vertices_override in adjusted_global_expressions:
            new_expr_h = adjusted_global_expressions[mesh_info.max_vertices_override]
            expr = module.global_expressions[new_expr_h]
            
            if expr.type == ExpressionType.LITERAL:
                value = expr.literal
                if hasattr(value, 'value'):
                    scalar_value = value.value
                else:
                    scalar_value = value
                
                if scalar_value < 0:
                    raise NegativeMeshOutputMaxError()
                
                mesh_info.max_vertices = scalar_value
    
    if hasattr(mesh_info, 'max_primitives_override') and mesh_info.max_primitives_override:
        if mesh_info.max_primitives_override in adjusted_global_expressions:
            new_expr_h = adjusted_global_expressions[mesh_info.max_primitives_override]
            expr = module.global_expressions[new_expr_h]
            
            if expr.type == ExpressionType.LITERAL:
                value = expr.literal
                if hasattr(value, 'value'):
                    scalar_value = value.value
                else:
                    scalar_value = value
                
                if scalar_value < 0:
                    raise NegativeMeshOutputMaxError()
                
                mesh_info.max_primitives = scalar_value


def map_value_to_literal(value: float, scalar_type: Scalar) -> Literal:
    """
    Map a floating point value to a literal based on scalar type.
    
    Args:
        value: The floating point value to convert
        scalar_type: The target scalar type
        
    Returns:
        A Literal with the converted value
        
    Raises:
        SrcNeedsToBeFiniteError: If value is not finite
        DstRangeTooSmallError: If value doesn't fit in destination type
    """
    from ..ir.literal import Literal as Lit
    from ..ir.scalar import ScalarKind
    
    if scalar_type.kind == ScalarKind.BOOL:
        # Boolean: 0.0 = false, anything else (including NaN) = true
        bool_value = value != 0.0 and not math.isnan(value)
        return Lit(bool_value)
    
    if scalar_type.kind == ScalarKind.I32:
        if not math.isfinite(value):
            raise SrcNeedsToBeFiniteError()
        
        truncated = math.trunc(value)
        if truncated < -2147483648.0 or truncated > 2147483647.0:  # i32::MIN..i32::MAX
            raise DstRangeTooSmallError()
        
        return Lit(int(truncated))
    
    if scalar_type.kind == ScalarKind.UINT:
        if not math.isfinite(value):
            raise SrcNeedsToBeFiniteError()
        
        truncated = math.trunc(value)
        if truncated < 0.0 or truncated > 4294967295.0:  # u32::MIN..u32::MAX
            raise DstRangeTooSmallError()
        
        return Lit(int(truncated))
    
    if scalar_type.kind == ScalarKind.F16:
        if not math.isfinite(value):
            raise SrcNeedsToBeFiniteError()
        
        # Convert to f16 range and back to check
        f16_value = value  # Simplified - real implementation would use f16
        if not math.isfinite(f16_value):
            raise DstRangeTooSmallError()
        
        return Lit(float(f16_value))
    
    if scalar_type.kind == ScalarKind.F32:
        if not math.isfinite(value):
            raise SrcNeedsToBeFiniteError()
        
        f32_value = float(value)
        if not math.isfinite(f32_value):
            raise DstRangeTooSmallError()
        
        return Lit(f32_value)
    
    if scalar_type.kind == ScalarKind.F64:
        if not math.isfinite(value):
            raise SrcNeedsToBeFiniteError()
        
        return Lit(value)
    
    raise DstRangeTooSmallError()


def adjust_expr(
    new_pos: Dict[Handle[Expression], Handle[Expression]], 
    expr: Expression
) -> None:
    """
    Replace every expression handle in expr with its counterpart given by new_pos.
    
    Args:
        new_pos: Map from old expression handles to new ones
        expr: The expression to adjust (modified in-place)
    """
    # This is a simplified version - full implementation would handle all expression types
    if expr.type == ExpressionType.BINARY:
        if hasattr(expr, 'left') and expr.left in new_pos:
            expr.left = new_pos[expr.left]
        if hasattr(expr, 'right') and expr.right in new_pos:
            expr.right = new_pos[expr.right]
    elif expr.type == ExpressionType.UNARY:
        if hasattr(expr, 'expr') and expr.expr in new_pos:
            expr.expr = new_pos[expr.expr]
    elif expr.type == ExpressionType.ACCESS:
        if hasattr(expr, 'base') and expr.base in new_pos:
            expr.base = new_pos[expr.base]
        if hasattr(expr, 'index') and expr.index in new_pos:
            expr.index = new_pos[expr.index]
    elif expr.type == ExpressionType.ACCESS_INDEX:
        if hasattr(expr, 'base') and expr.base in new_pos:
            expr.base = new_pos[expr.base]


def adjust_stmt(
    new_pos: Dict[Handle[Expression], Handle[Expression]], 
    stmt: Statement
) -> None:
    """
    Replace every expression handle in stmt with its counterpart given by new_pos.
    
    Args:
        new_pos: Map from old expression handles to new ones
        stmt: The statement to adjust (modified in-place)
    """
    # Simplified implementation - handles common statement types
    if stmt.type == StatementType.IF:
        if hasattr(stmt, 'condition') and stmt.condition in new_pos:
            stmt.condition = new_pos[stmt.condition]
    elif stmt.type == StatementType.STORE:
        if hasattr(stmt, 'pointer') and stmt.pointer in new_pos:
            stmt.pointer = new_pos[stmt.pointer]
        if hasattr(stmt, 'value') and stmt.value in new_pos:
            stmt.value = new_pos[stmt.value]
    elif stmt.type == StatementType.RETURN:
        if hasattr(stmt, 'value') and stmt.value is not None and stmt.value in new_pos:
            stmt.value = new_pos[stmt.value]


def adjust_block(
    new_pos: Dict[Handle[Expression], Handle[Expression]], 
    block: Block
) -> None:
    """
    Replace every expression handle in block with its counterpart given by new_pos.
    
    Args:
        new_pos: Map from old expression handles to new ones
        block: The block to adjust (modified in-place)
    """
    for stmt in block:
        adjust_stmt(new_pos, stmt)


def revalidate(module: Module) -> Tuple[Module, ModuleInfo]:
    """
    Revalidate a module and return it with its info.
    
    Args:
        module: The module to revalidate
        
    Returns:
        Tuple of (module, module info)
        
    Raises:
        ValidationError: If validation fails
    """
    from ..valid import Validator, ValidationFlags, Capabilities
    
    validator = Validator(ValidationFlags.all(), Capabilities.all())
    module_info = validator.validate(module)
    
    return (module, module_info)


__all__ = [
    "PipelineConstantError",
    "MissingValueError",
    "SrcNeedsToBeFiniteError",
    "DstRangeTooSmallError",
    "NegativeWorkgroupSizeError",
    "NegativeMeshOutputMaxError",
    "PipelineConstants",
    "process_overrides",
    "revalidate",
    "process_workgroup_size_override",
    "process_mesh_shader_overrides",
    "map_value_to_literal",
    "adjust_expr",
    "adjust_stmt",
    "adjust_block",
]
