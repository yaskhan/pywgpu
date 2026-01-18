"""naga.valid.module_info

This module defines the Python equivalents of Naga's `ModuleInfo` and related
analysis structures.

The Rust implementation carries rich type resolution, uniformity analysis, and
resource usage information. The Python port currently keeps a minimal but
structurally-compatible representation that backends can query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .flags import TypeFlags


@dataclass(slots=True)
class ExpressionInfo:
    """Holds information about a validated expression.

    Attributes:
        ty: Backend-specific type information for the expression.
        uniformity: Backend-specific uniformity information.
    """

    ty: Optional[object] = None
    uniformity: Optional[object] = None


@dataclass(slots=True)
class FunctionInfo:
    """Holds information about a validated function."""

    expressions: list[ExpressionInfo] = field(default_factory=list)
    uniformity: Optional[object] = None
    may_kill: bool = False
    sampling: Optional[object] = None


@dataclass(slots=True)
class ModuleInfo:
    """Information about a validated module.

    This is returned by :meth:`naga.valid.validator.Validator.validate`.

    The goal here is API compatibility with the Rust `naga::valid::ModuleInfo`:
    - `type_flags` can be indexed by type handle (here: integer indices)
    - `functions` contains per-function `FunctionInfo`
    - `entry_points` contains per-entry-point `FunctionInfo`

    Since the Python IR currently uses integer indices as handles, `ModuleInfo`
    helpers accept `int` indices.
    """

    type_flags: list[TypeFlags] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    entry_points: list[FunctionInfo] = field(default_factory=list)
    const_expression_types: list[object] = field(default_factory=list)

    def get_type_flags(self, type_handle: int) -> TypeFlags:
        """Return flags for the given type handle.

        Args:
            type_handle: Index into the module's type arena.

        Raises:
            IndexError: If `type_handle` is out of bounds.
        """

        return self.type_flags[type_handle]

    def get_function_info(self, function_handle: int) -> FunctionInfo:
        """Return info for the given function handle.

        Args:
            function_handle: Index into the module's function arena.

        Raises:
            IndexError: If `function_handle` is out of bounds.
        """

        return self.functions[function_handle]

    def get_entry_point_info(self, entry_point_index: int) -> FunctionInfo:
        """Return info for the given entry point index.

        Args:
            entry_point_index: Index into the module's entry point list.

        Raises:
            IndexError: If `entry_point_index` is out of bounds.
        """

        return self.entry_points[entry_point_index]

    def get_const_expression_type(self, expr_handle: int) -> object:
        """Return the type information for a constant expression.

        Args:
            expr_handle: Index into the module's global expressions list.

        Raises:
            IndexError: If `expr_handle` is out of bounds.
        """

        return self.const_expression_types[expr_handle]
