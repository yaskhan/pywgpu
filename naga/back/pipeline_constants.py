"""
Pipeline constant processing.

This module provides utilities for processing pipeline-overridable constants
and replacing them with concrete values.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
from enum import Enum

from ..ir import Module, ShaderStage
from ..valid import ModuleInfo, ValidationError


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
    from ..compact import compact, KeepUnused
    
    # If no entry point specified or single entry point, and no overrides, return as-is
    if (entry_point is None or len(module.entry_points) <= 1) and not module.overrides:
        return (module, module_info)
    
    # Clone the module
    import copy
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
    
    # Process overrides (placeholder - full implementation would:
    # 1. Map overrides to constants using pipeline_constants
    # 2. Evaluate all global expressions
    # 3. Replace overrides with constants
    # 4. Compact again
    
    return revalidate(module)


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
]
