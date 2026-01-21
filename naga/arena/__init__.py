"""
Arena types for Naga shader translation.

This module provides memory-efficient storage for shader translation components
using strongly-typed handles for referencing.
"""

from .handle import Handle
from .range import Range
from .unique_arena import UniqueArena
from .arena import Arena

__all__ = ["Handle", "Range", "UniqueArena", "Arena"]
