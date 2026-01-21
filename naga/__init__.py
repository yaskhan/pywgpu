"""
Naga can be used to translate source code written in one shading language to another.

The central structure is the Module, which contains:
- Functions with arguments, return types, local variables, and bodies
- EntryPoints which are specialized functions for pipeline stages  
- Constants and GlobalVariables used by EntryPoints and Functions
- Types used by the above
"""

from .arena import Arena, Handle, Range, UniqueArena
from .span import Span, WithSpan  # SourceLocation, SpanContext would be added here
from .diagnostic_filter import Severity, DiagnosticFilter

# Re-export IR items for convenience (matching pub use ir::* from lib.rs)
from .ir import *

# Submodules
from . import back, common, compact, front, keywords, proc, valid

# Type aliases for performance (matching Rust FastHashMap/FastHashSet patterns)
# In Python, we use dict and set which are already optimized
type FastHashMap = dict
type FastHashSet = set  
type FastIndexMap = dict  # Could use OrderedDict if order matters
type FastIndexSet = set  # Could use OrderedSet if order matters

# Width constants
BOOL_WIDTH = 1  # Width of a boolean type, in bytes
ABSTRACT_WIDTH = 8  # Width of abstract types, in bytes

__all__ = [
    # Arena types
    "Arena", "Handle", "Range", "UniqueArena",
    # Span types
    "Span", "WithSpan",
    # Diagnostic types
    "Severity", "DiagnosticFilter", 
    # IR re-exports (inherited from .ir import *)
    # Performance type aliases
    "FastHashMap", "FastHashSet", "FastIndexMap", "FastIndexSet",
    # Constants
    "BOOL_WIDTH", "ABSTRACT_WIDTH",
    # Submodules
    "back", "common", "compact", "front", "keywords", "proc", "valid"
]
