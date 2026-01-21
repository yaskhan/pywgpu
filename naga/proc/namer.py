"""
Namer for assigning identifiers to module elements.

This module provides the Namer class for generating unique identifiers
for variables, functions, types, and other elements in Naga IR.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from enum import Enum
import re

from ..arena import Handle


EntryPointIndex = int
SEPARATOR = '_'


class ExternalTextureNameKey(Enum):
    """
    A component of a lowered external texture.
    
    Whereas the WGSL backend implements ImageClass.External
    images directly, most other Naga backends lower them to a
    collection of ordinary textures that represent individual planes
    (as received from a video decoder, perhaps), together with a
    struct of parameters saying how they should be cropped, sampled,
    and color-converted.
    
    Attributes:
        PLANE_0: First texture plane
        PLANE_1: Second texture plane
        PLANE_2: Third texture plane
        PARAMS: Parameters struct
    """
    PLANE_0 = ("_plane0", 0)
    PLANE_1 = ("_plane1", 1)
    PLANE_2 = ("_plane2", 2)
    PARAMS = ("_params", -1)
    
    def suffix(self) -> str:
        """Get the suffix for this key."""
        return self.value[0]


class NameKey:
    """
    A key for naming various module elements.
    
    This is used to track unique names for constants, variables, functions,
    types, and other IR elements.
    """
    
    def __init__(self, kind: str, handles: tuple = ()) -> None:
        """
        Initialize a NameKey.
        
        Args:
            kind: The kind of element (e.g., "constant", "function", "type")
            handles: Tuple of handles identifying this element
        """
        self.kind = kind
        self.handles = handles
    
    @staticmethod
    def constant(handle: Handle) -> NameKey:
        """Create a key for a constant."""
        return NameKey("constant", (handle,))
    
    @staticmethod
    def override(handle: Handle) -> NameKey:
        """Create a key for an override."""
        return NameKey("override", (handle,))
    
    @staticmethod
    def global_variable(handle: Handle) -> NameKey:
        """Create a key for a global variable."""
        return NameKey("global_variable", (handle,))
    
    @staticmethod
    def type_(handle: Handle) -> NameKey:
        """Create a key for a type."""
        return NameKey("type", (handle,))
    
    @staticmethod
    def struct_member(type_handle: Handle, member_index: int) -> NameKey:
        """Create a key for a struct member."""
        return NameKey("struct_member", (type_handle, member_index))
    
    @staticmethod
    def function(handle: Handle) -> NameKey:
        """Create a key for a function."""
        return NameKey("function", (handle,))
    
    @staticmethod
    def function_argument(func_handle: Handle, arg_index: int) -> NameKey:
        """Create a key for a function argument."""
        return NameKey("function_argument", (func_handle, arg_index))
    
    @staticmethod
    def function_local(func_handle: Handle, local_handle: Handle) -> NameKey:
        """Create a key for a function local variable."""
        return NameKey("function_local", (func_handle, local_handle))
    
    @staticmethod
    def entry_point(index: EntryPointIndex) -> NameKey:
        """Create a key for an entry point."""
        return NameKey("entry_point", (index,))
    
    @staticmethod
    def entry_point_local(index: EntryPointIndex, local_handle: Handle) -> NameKey:
        """Create a key for an entry point local variable."""
        return NameKey("entry_point_local", (index, local_handle))
    
    @staticmethod
    def entry_point_argument(index: EntryPointIndex, arg_index: int) -> NameKey:
        """Create a key for an entry point argument."""
        return NameKey("entry_point_argument", (index, arg_index))
    
    def __hash__(self) -> int:
        """Hash the key."""
        return hash((self.kind, self.handles))
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, NameKey):
            return False
        return self.kind == other.kind and self.handles == other.handles


class Namer:
    """
    This processor assigns names to all the things in a module
    that may need identifiers in a textual backend.
    
    Attributes:
        unique: The last numeric suffix used for each base name. Zero means "no suffix".
        keywords: Set of reserved keywords
        keywords_case_insensitive: Set of case-insensitive keywords
        reserved_prefixes: List of reserved identifier prefixes
    """

    def __init__(
        self,
        keywords: Optional[set] = None,
        keywords_case_insensitive: Optional[set] = None,
    ) -> None:
        """
        Initialize a new Namer.
        
        Args:
            keywords: Set of reserved keywords
            keywords_case_insensitive: Set of case-insensitive keywords
        """
        self.unique: Dict[str, int] = {}
        self.keywords = keywords or set()
        self.keywords_case_insensitive = keywords_case_insensitive or set()
        self.reserved_prefixes: list[str] = []

    def reset(
        self,
        module: Any,
        keywords: Optional[set] = None,
        keywords_case_insensitive: Optional[set] = None,
        reserved_prefixes: Optional[list[str]] = None,
    ) -> None:
        """
        Reset the namer and prepare it for a new module.

        Args:
            module: The Naga IR module to name
            keywords: Set of reserved keywords
            keywords_case_insensitive: Set of case-insensitive keywords
            reserved_prefixes: List of reserved identifier prefixes
        """
        self.unique = {}
        if keywords is not None:
            self.keywords = keywords
        if keywords_case_insensitive is not None:
            self.keywords_case_insensitive = keywords_case_insensitive
        if reserved_prefixes is not None:
            self.reserved_prefixes = reserved_prefixes

    def call(self, label_raw: str) -> str:
        """
        Return a new identifier based on label_raw.

        Args:
            label_raw: The suggested name for the identifier

        Returns:
            A unique identifier string
        """
        base = self.sanitize(label_raw)

        # Check for reserved prefixes
        for prefix in self.reserved_prefixes:
            if base.startswith(prefix):
                base = f"gen_{base}"
                break

        # Make unique
        if base in self.unique:
            self.unique[base] += 1
            return f"{base}{SEPARATOR}{self.unique[base]}"
        else:
            self.unique[base] = 0
            return base

    def sanitize(self, label: str) -> str:
        """
        Return a form of string suitable for use as the base of an identifier.
        
        Args:
            label: The input string
            
        Returns:
            A sanitized identifier base
        """
        # Remove characters that are not alphanumeric or underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        
        # Drop leading digits
        sanitized = re.sub(r"^[0-9]+", "", sanitized)
        
        # Handle empty or still invalid starts
        if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "unnamed" + sanitized
        
        # Check against keywords
        if sanitized.lower() in self.keywords_case_insensitive or sanitized in self.keywords:
            sanitized = sanitized + "_"

        return sanitized


__all__ = [
    "EntryPointIndex",
    "ExternalTextureNameKey",
    "NameKey",
    "Namer",
]
