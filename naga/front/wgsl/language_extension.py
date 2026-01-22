"""
Language extension definitions.

Translated from wgpu-trunk/naga/src/front/wgsl/parse/directive/language_extension.rs

This module defines language extensions that affect parsing behavior.
"""

from enum import Enum
from typing import Set


class ImplementedLanguageExtension(Enum):
    """
    Language extensions that are implemented.
    
    These extensions change how the language is parsed.
    """
    # Allow pointer composite access (ptr.field, ptr[index])
    POINTER_COMPOSITE_ACCESS = "pointer_composite_access"


class UnimplementedLanguageExtension(Enum):
    """
    Language extensions that are recognized but not yet implemented.
    """
    # Packed 4x8 integer dot product
    PACKED_4X8_INTEGER_DOT_PRODUCT = "packed_4x8_integer_dot_product"
    
    # Unrestricted pointer parameters
    UNRESTRICTED_POINTER_PARAMETERS = "unrestricted_pointer_parameters"
    
    # Subgroup operations
    SUBGROUPS = "subgroups"
    
    # Subgroup operations with f16
    SUBGROUPS_F16 = "subgroups_f16"


class LanguageExtensionSet:
    """
    Set of active language extensions.
    
    Tracks which language extensions are currently active during parsing.
    """
    
    def __init__(self):
        """Initialize with no extensions active."""
        self._active: Set[ImplementedLanguageExtension] = set()
    
    def activate(self, extension: ImplementedLanguageExtension) -> None:
        """
        Activate a language extension.
        
        Args:
            extension: Extension to activate
        """
        self._active.add(extension)
    
    def is_active(self, extension: ImplementedLanguageExtension) -> bool:
        """
        Check if a language extension is active.
        
        Args:
            extension: Extension to check
            
        Returns:
            True if active
        """
        return extension in self._active
    
    def __contains__(self, extension: ImplementedLanguageExtension) -> bool:
        """Support 'in' operator."""
        return self.is_active(extension)
    
    def get_all(self) -> Set[ImplementedLanguageExtension]:
        """
        Get all active extensions.
        
        Returns:
            Set of active extensions
        """
        return self._active.copy()


def parse_language_extension(name: str, span: tuple[int, int]) -> ImplementedLanguageExtension:
    """
    Parse a language extension name.
    
    Args:
        name: Extension name
        span: Source location
        
    Returns:
        ImplementedLanguageExtension
        
    Raises:
        ParseError: If extension is unknown or unimplemented
    """
    from .error import ParseError
    
    # Try implemented extensions
    try:
        return ImplementedLanguageExtension(name)
    except ValueError:
        pass
    
    # Try unimplemented extensions
    try:
        unimpl = UnimplementedLanguageExtension(name)
        raise ParseError(
            message=f"language extension '{name}' is not implemented",
            labels=[(span[0], span[1], "")],
            notes=[f"This extension is recognized but not yet supported"]
        )
    except ValueError:
        pass
    
    # Unknown extension
    valid_extensions = [e.value for e in ImplementedLanguageExtension]
    raise ParseError(
        message=f"unknown language extension: '{name}'",
        labels=[(span[0], span[1], "")],
        notes=[f"Valid extensions: {', '.join(valid_extensions)}"]
    )


def get_extension_behavior(extension: ImplementedLanguageExtension) -> dict:
    """
    Get the parsing behavior changes for an extension.
    
    Args:
        extension: Extension to query
        
    Returns:
        Dictionary describing behavior changes
    """
    behaviors = {
        ImplementedLanguageExtension.POINTER_COMPOSITE_ACCESS: {
            'description': 'Allow member access and indexing on pointers',
            'affects': ['expression_parsing'],
            'allows': [
                'ptr.field',
                'ptr[index]',
            ],
        },
    }
    
    return behaviors.get(extension, {})
