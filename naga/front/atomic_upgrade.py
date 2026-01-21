"""
Upgrade the types of scalars observed to be accessed as atomics to Atomic types.

In SPIR-V, atomic operations can be applied to any scalar value, but in Naga
IR atomic operations can only be applied to values of type Atomic. Naga
IR's restriction matches Metal Shading Language and WGSL, so we don't want
to relax that. Instead, when the SPIR-V front end observes a value being
accessed using atomic instructions, it promotes the value's type from
Scalar to Atomic. This module implements Module.upgrade_atomics,
the function that makes that change.

Atomics can only appear in global variables in the Storage and
Workgroup address spaces. These variables can either have Atomic types
themselves, or be Arrays of such, or be Structs containing such.
So we only need to change the types of globals and struct fields.

Naga IR Load expressions and Store statements can operate directly
on Atomic values, retrieving and depositing ordinary Scalar values,
so changing the types doesn't have much effect on the code that operates on
those values.

Future work:
- The GLSL front end could use this transformation as well.
"""

from typing import Any, Set
from ..arena import HandleSet, Handle
from ..ir import GlobalVariable, Type, Module


class Error(Exception):
    """Base class for atomic upgrade errors."""
    pass


class UnsupportedError(Error):
    """Encountered an unsupported expression."""
    pass


class UnexpectedEndOfIndicesError(Error):
    """Unexpected end of struct field access indices."""
    pass


class GlobalInitUnsupportedError(Error):
    """Encountered unsupported global initializer in an atomic variable."""
    pass


class GlobalVariableMissingError(Error):
    """Expected to find a global variable."""
    pass


class CompareExchangeNonScalarBaseTypeError(Error):
    """Atomic compare exchange requires a scalar base type."""
    pass


class Upgrades:
    """
    Tracks which types need to be upgraded to atomic types.
    
    Attributes:
        globals: Global variables that we've accessed using atomic operations.
            This includes globals with composite types (arrays, structs) where we've
            only accessed some components (elements, fields) atomically.
        fields: Struct fields that we've accessed using atomic operations.
            Each key refers to some Struct type, and each value is a set of
            field indices within that struct that we've accessed atomically.
    """
    
    def __init__(self) -> None:
        """Initialize a new Upgrades tracker."""
        self.globals: HandleSet[GlobalVariable] = HandleSet()
        self.fields: dict[Handle[Type], Set[int]] = {}
    
    def note_atomic_global(self, global_var: Handle[GlobalVariable]) -> None:
        """
        Note that we've observed an atomic access to the given global variable.
        
        Args:
            global_var: Handle to the global variable
        """
        self.globals.insert(global_var)
    
    def note_atomic_field(self, type_handle: Handle[Type], field_index: int) -> None:
        """
        Note that we've observed an atomic access to the given struct field.
        
        Args:
            type_handle: Handle to the struct type
            field_index: Index of the field within the struct
        """
        if type_handle not in self.fields:
            self.fields[type_handle] = set()
        self.fields[type_handle].add(field_index)


def upgrade_atomics(module: Module, upgrades: Upgrades) -> None:
    """
    Upgrade types in the module based on observed atomic operations.
    
    This function modifies the module in place, upgrading scalar types to
    atomic types where necessary based on the information in upgrades.
    
    Args:
        module: The module to upgrade
        upgrades: Information about which variables and fields need upgrading
    """
    # Placeholder implementation
    # In a full implementation, this would:
    # 1. Iterate through globals in upgrades.globals
    # 2. For each global, upgrade its type to atomic
    # 3. For struct fields in upgrades.fields, create new struct types
    #    with atomic fields and replace the old types
    pass


__all__ = [
    "Error",
    "UnsupportedError",
    "UnexpectedEndOfIndicesError",
    "GlobalInitUnsupportedError",
    "GlobalVariableMissingError",
    "CompareExchangeNonScalarBaseTypeError",
    "Upgrades",
    "upgrade_atomics",
]
