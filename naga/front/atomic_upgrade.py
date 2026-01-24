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
        
    Raises:
        GlobalInitUnsupportedError: If a global variable that needs upgrading has an initializer
    """
    # Map from old types to their upgraded versions
    upgraded_types: dict[Handle[Type], Handle[Type]] = {}
    
    def upgrade_type(ty: Handle[Type]) -> Handle[Type]:
        """
        Get a type equivalent to `ty`, but with Scalar leaves upgraded to Atomic scalars.
        
        If such a type already exists in the module, return its handle.
        Otherwise, construct a new one and return that handle.
        """
        # If we've already upgraded this type, return the cached handle
        if ty in upgraded_types:
            return upgraded_types[ty]
        
        old_type = module.types[ty]
        inner_type = type(old_type.inner).__name__
        
        # Base case: upgrade Scalar to Atomic
        if inner_type == "Scalar":
            # Create an Atomic type with the same scalar
            from ..ir import TypeInner
            new_inner = TypeInner.Atomic(scalar=old_type.inner.scalar)
            new_type = Type(name=old_type.name, inner=new_inner)
            new_handle = module.types.append(new_type, old_type.span)
            upgraded_types[ty] = new_handle
            return new_handle
        
        # Recursive cases: recurse into composite types
        elif inner_type == "Pointer":
            new_base = upgrade_type(old_type.inner.base)
            if new_base == old_type.inner.base:
                # No change needed
                return ty
            from ..ir import TypeInner
            new_inner = TypeInner.Pointer(base=new_base, space=old_type.inner.space)
            new_type = Type(name=old_type.name, inner=new_inner)
            new_handle = module.types.append(new_type, old_type.span)
            upgraded_types[ty] = new_handle
            return new_handle
        
        elif inner_type == "Array":
            new_base = upgrade_type(old_type.inner.base)
            if new_base == old_type.inner.base:
                return ty
            from ..ir import TypeInner
            new_inner = TypeInner.Array(
                base=new_base,
                size=old_type.inner.size,
                stride=old_type.inner.stride
            )
            new_type = Type(name=old_type.name, inner=new_inner)
            new_handle = module.types.append(new_type, old_type.span)
            upgraded_types[ty] = new_handle
            return new_handle
        
        elif inner_type == "BindingArray":
            new_base = upgrade_type(old_type.inner.base)
            if new_base == old_type.inner.base:
                return ty
            from ..ir import TypeInner
            new_inner = TypeInner.BindingArray(base=new_base, size=old_type.inner.size)
            new_type = Type(name=old_type.name, inner=new_inner)
            new_handle = module.types.append(new_type, old_type.span)
            upgraded_types[ty] = new_handle
            return new_handle
        
        elif inner_type == "Struct":
            # Only upgrade fields that have been accessed atomically
            if ty not in upgrades.fields:
                # No fields in this struct were accessed atomically
                return ty
            
            # Clone members and upgrade the accessed fields
            new_members = old_type.inner.members.copy()
            fields_to_upgrade = upgrades.fields[ty]
            for field_index in fields_to_upgrade:
                new_members[field_index].ty = upgrade_type(new_members[field_index].ty)
            
            from ..ir import TypeInner
            new_inner = TypeInner.Struct(members=new_members, span=old_type.inner.span)
            new_type = Type(name=old_type.name, inner=new_inner)
            new_handle = module.types.append(new_type, old_type.span)
            upgraded_types[ty] = new_handle
            return new_handle
        
        # No upgrade needed for other types
        else:
            return ty
    
    # Upgrade all global variables
    for global_handle in upgrades.globals.iter():
        global_var = module.global_variables[global_handle]
        
        # Check for initializers (not supported for atomic variables)
        if hasattr(global_var, 'init') and global_var.init is not None:
            raise GlobalInitUnsupportedError(
                f"Global variable {global_var.name} has an initializer, "
                "which is not supported for atomic variables"
            )
        
        # Upgrade the type
        old_ty = global_var.ty
        new_ty = upgrade_type(old_ty)
        
        if new_ty != old_ty:
            module.global_variables[global_handle].ty = new_ty


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
