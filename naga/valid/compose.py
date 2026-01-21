"""
Validation for compose expressions.

This module provides validation logic for Compose expressions that construct
values from components (vectors from scalars, matrices from vectors, etc).
"""

from __future__ import annotations

from typing import Iterator
import logging

from ..arena import Handle
from ..ir import Type, TypeInner, Scalar, VectorSize, ArraySize
from ..proc import TypeResolution, GlobalCtx

log = logging.getLogger(__name__)


class ComposeError(Exception):
    """Base class for compose validation errors."""
    pass


class ComposeTypeError(ComposeError):
    """Composing of the given type can't be done."""
    
    def __init__(self, type_handle: Handle[Type]) -> None:
        """
        Initialize a ComposeTypeError.
        
        Args:
            type_handle: Handle to the type that can't be composed
        """
        super().__init__(f"Composing of type {type_handle!r} can't be done")
        self.type_handle = type_handle


class ComposeComponentCountError(ComposeError):
    """Wrong number of components provided for composition."""
    
    def __init__(self, given: int, expected: int) -> None:
        """
        Initialize a ComposeComponentCountError.
        
        Args:
            given: Number of components provided
            expected: Number of components expected
        """
        super().__init__(f"Composing expects {expected} components but {given} were given")
        self.given = given
        self.expected = expected


class ComposeComponentTypeError(ComposeError):
    """Component type doesn't match expected type."""
    
    def __init__(self, index: int) -> None:
        """
        Initialize a ComposeComponentTypeError.
        
        Args:
            index: Index of the component with wrong type
        """
        super().__init__(f"Composing {index}'s component type is not expected")
        self.index = index


def validate_compose(
    self_ty_handle: Handle[Type],
    gctx: GlobalCtx,
    component_resolutions: Iterator[TypeResolution],
) -> None:
    """
    Validate a Compose expression.
    
    Checks that the components provided to a Compose expression are valid
    for constructing a value of the target type.
    
    Args:
        self_ty_handle: Handle to the type being composed
        gctx: Global context for type resolution
        component_resolutions: Iterator over component type resolutions
        
    Raises:
        ComposeTypeError: If the type can't be composed
        ComposeComponentCountError: If wrong number of components
        ComposeComponentTypeError: If component type is wrong
    """
    # Convert iterator to list to get length
    component_list = list(component_resolutions)
    
    ty = gctx.types[self_ty_handle]
    inner = ty.inner
    
    if isinstance(inner, TypeInner.Vector):
        # Vectors are composed from scalars or other vectors
        size = inner.size
        scalar = inner.scalar
        total = 0
        
        for index, comp_res in enumerate(component_list):
            comp_inner = comp_res.inner_with(gctx.types)
            
            if isinstance(comp_inner, TypeInner.Scalar):
                if comp_inner.scalar == scalar:
                    total += 1
                else:
                    log.error(f"Vector component[{index}] type {comp_inner!r}, building {scalar!r}")
                    raise ComposeComponentTypeError(index)
            elif isinstance(comp_inner, TypeInner.Vector):
                if comp_inner.scalar == scalar:
                    total += comp_inner.size.value()
                else:
                    log.error(f"Vector component[{index}] type {comp_inner!r}, building {scalar!r}")
                    raise ComposeComponentTypeError(index)
            else:
                log.error(f"Vector component[{index}] type {comp_inner!r}, building {scalar!r}")
                raise ComposeComponentTypeError(index)
        
        if size.value() != total:
            raise ComposeComponentCountError(given=total, expected=size.value())
    
    elif isinstance(inner, TypeInner.Matrix):
        # Matrices are composed from column vectors
        columns = inner.columns
        rows = inner.rows
        scalar = inner.scalar
        expected_inner = TypeInner.Vector(size=rows, scalar=scalar)
        
        if columns.value() != len(component_list):
            raise ComposeComponentCountError(
                expected=columns.value(),
                given=len(component_list),
            )
        
        for index, comp_res in enumerate(component_list):
            comp_inner = comp_res.inner_with(gctx.types)
            if comp_inner != expected_inner:
                log.error(f"Matrix component[{index}] type {comp_res!r}")
                raise ComposeComponentTypeError(index)
    
    elif isinstance(inner, TypeInner.Array):
        # Arrays are composed from elements of the base type
        if isinstance(inner.size, ArraySize.Constant):
            count = inner.size.value
            
            if count != len(component_list):
                raise ComposeComponentCountError(
                    expected=count,
                    given=len(component_list),
                )
            
            base_resolution = TypeResolution.Handle(inner.base)
            for index, comp_res in enumerate(component_list):
                if not gctx.compare_types(base_resolution, comp_res):
                    log.error(f"Array component[{index}] type {comp_res!r}")
                    raise ComposeComponentTypeError(index)
        else:
            log.error(f"Composing of {inner!r}")
            raise ComposeTypeError(self_ty_handle)
    
    elif isinstance(inner, TypeInner.Struct):
        # Structs are composed from members
        members = inner.members
        
        if len(members) != len(component_list):
            raise ComposeComponentCountError(
                given=len(component_list),
                expected=len(members),
            )
        
        for index, (member, comp_res) in enumerate(zip(members, component_list)):
            member_resolution = TypeResolution.Handle(member.ty)
            if not gctx.compare_types(member_resolution, comp_res):
                log.error(f"Struct component[{index}] type {comp_res!r}")
                raise ComposeComponentTypeError(index)
    
    else:
        log.error(f"Composing of {inner!r}")
        raise ComposeTypeError(self_ty_handle)


__all__ = [
    "ComposeError",
    "ComposeTypeError",
    "ComposeComponentCountError",
    "ComposeComponentTypeError",
    "validate_compose",
]
