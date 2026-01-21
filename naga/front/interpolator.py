"""
Interpolation defaults.

This module provides utilities for applying default interpolation to bindings.
"""

from typing import Optional, Any
from ..ir import Binding, TypeInner, Interpolation, Sampling, ScalarKind


def apply_default_interpolation(binding: Binding, ty: TypeInner) -> None:
    """
    Apply the usual default interpolation for ty to binding.
    
    This function is a utility front ends may use to satisfy the Naga IR's
    requirement, meant to ensure that input languages' policies have been
    applied appropriately, that all I/O Bindings from the vertex shader to the
    fragment shader must have non-None interpolation values.
    
    All the shader languages Naga supports have similar rules:
    perspective-correct, center-sampled interpolation is the default for any
    binding that can vary, and everything else either defaults to flat, or
    requires an explicit flat qualifier/attribute/what-have-you.
    
    If binding is not a Location binding, or if its interpolation is
    already set, then make no changes. Otherwise, set binding's interpolation
    and sampling to reasonable defaults depending on ty, the type of the value
    being interpolated:
    
    - If ty is a floating-point scalar, vector, or matrix type, then
      default to Perspective interpolation and Center sampling.
    
    - If ty is an integral scalar or vector, then default to Flat
      interpolation, which has no associated sampling.
    
    - For any other types, make no change. Such types are not permitted as
      user-defined IO values, and will probably be flagged by the verifier
    
    When structs appear in input or output types, each member ought to have its
    own Binding, so structs are simply covered by the third case.
    
    Args:
        binding: The binding to modify
        ty: The type inner of the value being interpolated
    """
    # Check if this is a Location binding with no interpolation set
    if (
        hasattr(binding, "location")
        and hasattr(binding, "interpolation")
        and binding.interpolation is None
    ):
        scalar_kind = ty.scalar_kind() if hasattr(ty, "scalar_kind") else None
        
        if scalar_kind == ScalarKind.FLOAT:
            binding.interpolation = Interpolation.PERSPECTIVE
            binding.sampling = Sampling.CENTER
        elif scalar_kind in (ScalarKind.SINT, ScalarKind.UINT):
            binding.interpolation = Interpolation.FLAT
            binding.sampling = None


__all__ = [
    "apply_default_interpolation",
]
