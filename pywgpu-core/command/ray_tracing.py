"""
Ray tracing command encoding.

This module implements ray tracing commands for wgpu-core. It provides:
- BuildAccelerationStructures: Command to build acceleration structures
- Ray tracing command encoding support

Ray tracing commands are used to build and update acceleration structures
for efficient ray tracing on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class BuildAccelerationStructures:
    """
    Command to build acceleration structures.

    Attributes:
        blas: List of bottom-level acceleration structures to build.
        tlas: List of top-level acceleration structures to build.
    """

    blas: List[Any]
    tlas: List[Any]


@dataclass
class ArcBuildAccelerationStructures:
    """
    Command to build acceleration structures with Arc references.

    Attributes:
        blas: List of bottom-level acceleration structures to build.
        tlas: List of top-level acceleration structures to build.
    """

    blas: List[Any]
    tlas: List[Any]


def encode_build_acceleration_structures(
    state: Any,
    blas: List[Any],
    tlas: List[Any],
) -> None:
    """
    Encode acceleration structure building commands.

    This function encodes commands to build bottom-level and top-level
    acceleration structures on the GPU. The structures are built using
    the HAL command encoder.

    Args:
        state: The encoding state containing the command encoder.
        blas: Bottom-level acceleration structures to build.
        tlas: Top-level acceleration structures to build.

    Raises:
        BuildAccelerationStructureError: If building fails.
    """
    # Import HAL
    try:
        import sys
        import os

        _hal_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pywgpu-hal"
        )
        if _hal_path not in sys.path:
            sys.path.insert(0, _hal_path)
        import lib as hal
    except ImportError:
        raise RuntimeError("pywgpu_hal module not available for ray tracing")

    # Get the command encoder from state
    if not hasattr(state, "encoder") and not hasattr(state, "raw_encoder"):
        raise RuntimeError("Encoding state does not have a command encoder")

    encoder = getattr(state, "encoder", None) or getattr(state, "raw_encoder", None)

    # Build BLAS (Bottom-Level Acceleration Structures)
    for blas_entry in blas:
        if not hasattr(encoder, "build_acceleration_structure"):
            # Fallback: log warning if encoder doesn't support ray tracing
            import warnings

            warnings.warn(
                "Command encoder does not support acceleration structure building"
            )
            continue

        try:
            # Call HAL method to build BLAS
            encoder.build_acceleration_structure(blas_entry)
        except Exception as e:
            raise RuntimeError(f"Failed to build BLAS: {e}") from e

    # Build TLAS (Top-Level Acceleration Structures)
    for tlas_entry in tlas:
        if not hasattr(encoder, "build_acceleration_structure"):
            import warnings

            warnings.warn(
                "Command encoder does not support acceleration structure building"
            )
            continue

        try:
            # Call HAL method to build TLAS
            encoder.build_acceleration_structure(tlas_entry)
        except Exception as e:
            raise RuntimeError(f"Failed to build TLAS: {e}") from e
