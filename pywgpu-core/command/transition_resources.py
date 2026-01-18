"""
Resource transition management.

This module implements resource transition management for wgpu-core. It provides:
- TransitionResources: Command to transition resources between different usage states

Resource transitions are used to ensure proper synchronization and memory
consistency when resources are used in different ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class TransitionResources:
    """
    Command to transition resources.
    
    Attributes:
        buffer_transitions: Buffer transitions.
        texture_transitions: Texture transitions.
    """

    buffer_transitions: List[Any]
    texture_transitions: List[Any]


@dataclass
class ArcTransitionResources:
    """
    Command to transition resources with Arc references.
    
    Attributes:
        buffer_transitions: Buffer transitions.
        texture_transitions: Texture transitions.
    """

    buffer_transitions: List[Any]
    texture_transitions: List[Any]


def transition_resources(
    state: Any,
    buffer_transitions: List[Any],
    texture_transitions: List[Any],
) -> None:
    """
    Transition resources between different usage states.
    
    This function encodes resource transition barriers to ensure proper
    synchronization and memory consistency when resources are used in
    different ways. It uses HAL command encoder methods to record the
    necessary barriers.
    
    Args:
        state: The encoding state containing the command encoder.
        buffer_transitions: Buffer transitions (list of BufferBarrier).
        texture_transitions: Texture transitions (list of TextureBarrier).
    
    Raises:
        TransitionError: If transition fails.
    """
    # Import HAL
    try:
        import sys
        import os
        _hal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pywgpu-hal')
        if _hal_path not in sys.path:
            sys.path.insert(0, _hal_path)
        import lib as hal
    except ImportError:
        raise RuntimeError("pywgpu_hal module not available for resource transitions")
    
    # Get the command encoder from state
    if not hasattr(state, 'encoder') and not hasattr(state, 'raw_encoder'):
        raise RuntimeError("Encoding state does not have a command encoder")
    
    encoder = getattr(state, 'encoder', None) or getattr(state, 'raw_encoder', None)
    
    # Transition buffers
    if buffer_transitions and hasattr(encoder, 'transition_buffers'):
        try:
            # Call HAL method to transition buffers
            # The encoder expects an iterator of buffer barriers
            encoder.transition_buffers(iter(buffer_transitions))
        except Exception as e:
            raise RuntimeError(f"Failed to transition buffers: {e}") from e
    
    # Transition textures
    if texture_transitions and hasattr(encoder, 'transition_textures'):
        try:
            # Call HAL method to transition textures
            # The encoder expects an iterator of texture barriers
            encoder.transition_textures(iter(texture_transitions))
        except Exception as e:
            raise RuntimeError(f"Failed to transition textures: {e}") from e

