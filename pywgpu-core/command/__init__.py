"""
Command management for wgpu-core.

This module implements command encoding and execution for wgpu-core. It provides:
- Command allocators for managing command encoders
- Bind group management for resource binding
- Render bundle encoding and execution
- Clear operations for buffers and textures
- Compute pass encoding
- Draw command encoding
- Command encoder management
- FFI interface for command encoding
- Memory initialization tracking
- Pass management (compute and render)
- Query management
- Ray tracing command encoding
- Render pass encoding
- Timestamp writes management
- Transfer operations
- Resource transition management

Commands are recorded into command encoders and then submitted to a queue for
execution on the GPU.
"""

from __future__ import annotations

from . import allocator
from . import bind
from . import bundle
from . import clear
from . import compute
from . import compute_command
from . import draw
from . import encoder
from . import encoder_command
from . import ffi
from . import memory_init
from . import pass_module
from . import query
from . import ray_tracing
from . import render
from . import render_command
from . import timestamp_writes
from . import transfer
from . import transition_resources

__all__ = [
    "allocator",
    "bind",
    "bundle",
    "clear",
    "compute",
    "compute_command",
    "draw",
    "encoder",
    "encoder_command",
    "ffi",
    "memory_init",
    "pass_module",
    "query",
    "ray_tracing",
    "render",
    "render_command",
    "timestamp_writes",
    "transfer",
    "transition_resources",
]
