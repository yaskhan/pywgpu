"""
FFI interface for command encoding.

This module provides FFI (Foreign Function Interface) support for command
encoding. It defines types and structures for referencing wgpu objects
through different mechanisms:
- IdReferences: Reference objects by numeric IDs
- PointerReferences: Reference objects by pointer values (for tracing)
- ArcReferences: Reference objects by Arc references

This is used for trace recording and playback, as well as for FFI bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class ReferenceType:
    """
    Trait for types that can reference wgpu objects.
    
    This trait defines the types used to reference different wgpu objects.
    """

    Buffer: type
    Surface: type
    Texture: type
    TextureView: type
    ExternalTexture: type
    QuerySet: type
    BindGroup: type
    RenderPipeline: type
    RenderBundle: type
    ComputePipeline: type
    Blas: type
    Tlas: type


class IdReferences(ReferenceType):
    """
    Reference wgpu objects via numeric IDs.
    
    This type references objects by their numeric IDs assigned by the
    IdentityManager.
    """

    Buffer: type = Any
    Surface: type = Any
    Texture: type = Any
    TextureView: type = Any
    ExternalTexture: type = Any
    QuerySet: type = Any
    BindGroup: type = Any
    RenderPipeline: type = Any
    RenderBundle: type = Any
    ComputePipeline: type = Any
    Blas: type = Any
    Tlas: type = Any


class PointerReferences(ReferenceType):
    """
    Reference wgpu objects via pointer values.
    
    This type references objects by the integer value of their pointers.
    This is used for trace recording and playback.
    """

    Buffer: type = Any
    Surface: type = Any
    Texture: type = Any
    TextureView: type = Any
    ExternalTexture: type = Any
    QuerySet: type = Any
    BindGroup: type = Any
    RenderPipeline: type = Any
    RenderBundle: type = Any
    ComputePipeline: type = Any
    Blas: type = Any
    Tlas: type = Any


class ArcReferences(ReferenceType):
    """
    Reference wgpu objects via Arc references.
    
    This type references objects by Arc references, which is the most
    common way to reference objects in wgpu-core.
    """

    Buffer: type = Any
    Surface: type = Any
    Texture: type = Any
    TextureView: type = Any
    ExternalTexture: type = Any
    QuerySet: type = Any
    BindGroup: type = Any
    RenderPipeline: type = Any
    RenderBundle: type = Any
    ComputePipeline: type = Any
    Blas: type = Any
    Tlas: type = Any


@dataclass
class Command(Generic[T]):
    """
    Command for encoding.
    
    This enum represents different types of commands that can be recorded
    into a command encoder.
    
    Attributes:
        copy_buffer_to_buffer: Copy data between buffers.
        copy_buffer_to_texture: Copy data from buffer to texture.
        copy_texture_to_buffer: Copy data from texture to buffer.
        copy_texture_to_texture: Copy data between textures.
        clear_buffer: Clear a buffer.
        clear_texture: Clear a texture.
        write_timestamp: Write a timestamp.
        resolve_query_set: Resolve a query set.
        push_debug_group: Push a debug group.
        pop_debug_group: Pop a debug group.
        insert_debug_marker: Insert a debug marker.
        run_compute_pass: Run a compute pass.
        run_render_pass: Run a render pass.
        build_acceleration_structures: Build acceleration structures.
        transition_resources: Transition resources.
    """

    copy_buffer_to_buffer: Optional[Any] = None
    copy_buffer_to_texture: Optional[Any] = None
    copy_texture_to_buffer: Optional[Any] = None
    copy_texture_to_texture: Optional[Any] = None
    clear_buffer: Optional[Any] = None
    clear_texture: Optional[Any] = None
    write_timestamp: Optional[Any] = None
    resolve_query_set: Optional[Any] = None
    push_debug_group: Optional[str] = None
    pop_debug_group: Optional[bool] = None
    insert_debug_marker: Optional[str] = None
    run_compute_pass: Optional[Any] = None
    run_render_pass: Optional[Any] = None
    build_acceleration_structures: Optional[Any] = None
    transition_resources: Optional[Any] = None


@dataclass
class ArcCommand(Command[ArcReferences]):
    """Command with Arc references."""
    pass
