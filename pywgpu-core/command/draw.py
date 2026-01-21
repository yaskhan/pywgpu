"""
Draw command encoding.

This module defines draw commands and errors for render passes. It provides:
- DrawError: Errors related to draw operations
- RenderCommandError: Errors related to render command encoding
- Rect: Rectangle structure for viewport and scissor operations

Draw commands are used to render geometry on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DrawError(Exception):
    """
    Error validating a draw call.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class RenderCommandError(Exception):
    """
    Error encountered when encoding a render command.

    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class Rect:
    """
    Rectangle structure.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        w: Width.
        h: Height.
    """

    x: Any
    y: Any
    w: Any
    h: Any


@dataclass
class Draw:
    """
    Command to draw geometry.

    Attributes:
        vertex_count: Number of vertices to draw.
        instance_count: Number of instances to draw.
        first_vertex: Index of the first vertex.
        first_instance: Index of the first instance.
    """

    vertex_count: int
    instance_count: int
    first_vertex: int
    first_instance: int


@dataclass
class DrawIndexed:
    """
    Command to draw indexed geometry.

    Attributes:
        index_count: Number of indices to draw.
        instance_count: Number of instances to draw.
        first_index: Index of the first index.
        base_vertex: Base vertex offset.
        first_instance: Index of the first instance.
    """

    index_count: int
    instance_count: int
    first_index: int
    base_vertex: int
    first_instance: int


@dataclass
class DrawIndirect:
    """
    Command to draw geometry indirectly.

    Attributes:
        buffer: The buffer containing draw parameters.
        offset: Offset into the buffer.
        count: Number of draws (for multi-draw).
        family: Draw command family.
        vertex_or_index_limit: Vertex or index limit.
        instance_limit: Instance limit.
    """

    buffer: Any
    offset: int
    count: int
    family: Any
    vertex_or_index_limit: Optional[int]
    instance_limit: Optional[int]


@dataclass
class MultiDrawIndirectCount:
    """
    Command to draw geometry indirectly with count.

    Attributes:
        buffer: The buffer containing draw parameters.
        offset: Offset into the buffer.
        count_buffer: Buffer containing the draw count.
        count_buffer_offset: Offset into the count buffer.
        max_count: Maximum number of draws.
        family: Draw command family.
    """

    buffer: Any
    offset: int
    count_buffer: Any
    count_buffer_offset: int
    max_count: int
    family: Any


@dataclass
class SetBlendConstant:
    """
    Command to set blend constant.

    Attributes:
        color: The blend constant color.
    """

    color: Any


@dataclass
class SetStencilReference:
    """
    Command to set stencil reference.

    Attributes:
        reference: The stencil reference value.
    """

    reference: int


@dataclass
class SetViewport:
    """
    Command to set viewport.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        w: Width.
        h: Height.
        min_depth: Minimum depth.
        max_depth: Maximum depth.
    """

    x: float
    y: float
    w: float
    h: float
    min_depth: float
    max_depth: float


@dataclass
class SetScissor:
    """
    Command to set scissor.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        w: Width.
        h: Height.
    """

    x: int
    y: int
    w: int
    h: int


@dataclass
class DrawMeshTasks:
    """
    Command to draw mesh tasks.

    Attributes:
        group_count_x: Workgroup count in X dimension.
        group_count_y: Workgroup count in Y dimension.
        group_count_z: Workgroup count in Z dimension.
    """

    group_count_x: int
    group_count_y: int
    group_count_z: int
