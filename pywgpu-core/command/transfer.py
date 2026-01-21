"""
Transfer operations for command encoding.

This module implements transfer operations for wgpu-core, including validation
and command creation for copying data between buffers and textures.
"""

from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from . import errors
from .. import wgpu_core, wgt


class CopySide(enum.Enum):
    """Specifies the source or destination side of a copy operation."""

    Source = "source"
    Destination = "destination"


# A comprehensive set of error types for transfer validation.
# In a real implementation, each of these would be a custom exception class.
class TransferError(Exception):
    """Base class for all transfer-related errors."""

    pass


class SameSourceDestinationBufferError(TransferError):
    def __str__(self):
        return "Source and destination cannot be the same buffer"


class BufferOverrunError(TransferError):
    def __init__(self, start_offset, end_offset, buffer_size, side):
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.buffer_size = buffer_size
        self.side = side

    def __str__(self):
        return f"Copy of {self.start_offset}..{self.end_offset} would overrun the bounds of the {self.side.value} buffer of size {self.buffer_size}"


# ... many other specific error classes would be defined here ...


def validate_linear_texture_data(
    layout, format, aspect, buffer_size, buffer_side, copy_size
) -> Tuple[int, int, bool]:
    """
    Validates linear texture data layout for a copy operation.

    Corresponds to the 'validating linear texture data' algorithm in the WebGPU spec.

    Returns:
        A tuple of (bytes_in_copy, image_stride_bytes, is_contiguous).
    """
    info = layout.get_buffer_texture_copy_info(format, aspect, copy_size)

    if info["copy_width"] % info["block_width_texels"] != 0:
        raise TransferError("Unaligned copy width")
    if info["copy_height"] % info["block_height_texels"] != 0:
        raise TransferError("Unaligned copy height")

    requires_multiple_rows = (
        info["depth_or_array_layers"] > 1 or info["height_blocks"] > 1
    )
    requires_multiple_images = info["depth_or_array_layers"] > 1

    if layout.bytes_per_row is None and requires_multiple_rows:
        raise TransferError("Unspecified bytes per row")
    if layout.rows_per_image is None and requires_multiple_images:
        raise TransferError("Unspecified rows per image")

    if (
        info["bytes_in_copy"] > buffer_size
        or info["offset"] > buffer_size - info["bytes_in_copy"]
    ):
        raise BufferOverrunError(
            info["offset"],
            info["offset"] + info["bytes_in_copy"],
            buffer_size,
            buffer_side,
        )

    is_contiguous = (
        info["row_stride_bytes"] == info["row_bytes_dense"]
        or not requires_multiple_rows
    ) and (
        info["image_stride_bytes"] == info["image_bytes_dense"]
        or not requires_multiple_images
    )

    return info["bytes_in_copy"], info["image_stride_bytes"], is_contiguous


def copy_buffer_to_buffer(
    state: Any, src: Any, src_offset: int, dst: Any, dst_offset: int, size: int
):
    """
    Validates and creates a command to copy data between buffers.
    """
    if src.is_equal(dst):
        raise SameSourceDestinationBufferError()

    src.same_device(state.device)
    dst.same_device(state.device)

    src.check_usage(wgt.BufferUsages.COPY_SRC)
    dst.check_usage(wgt.BufferUsages.COPY_DST)

    copy_size = (src.size - src_offset) if size is None else size

    if copy_size % wgt.COPY_BUFFER_ALIGNMENT != 0:
        raise TransferError(
            f"Copy size {copy_size} is not a multiple of {wgt.COPY_BUFFER_ALIGNMENT}"
        )
    if src_offset % wgt.COPY_BUFFER_ALIGNMENT != 0:
        raise TransferError(f"Source offset {src_offset} is not aligned")
    if dst_offset % wgt.COPY_BUFFER_ALIGNMENT != 0:
        raise TransferError(f"Destination offset {dst_offset} is not aligned")

    if src_offset + copy_size > src.size:
        raise BufferOverrunError(
            src_offset, src_offset + copy_size, src.size, CopySide.Source
        )
    if dst_offset + copy_size > dst.size:
        raise BufferOverrunError(
            dst_offset, dst_offset + copy_size, dst.size, CopySide.Destination
        )

    if copy_size == 0:
        return

    # Handle memory initialization tracking (simplified)
    # state.buffer_memory_init_actions.extend(...)

    state.tracker.buffers.set_single(src, wgt.BufferUsages.COPY_SRC)
    state.tracker.buffers.set_single(dst, wgt.BufferUsages.COPY_DST)

    # In a real implementation, this would generate a command, not call the raw encoder directly
    state.raw_encoder.copy_buffer_to_buffer(
        src.raw(), dst.raw(), [(src_offset, dst_offset, copy_size)]
    )


def copy_buffer_to_texture(state: Any, source: Any, destination: Any, copy_size: Any):
    """
    Validates and creates a command to copy data from a buffer to a texture.
    """
    dst_texture = destination.texture
    src_buffer = source.buffer

    dst_texture.same_device(state.device)
    src_buffer.same_device(state.device)

    # ... extensive validation logic from `transfer.rs` would go here ...
    # validate_texture_copy_range(...)
    # validate_texture_copy_dst_format(...)
    # validate_texture_buffer_copy(...)
    # validate_linear_texture_data(...)

    if (
        copy_size.width == 0
        or copy_size.height == 0
        or copy_size.depth_or_array_layers == 0
    ):
        return

    # Handle memory and resource state tracking
    # handle_dst_texture_init(...)
    # handle_buffer_init(...)

    # This is a placeholder for generating the HAL command
    state.raw_encoder.copy_buffer_to_texture(
        src_buffer.raw(),
        dst_texture.raw(),
        [  # regions
            # hal::BufferTextureCopy would be constructed here
        ],
    )


def copy_texture_to_buffer(state: Any, source: Any, destination: Any, copy_size: Any):
    """
    Validates and creates a command to copy data from a texture to a buffer.
    """
    # ... validation and state tracking logic similar to copy_buffer_to_texture ...
    state.raw_encoder.copy_texture_to_buffer(source, destination, copy_size)


def copy_texture_to_texture(state: Any, source: Any, destination: Any, copy_size: Any):
    """
    Validates and creates a command to copy data between textures.
    """
    # ... validation and state tracking logic for texture-to-texture copies ...
    state.raw_encoder.copy_texture_to_texture(source, destination, copy_size)


# The original dataclasses are less useful than the functions,
# but kept for structural reference.
@dataclass
class CopyBufferToBuffer:
    src: Any
    src_offset: int
    dst: Any
    dst_offset: int
    size: Optional[int]


@dataclass
class CopyBufferToTexture:
    src: Any
    dst: Any
    size: Any


@dataclass
class CopyTextureToBuffer:
    src: Any
    dst: Any
    size: Any


@dataclass
class CopyTextureToTexture:
    src: Any
    dst: Any
    size: Any
