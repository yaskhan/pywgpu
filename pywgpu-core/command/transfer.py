"""
Transfer operations for command encoding.

This module implements transfer operations for wgpu-core. It provides:
- CopyBufferToBuffer: Copy data between buffers
- CopyBufferToTexture: Copy data from buffer to texture
- CopyTextureToBuffer: Copy data from texture to buffer
- CopyTextureToTexture: Copy data between textures

Transfer operations are used to copy data between GPU resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from . import errors


@dataclass
class CopyBufferToBuffer:
    """
    Command to copy data between buffers.
    
    Attributes:
        src: Source buffer.
        src_offset: Offset in source buffer.
        dst: Destination buffer.
        dst_offset: Offset in destination buffer.
        size: Size of data to copy.
    """

    src: Any
    src_offset: int
    dst: Any
    dst_offset: int
    size: Optional[int]


@dataclass
class CopyBufferToTexture:
    """
    Command to copy data from buffer to texture.
    
    Attributes:
        src: Source buffer info.
        dst: Destination texture info.
        size: Size of data to copy.
    """

    src: Any
    dst: Any
    size: Any


@dataclass
class CopyTextureToBuffer:
    """
    Command to copy data from texture to buffer.
    
    Attributes:
        src: Source texture info.
        dst: Destination buffer info.
        size: Size of data to copy.
    """

    src: Any
    dst: Any
    size: Any


@dataclass
class CopyTextureToTexture:
    """
    Command to copy data between textures.
    
    Attributes:
        src: Source texture info.
        dst: Destination texture info.
        size: Size of data to copy.
    """

    src: Any
    dst: Any
    size: Any


@dataclass
class TransferError(Exception):
    """
    Error related to transfer operations.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


def copy_buffer_to_buffer(
    state: Any,
    src: Any,
    src_offset: int,
    dst: Any,
    dst_offset: int,
    size: int,
) -> None:
    """
    Copy data between buffers.
    
    Args:
        state: The encoding state.
        src: The source buffer.
        src_offset: The offset in the source buffer.
        dst: The destination buffer.
        dst_offset: The offset in the destination buffer.
        size: The size of the data to copy.
    
    Raises:
        TransferError: If the copy is invalid.
    """
    if size == 0:
        return

    src.same_device(state.device)
    dst.same_device(state.device)

    src.check_usage("COPY_SRC")
    dst.check_usage("COPY_DST")

    if src_offset + size > src.size:
        raise TransferError(f"Source buffer overrun: {src_offset} + {size} > {src.size}")
    if dst_offset + size > dst.size:
        raise TransferError(f"Destination buffer overrun: {dst_offset} + {size} > {dst.size}")

    state.tracker.buffers.set_single(src, "COPY_SRC")
    state.tracker.buffers.set_single(dst, "COPY_DST")

    state.raw_encoder.copy_buffer_to_buffer(
        src.raw(),
        src_offset,
        dst.raw(),
        dst_offset,
        size,
    )


def copy_buffer_to_texture(
    state: Any,
    src: Any,
    dst: Any,
    size: Any,
) -> None:
    """
    Copy data from buffer to texture.
    
    Args:
        state: The encoding state.
        src: The source texel copy buffer info.
        dst: The destination texel copy texture info.
        size: The size of the data to copy.
    """
    # Simplified validation
    # src_buffer.check_usage("COPY_SRC")
    # dst_texture.check_usage("COPY_DST")
    
    state.raw_encoder.copy_buffer_to_texture(src, dst, size)


def copy_texture_to_buffer(
    state: Any,
    src: Any,
    dst: Any,
    size: Any,
) -> None:
    """
    Copy data from texture to buffer.
    
    Args:
        state: The encoding state.
        src: The source texel copy texture info.
        dst: The destination texel copy buffer info.
        size: The size of the data to copy.
    """
    state.raw_encoder.copy_texture_to_buffer(src, dst, size)


def copy_texture_to_texture(
    state: Any,
    src: Any,
    dst: Any,
    size: Any,
) -> None:
    """
    Copy data between textures.
    
    Args:
        state: The encoding state.
        src: The source texel copy texture info.
        dst: The destination texel copy texture info.
        size: The size of the data to copy.
    """
    state.raw_encoder.copy_texture_to_texture(src, dst, size)
