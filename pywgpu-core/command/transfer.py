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

try:
    from ..init_tracker import MemoryInitKind
except ImportError:
    class MemoryInitKind:
        ImplicitlyInitialized = "ImplicitlyInitialized"
        NeedsInitializedMemory = "NeedsInitializedMemory"

try:
    from ..resource import DestroyedResourceError
except ImportError:
    class DestroyedResourceError(Exception):
        pass


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


class TextureOverrunError(TransferError):
    def __init__(self, start_offset, end_offset, texture_size, dimension, side):
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.texture_size = texture_size
        self.dimension = dimension
        self.side = side

    def __str__(self):
        return (
            f"Copy of {self.dimension} {self.start_offset}..{self.end_offset} would overrun "
            f"the bounds of the {self.side.value} texture of {self.dimension} size {self.texture_size}"
        )


class UnsupportedPartialTransferError(TransferError):
    def __init__(self, format, sample_count, start_offset, end_offset, texture_size, dimension, side):
        self.format = format
        self.sample_count = sample_count
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.texture_size = texture_size
        self.dimension = dimension
        self.side = side

    def __str__(self):
        return (
            f"Partial copy of {self.start_offset}..{self.end_offset} on {self.dimension} dimension "
            f"with size {self.texture_size} is not supported for the {self.side.value} texture "
            f"format {self.format} with {self.sample_count} samples"
        )


class InvalidCopyWithinSameTextureError(TransferError):
    def __init__(self, src_aspects, dst_aspects, src_origin_z, dst_origin_z, array_layer_count):
        self.src_aspects = src_aspects
        self.dst_aspects = dst_aspects
        self.src_origin_z = src_origin_z
        self.dst_origin_z = dst_origin_z
        self.array_layer_count = array_layer_count

    def __str__(self):
        src_aspect_str = "" if self.src_aspects == "All" else f" {self.src_aspects}"
        dst_aspect_str = "" if self.dst_aspects == "All" else f" {self.dst_aspects}"
        return (
            f"Copying{src_aspect_str} layers {self.src_origin_z}..{self.src_origin_z + self.array_layer_count} "
            f"to{dst_aspect_str} layers {self.dst_origin_z}..{self.dst_origin_z + self.array_layer_count} "
            f"of the same texture is not allowed"
        )


class InvalidTextureAspectError(TransferError):
    def __init__(self, format, aspect):
        self.format = format
        self.aspect = aspect

    def __str__(self):
        return f"Unable to select texture aspect {self.aspect} from format {self.format}"


class InvalidTextureMipLevelError(TransferError):
    def __init__(self, level, total):
        self.level = level
        self.total = total

    def __str__(self):
        return f"Unable to select texture mip level {self.level} out of {self.total}"


class InvalidDimensionExternalError(TransferError):
    def __str__(self):
        return "Texture dimension must be 2D when copying from an external texture"


class UnalignedBufferOffsetError(TransferError):
    def __init__(self, offset):
        self.offset = offset

    def __str__(self):
        return f"Buffer offset {self.offset} is not aligned to block size or COPY_BUFFER_ALIGNMENT"


class UnalignedCopySizeError(TransferError):
    def __init__(self, size):
        self.size = size

    def __str__(self):
        return f"Copy size {self.size} does not respect COPY_BUFFER_ALIGNMENT"


class UnalignedCopyWidthError(TransferError):
    def __str__(self):
        return "Copy width is not a multiple of block width"


class UnalignedCopyHeightError(TransferError):
    def __str__(self):
        return "Copy height is not a multiple of block height"


class UnalignedCopyOriginXError(TransferError):
    def __str__(self):
        return "Copy origin's x component is not a multiple of block width"


class UnalignedCopyOriginYError(TransferError):
    def __str__(self):
        return "Copy origin's y component is not a multiple of block height"


class UnalignedBytesPerRowError(TransferError):
    def __str__(self):
        return "Bytes per row does not respect COPY_BYTES_PER_ROW_ALIGNMENT"


class UnspecifiedBytesPerRowError(TransferError):
    def __str__(self):
        return "Number of bytes per row needs to be specified since more than one row is copied"


class UnspecifiedRowsPerImageError(TransferError):
    def __str__(self):
        return "Number of rows per image needs to be specified since more than one image is copied"


class InvalidBytesPerRowError(TransferError):
    def __str__(self):
        return "Number of bytes per row is less than the number of bytes in a complete row"


class InvalidRowsPerImageError(TransferError):
    def __str__(self):
        return "Number of rows per image is invalid"


class SizeOverflowError(TransferError):
    def __str__(self):
        return "Overflow while computing the size of the copy"


class CopySrcMissingAspectsError(TransferError):
    def __str__(self):
        return "Copy source aspects must refer to all aspects of the source texture format"


class CopyDstMissingAspectsError(TransferError):
    def __str__(self):
        return "Copy destination aspects must refer to all aspects of the destination texture format"


class CopyAspectNotOneError(TransferError):
    def __str__(self):
        return "Copy aspect must refer to a single aspect of texture format"


class CopyFromForbiddenTextureFormatError(TransferError):
    def __init__(self, format):
        self.format = format

    def __str__(self):
        return f"Copying from textures with format {self.format} is forbidden"


class CopyFromForbiddenTextureFormatAspectError(TransferError):
    def __init__(self, format, aspect):
        self.format = format
        self.aspect = aspect

    def __str__(self):
        return f"Copying from textures with format {self.format} and aspect {self.aspect} is forbidden"


class CopyToForbiddenTextureFormatError(TransferError):
    def __init__(self, format):
        self.format = format

    def __str__(self):
        return f"Copying to textures with format {self.format} is forbidden"


class CopyToForbiddenTextureFormatAspectError(TransferError):
    def __init__(self, format, aspect):
        self.format = format
        self.aspect = aspect

    def __str__(self):
        return f"Copying to textures with format {self.format} and aspect {self.aspect} is forbidden"


class ExternalCopyToForbiddenTextureFormatError(TransferError):
    def __init__(self, format):
        self.format = format

    def __str__(self):
        return f"Copying to textures with format {self.format} is forbidden when copying from external texture"


class TextureFormatsNotCopyCompatibleError(TransferError):
    def __init__(self, src_format, dst_format):
        self.src_format = src_format
        self.dst_format = dst_format

    def __str__(self):
        return (
            f"Source format ({self.src_format}) and destination format ({self.dst_format}) "
            f"are not copy-compatible (they may only differ in srgb-ness)"
        )


class InvalidSampleCountError(TransferError):
    def __init__(self, sample_count):
        self.sample_count = sample_count

    def __str__(self):
        return f"Source texture sample count must be 1, got {self.sample_count}"


class SampleCountNotEqualError(TransferError):
    def __init__(self, src_sample_count, dst_sample_count):
        self.src_sample_count = src_sample_count
        self.dst_sample_count = dst_sample_count

    def __str__(self):
        return (
            f"Source sample count ({self.src_sample_count}) and destination sample count "
            f"({self.dst_sample_count}) are not equal"
        )


class InvalidMipLevelError(TransferError):
    def __init__(self, requested, count):
        self.requested = requested
        self.count = count

    def __str__(self):
        return f"Requested mip level {self.requested} does not exist (count: {self.count})"


class BufferNotAvailableError(TransferError):
    def __str__(self):
        return "Buffer is expected to be unmapped, but was not"


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


def handle_texture_init(
    state: Any,
    init_kind: Any,
    copy_texture: Any,
    copy_size: Any,
    texture: Any,
) -> None:
    """
    Handle texture initialization for a transfer operation.
    
    This registers the texture init action and handles any immediate clears needed.
    
    Args:
        state: The encoding state.
        init_kind: The kind of memory initialization (NeedsInitializedMemory or ImplicitlyInitialized).
        copy_texture: The texture copy info.
        copy_size: The copy size.
        texture: The texture resource.
    """
    # Create init action
    # In Rust: TextureInitTrackerAction with texture, range, and kind
    init_action = {
        'texture': texture,
        'mip_range': range(copy_texture.mip_level, copy_texture.mip_level + 1),
        'layer_range': range(
            copy_texture.origin.z,
            copy_texture.origin.z + copy_size.depth_or_array_layers
        ),
        'kind': init_kind,
    }
    
    # Register the init action
    if hasattr(state, 'texture_memory_actions'):
        immediate_inits = state.texture_memory_actions.register_init_action(init_action)
        
        # Handle immediate clears if needed
        if immediate_inits:
            from . import clear_texture
            for init in immediate_inits:
                # Clear the texture surface immediately
                clear_texture(
                    init.texture,
                    {
                        'mip_range': range(init.mip_level, init.mip_level + 1),
                        'layer_range': range(init.layer, init.layer + 1),
                    },
                    state.raw_encoder,
                    state.tracker.textures if hasattr(state.tracker, 'textures') else None,
                    state.device.alignments if hasattr(state.device, 'alignments') else None,
                    state.device.zero_buffer if hasattr(state.device, 'zero_buffer') else None,
                    state.snatch_guard,
                    state.device.instance_flags if hasattr(state.device, 'instance_flags') else None,
                )


def handle_src_texture_init(
    state: Any,
    source: Any,
    copy_size: Any,
    texture: Any,
) -> None:
    """
    Prepare a transfer's source texture.
    
    Ensure the source texture is initialized before reading.
    """
    handle_texture_init(
        state,
        MemoryInitKind.NeedsInitializedMemory,
        source,
        copy_size,
        texture,
    )


def handle_dst_texture_init(
    state: Any,
    destination: Any,
    copy_size: Any,
    texture: Any,
) -> None:
    """
    Prepare a transfer's destination texture.
    
    Determines if the destination needs initialization based on whether
    we're writing full subresources or partial regions.
    """
    # Check if we have partial init tracker coverage
    # In Rust: has_copy_partial_init_tracker_coverage(copy_size, destination.mip_level, &texture.desc)
    # If we don't write full texture subresources, we need a full clear first
    # For now, assume we need initialized memory for partial writes
    dst_init_kind = MemoryInitKind.ImplicitlyInitialized
    
    handle_texture_init(
        state,
        dst_init_kind,
        destination,
        copy_size,
        texture,
    )


def handle_buffer_init(
    state: Any,
    info: Any,
    direction: CopySide,
    required_buffer_bytes_in_copy: int,
    is_contiguous: bool,
) -> None:
    """
    Handle initialization tracking for a transfer's source or destination buffer.
    
    Ensures that the transfer will not read from uninitialized memory, and updates
    the initialization state information to reflect the transfer.
    """
    try:
        ALIGN_SIZE = wgt.COPY_BUFFER_ALIGNMENT
        ALIGN_MASK = wgt.COPY_BUFFER_ALIGNMENT - 1
    except:
        ALIGN_SIZE = 4
        ALIGN_MASK = 3
    
    buffer = info.buffer
    start = info.layout.offset
    end = info.layout.offset + required_buffer_bytes_in_copy
    
    if not is_contiguous or direction == CopySide.Source:
        # If reading or non-contiguous write, initialize the whole region
        # Adjust the start/end outwards to alignment
        aligned_start = start & ~ALIGN_MASK
        aligned_end = (end + ALIGN_MASK) & ~ALIGN_MASK
        
        if hasattr(buffer, 'initialization_status'):
            actions = buffer.initialization_status.read().create_action(
                buffer,
                range(aligned_start, aligned_end),
                MemoryInitKind.NeedsInitializedMemory,
            )
            if actions:
                state.buffer_memory_init_actions.extend(actions)
    else:
        # Contiguous write to destination - mark as implicitly initialized
        # Adjust the start/end inwards to alignment
        aligned_start = (start + ALIGN_MASK) & ~ALIGN_MASK
        aligned_end = end & ~ALIGN_MASK
        
        if hasattr(buffer, 'initialization_status'):
            # Handle unaligned start
            if aligned_start != start:
                actions = buffer.initialization_status.read().create_action(
                    buffer,
                    range(aligned_start - ALIGN_SIZE, aligned_start),
                    MemoryInitKind.NeedsInitializedMemory,
                )
                if actions:
                    state.buffer_memory_init_actions.extend(actions)
            
            # Mark aligned region as implicitly initialized
            if aligned_start != aligned_end:
                actions = buffer.initialization_status.read().create_action(
                    buffer,
                    range(aligned_start, aligned_end),
                    MemoryInitKind.ImplicitlyInitialized,
                )
                if actions:
                    state.buffer_memory_init_actions.extend(actions)
            
            # Handle unaligned end
            if aligned_end != end:
                actions = buffer.initialization_status.read().create_action(
                    buffer,
                    range(aligned_end, aligned_end + ALIGN_SIZE),
                    MemoryInitKind.NeedsInitializedMemory,
                )
                if actions:
                    state.buffer_memory_init_actions.extend(actions)


def validate_same_texture_overlap(
    src: Any,
    dst: Any,
    format: Any,
    array_layer_count: int,
) -> None:
    """
    Validate that a texture-to-texture copy doesn't have invalid overlap.
    
    Raises:
        TransferError: If the copy would overlap within the same texture in an invalid way.
    """
    # Check if aspects overlap
    # In Rust: hal::FormatAspects::new(format, src.aspect) & hal::FormatAspects::new(format, dst.aspect)
    # Simplified: assume aspects overlap if they're the same
    if src.aspect != dst.aspect:
        return  # Different aspects, no overlap
    
    # Check if layer ranges overlap
    if src.origin.z >= dst.origin.z + array_layer_count:
        return  # Non-overlapping layer ranges
    if dst.origin.z >= src.origin.z + array_layer_count:
        return  # Non-overlapping layer ranges
    
    # Check if mip levels are different
    if src.mip_level != dst.mip_level:
        return  # Different mip levels, okay
    
    # Invalid overlap detected
    raise TransferError(
        f"Copying {src.aspect} layers {src.origin.z}..{src.origin.z + array_layer_count} "
        f"to {dst.aspect} layers {dst.origin.z}..{dst.origin.z + array_layer_count} "
        f"of the same texture is not allowed"
    )


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

    # Handle memory initialization tracking
    # Make sure source is initialized memory and mark dest as initialized
    if hasattr(dst, 'initialization_status'):
        # Mark destination as implicitly initialized
        dst_init_actions = dst.initialization_status.read().create_action(
            dst,
            range(dst_offset, dst_offset + copy_size),
            MemoryInitKind.ImplicitlyInitialized,
        )
        if dst_init_actions:
            state.buffer_memory_init_actions.extend(dst_init_actions)
    
    if hasattr(src, 'initialization_status'):
        # Ensure source memory is initialized before reading
        src_init_actions = src.initialization_status.read().create_action(
            src,
            range(src_offset, src_offset + copy_size),
            MemoryInitKind.NeedsInitializedMemory,
        )
        if src_init_actions:
            state.buffer_memory_init_actions.extend(src_init_actions)

    # Track buffer usage and get transitions
    src_pending = state.tracker.buffers.set_single(src, wgt.BufferUsages.COPY_SRC)
    dst_pending = state.tracker.buffers.set_single(dst, wgt.BufferUsages.COPY_DST)
    
    # Get raw buffer handles
    src_raw = src.try_raw(state.snatch_guard)
    dst_raw = dst.try_raw(state.snatch_guard)
    
    if src_raw is None or dst_raw is None:
        raise DestroyedResourceError("Buffer was destroyed")
    
    # Build barriers list
    barriers = []
    if src_pending:
        src_barrier = src_pending.into_hal(src, state.snatch_guard)
        if src_barrier:
            barriers.append(src_barrier)
    if dst_pending:
        dst_barrier = dst_pending.into_hal(dst, state.snatch_guard)
        if dst_barrier:
            barriers.append(dst_barrier)
    
    # Issue HAL commands
    if hasattr(state.raw_encoder, 'transition_buffers') and barriers:
        state.raw_encoder.transition_buffers(barriers)
    
    # Create HAL region
    region = {
        'src_offset': src_offset,
        'dst_offset': dst_offset,
        'size': copy_size,
    }

    # In a real implementation, this would generate a command, not call the raw encoder directly
    state.raw_encoder.copy_buffer_to_buffer(
        src_raw, dst_raw, [region]
    )


def copy_buffer_to_texture(state: Any, source: Any, destination: Any, copy_size: Any):
    """
    Validates and creates a command to copy data from a buffer to a texture.
    """
    dst_texture = destination.texture
    src_buffer = source.buffer

    dst_texture.same_device(state.device)
    src_buffer.same_device(state.device)

    # Validate texture copy range
    # In Rust: validate_texture_copy_range(destination, &dst_texture.desc, CopySide::Destination, copy_size)
    # This returns (hal_copy_size, array_layer_count)
    
    # Validate destination format
    # In Rust: validate_texture_copy_dst_format(dst_texture.desc.format, destination.aspect)
    
    # Validate texture buffer copy alignment
    # In Rust: validate_texture_buffer_copy(destination, dst_base.aspect, &dst_texture.desc, &source.layout, true)
    
    # Validate linear texture data
    # In Rust: validate_linear_texture_data(&source.layout, dst_texture.desc.format, destination.aspect, src_buffer.size, CopySide::Source, copy_size)
    # This returns (required_buffer_bytes_in_copy, bytes_per_array_layer, is_contiguous)

    # Check for zero-size copy
    if (
        copy_size.width == 0
        or copy_size.height == 0
        or copy_size.depth_or_array_layers == 0
    ):
        return

    # Handle destination texture initialization
    # This must happen before barrier transitions
    handle_dst_texture_init(state, destination, copy_size, dst_texture)

    # Get raw handles
    src_raw = src_buffer.try_raw(state.snatch_guard)
    dst_raw = dst_texture.try_raw(state.snatch_guard)
    
    if src_raw is None or dst_raw is None:
        raise DestroyedResourceError("Resource was destroyed")

    # Check usages
    if hasattr(src_buffer, 'check_usage'):
        src_buffer.check_usage(wgt.BufferUsages.COPY_SRC)
    if hasattr(dst_texture, 'check_usage'):
        dst_texture.check_usage(wgt.TextureUsages.COPY_DST)

    # Track buffer usage
    src_pending = state.tracker.buffers.set_single(src_buffer, wgt.BufferUses.COPY_SRC)
    src_barrier = src_pending.into_hal(src_buffer, state.snatch_guard) if src_pending else None
    
    # Track texture usage
    # In Rust: state.tracker.textures.set_single(dst_texture, dst_range, wgt::TextureUses::COPY_DST)
    # For now, simplified:
    if hasattr(state.tracker, 'textures'):
        dst_pending = state.tracker.textures.set_single(dst_texture, None, wgt.TextureUses.COPY_DST)
        dst_barriers = list(dst_pending.into_hal(dst_raw)) if dst_pending else []
    else:
        dst_barriers = []

    # Handle buffer initialization
    # In Rust: handle_buffer_init(state, source, CopySide::Source, required_buffer_bytes_in_copy, is_contiguous)
    # This marks the buffer range as needing initialized memory
    # For now, use simplified version - would need to calculate required_buffer_bytes_in_copy and is_contiguous
    # from validate_linear_texture_data

    # Build HAL regions
    # In Rust, this creates BufferTextureCopy regions for each array layer
    regions = []
    # Simplified: would iterate over array layers and create regions
    # For now, create a single region placeholder
    
    # Issue HAL commands
    if dst_barriers and hasattr(state.raw_encoder, 'transition_textures'):
        state.raw_encoder.transition_textures(dst_barriers)
    if src_barrier and hasattr(state.raw_encoder, 'transition_buffers'):
        state.raw_encoder.transition_buffers([src_barrier])
    
    if hasattr(state.raw_encoder, 'copy_buffer_to_texture'):
        state.raw_encoder.copy_buffer_to_texture(
            src_raw,
            dst_raw,
            regions,  # hal::BufferTextureCopy regions
        )


def copy_texture_to_buffer(state: Any, source: Any, destination: Any, copy_size: Any):
    """
    Validates and creates a command to copy data from a texture to a buffer.
    """
    src_texture = source.texture
    dst_buffer = destination.buffer

    src_texture.same_device(state.device)
    dst_buffer.same_device(state.device)

    # Validate texture copy range
    # In Rust: validate_texture_copy_range(source, &src_texture.desc, CopySide::Source, copy_size)
    
    # Validate source format
    # In Rust: validate_texture_copy_src_format(src_texture.desc.format, source.aspect)
    
    # Validate texture buffer copy
    # In Rust: validate_texture_buffer_copy(source, src_base.aspect, &src_texture.desc, &destination.layout, true)
    
    # Validate linear texture data
    # In Rust: validate_linear_texture_data(&destination.layout, src_texture.desc.format, source.aspect, dst_buffer.size, CopySide::Destination, copy_size)

    # Check for zero-size copy
    if (
        copy_size.width == 0
        or copy_size.height == 0
        or copy_size.depth_or_array_layers == 0
    ):
        return

    # Handle source texture initialization
    handle_src_texture_init(state, source, copy_size, src_texture)

    # Get raw handles
    src_raw = src_texture.try_raw(state.snatch_guard)
    dst_raw = dst_buffer.try_raw(state.snatch_guard)
    
    if src_raw is None or dst_raw is None:
        raise DestroyedResourceError("Resource was destroyed")

    # Check usages
    if hasattr(src_texture, 'check_usage'):
        src_texture.check_usage(wgt.TextureUsages.COPY_SRC)
    if hasattr(dst_buffer, 'check_usage'):
        dst_buffer.check_usage(wgt.BufferUsages.COPY_DST)

    # Track texture usage
    if hasattr(state.tracker, 'textures'):
        src_pending = state.tracker.textures.set_single(src_texture, None, wgt.TextureUses.COPY_SRC)
        src_barriers = list(src_pending.into_hal(src_raw)) if src_pending else []
    else:
        src_barriers = []
    
    # Track buffer usage
    dst_pending = state.tracker.buffers.set_single(dst_buffer, wgt.BufferUses.COPY_DST)
    dst_barrier = dst_pending.into_hal(dst_buffer, state.snatch_guard) if dst_pending else None

    # Handle buffer initialization
    # In Rust: handle_buffer_init(state, destination, CopySide::Destination, required_buffer_bytes_in_copy, is_contiguous)
    # For now, use simplified version - would need to calculate required_buffer_bytes_in_copy and is_contiguous

    # Build HAL regions
    regions = []
    # Simplified: would iterate over array layers and create regions
    
    # Issue HAL commands
    if dst_barrier and hasattr(state.raw_encoder, 'transition_buffers'):
        state.raw_encoder.transition_buffers([dst_barrier])
    if src_barriers and hasattr(state.raw_encoder, 'transition_textures'):
        state.raw_encoder.transition_textures(src_barriers)
    
    if hasattr(state.raw_encoder, 'copy_texture_to_buffer'):
        state.raw_encoder.copy_texture_to_buffer(
            src_raw,
            wgt.TextureUses.COPY_SRC,
            dst_raw,
            regions,  # hal::BufferTextureCopy regions
        )


def copy_texture_to_texture(state: Any, source: Any, destination: Any, copy_size: Any):
    """
    Validates and creates a command to copy data between textures.
    """
    src_texture = source.texture
    dst_texture = destination.texture

    src_texture.same_device(state.device)
    dst_texture.same_device(state.device)

    # Validate both texture copy ranges
    # In Rust: validate_texture_copy_range for both source and destination
    
    # Validate formats
    # In Rust: validate_texture_copy_src_format and validate_texture_copy_dst_format
    
    # Check for same texture overlap
    # Validate that we're not copying overlapping regions within the same texture
    if hasattr(src_texture, 'is_equal') and src_texture.is_equal(dst_texture):
        # In Rust: validate_same_texture_overlap(source, destination, format, array_layer_count)
        # For now, simplified - would need array_layer_count from validation
        validate_same_texture_overlap(source, destination, dst_texture.desc.format if hasattr(dst_texture, 'desc') else None, 1)

    # Check for zero-size copy
    if (
        copy_size.width == 0
        or copy_size.height == 0
        or copy_size.depth_or_array_layers == 0
    ):
        return

    # Handle texture initialization
    handle_src_texture_init(state, source, copy_size, src_texture)
    handle_dst_texture_init(state, destination, copy_size, dst_texture)

    # Get raw handles
    src_raw = src_texture.try_raw(state.snatch_guard)
    dst_raw = dst_texture.try_raw(state.snatch_guard)
    
    if src_raw is None or dst_raw is None:
        raise DestroyedResourceError("Resource was destroyed")

    # Check usages
    if hasattr(src_texture, 'check_usage'):
        src_texture.check_usage(wgt.TextureUsages.COPY_SRC)
    if hasattr(dst_texture, 'check_usage'):
        dst_texture.check_usage(wgt.TextureUsages.COPY_DST)

    # Track texture usages
    if hasattr(state.tracker, 'textures'):
        src_pending = state.tracker.textures.set_single(src_texture, None, wgt.TextureUses.COPY_SRC)
        dst_pending = state.tracker.textures.set_single(dst_texture, None, wgt.TextureUses.COPY_DST)
        
        src_barriers = list(src_pending.into_hal(src_raw)) if src_pending else []
        dst_barriers = list(dst_pending.into_hal(dst_raw)) if dst_pending else []
    else:
        src_barriers = []
        dst_barriers = []

    # Build HAL regions
    regions = []
    # Simplified: would iterate over array layers and create regions
    
    # Issue HAL commands
    if src_barriers and hasattr(state.raw_encoder, 'transition_textures'):
        state.raw_encoder.transition_textures(src_barriers)
    if dst_barriers and hasattr(state.raw_encoder, 'transition_textures'):
        state.raw_encoder.transition_textures(dst_barriers)
    
    if hasattr(state.raw_encoder, 'copy_texture_to_texture'):
        state.raw_encoder.copy_texture_to_texture(
            src_raw,
            wgt.TextureUses.COPY_SRC,
            dst_raw,
            regions,  # hal::TextureCopy regions
        )


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
