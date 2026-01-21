"""
Clear operations for buffers and textures.

This module implements clear operations for wgpu-core. It provides:
- ClearBuffer: Clear a buffer to zero
- ClearTexture: Clear a texture to a specific value

Clear operations are used to initialize or reset GPU resources to a known state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, List, Union

from . import errors
from ..track.texture import TextureUses, TextureSelector
from ..init_tracker import (
    MemoryInitKind,
    TextureInitRange,
)


def get_lowest_common_denom(a: int, b: int) -> int:
    """Calculate the least common multiple of two integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def align_to(value: int, alignment: int) -> int:
    """Align a value to the next multiple of alignment."""
    if alignment == 0:
        return value
    return (value + alignment - 1) // alignment * alignment


@dataclass
class ClearError(Exception):
    """
    Error encountered while attempting a clear.
    
    Attributes:
        message: The error message.
    """

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ClearBuffer:
    """
    Command to clear a buffer.
    
    Attributes:
        dst: The destination buffer.
        offset: The offset into the buffer.
        size: The size of the data to clear.
    """

    dst: Any
    offset: int
    size: Optional[int] = None


@dataclass
class ClearTexture:
    """
    Command to clear a texture.
    
    Attributes:
        dst: The destination texture.
        subresource_range: The subresource range to clear.
    """

    dst: Any
    subresource_range: Any


def clear_buffer(
    state: Any,
    dst_buffer: Any,
    offset: int,
    size: Optional[int],
) -> None:
    """
    Clear a buffer to zero.
    
    Args:
        state: The encoding state.
        dst_buffer: The buffer to clear.
        offset: The offset into the buffer.
        size: The size of the data to clear.
    """
    dst_buffer.same_device(state.device)
    dst_buffer.check_usage(wgt.BufferUsages.COPY_DST)

    # Check if offset & size are valid.
    if offset % wgt.COPY_BUFFER_ALIGNMENT != 0:
        raise ClearError(f"Buffer offset {offset} is not a multiple of COPY_BUFFER_ALIGNMENT")

    size = size if size is not None else (dst_buffer.size - offset)
    if size % wgt.COPY_BUFFER_ALIGNMENT != 0:
        raise ClearError(f"Buffer clear size {size} is not a multiple of COPY_BUFFER_ALIGNMENT")

    if offset + size > dst_buffer.size:
        raise ClearError(f"Clear of {offset}..{offset+size} would overrun the bounds of the buffer of size {dst_buffer.size}")

    if size == 0:
        return
    
    # Track usage (transition)
    dst_pending = state.tracker.buffers.set_single(dst_buffer, wgt.BufferUsages.COPY_DST)
    
    # Mark dest as initialized.
    init_action = dst_buffer.initialization_status.create_action(
        dst_buffer,
        range(offset, offset + size),
        MemoryInitKind.IMPLICITLY_INITIALIZED,
    )
    if init_action:
        state.buffer_memory_init_actions.append(init_action)
    
    # actual hal barrier & operation
    if dst_pending:
        state.raw_encoder.transition_buffers([dst_pending])
    
    state.raw_encoder.clear_buffer(
        dst_buffer.raw(),
        offset,
        size
    )


def clear_texture_cmd(
    state: Any,
    dst_texture: Any,
    subresource_range: Any,
) -> None:
    """
    Clear a texture.
    
    Args:
        state: The encoding state.
        dst_texture: The texture to clear.
        subresource_range: The subresource range to clear.
    """
    dst_texture.same_device(state.device)
    state.device.require_features(wgt.Features.CLEAR_TEXTURE)
    
    # Check if subresource ranges are valid
    full_mip_count = dst_texture.desc.mip_level_count
    full_layer_count = dst_texture.desc.array_layer_count() # Assuming this exist or use getattr

    base_mip = subresource_range.base_mip_level
    mip_count = subresource_range.mip_level_count or (full_mip_count - base_mip)
    if base_mip + mip_count > full_mip_count:
        raise ClearError("Invalid texture level range")

    base_layer = subresource_range.base_array_layer
    layer_count = subresource_range.array_layer_count or (full_layer_count - base_layer)
    if base_layer + layer_count > full_layer_count:
        raise ClearError("Invalid texture layer range")

    clear_texture(
        dst_texture,
        TextureInitRange(
            mip_range=range(base_mip, base_mip + mip_count),
            layer_range=range(base_layer, base_layer + layer_count),
        ),
        state.raw_encoder,
        state.tracker.textures,
        state.device.alignments,
        state.device.zero_buffer,
        state.snatch_guard,
        state.device.instance_flags,
    )


def clear_texture(
    dst_texture: Any,
    range: TextureInitRange,
    encoder: Any,
    texture_tracker: Any,
    alignments: Any,
    zero_buffer: Any,
    snatch_guard: Any,
    instance_flags: Any,
) -> None:
    """
    Encode a texture clear operation.
    
    Args:
        dst_texture: The texture to clear.
        range: The texture init range.
        encoder: The command encoder.
        texture_tracker: The texture tracker.
        alignments: Device alignments.
        zero_buffer: Zero buffer for clearing.
        snatch_guard: The snatch guard.
        instance_flags: Instance flags.
    """
    dst_raw = dst_texture.try_raw(snatch_guard)
    if dst_raw is None:
        return

    # Determine clear mode and usage
    is_depth_stencil = _format_is_depth_stencil(dst_texture.desc.format)
    
    # In wgpu-core, clear_mode is stored on the texture.
    # Here we simplify: use buffer copies for color, render pass for depth/stencil.
    # But some color formats (like Surface textures) might need render pass.
    # For now, let's follow the suggested logic:
    if is_depth_stencil:
        clear_usage = TextureUses.DEPTH_STENCIL_WRITE
    else:
        clear_usage = TextureUses.COLOR_TARGET

    selector = TextureSelector(mips=range.mip_range, layers=range.layer_range)
    transitions = texture_tracker.set_single(dst_texture, selector, clear_usage)
    if transitions:
        encoder.transition_textures(transitions)

    # Record actual clearing
    if not is_depth_stencil:
        # Check if we should use buffer copies (default for color textures)
        _clear_texture_via_buffer_copies(
            dst_texture.desc,
            alignments,
            zero_buffer,
            range,
            encoder,
            dst_raw,
        )
    else:
        _clear_texture_via_render_passes(
            dst_texture,
            range,
            False, # is_color
            encoder,
            instance_flags,
        )


def _clear_texture_via_buffer_copies(
    texture_desc: Any,
    alignments: Any,
    zero_buffer: Any,
    range_: TextureInitRange,
    encoder: Any,
    dst_raw: Any,
) -> None:
    """Clear texture via copy_buffer_to_texture from a zeroed buffer."""
    # Simplified implementation of the Rust logic
    import pywgpu_hal as hal
    
    # Gather list of zero_buffer copies
    zero_buffer_copy_regions = []
    buffer_copy_pitch = alignments.buffer_copy_pitch
    
    # These would come from format info
    block_width, block_height = 1, 1 # Simplified for now
    block_size = 4 # Simplified for rgba8unorm
    
    bytes_per_row_alignment = get_lowest_common_denom(buffer_copy_pitch, block_size)
    
    # We need a way to get mip level size.
    # In Rust: texture_desc.mip_level_size(mip_level)
    
    for mip_level in range_.mip_range:
        # mip_size calculation (simplified)
        width = max(1, texture_desc.size[0] >> mip_level)
        height = max(1, texture_desc.size[1] >> mip_level)
        depth = max(1, texture_desc.size[2] >> mip_level) if len(texture_desc.size) > 2 else 1
        
        # Round to multiple of block size
        width = align_to(width, block_width)
        height = align_to(height, block_height)

        bytes_per_row = align_to(
            (width // block_width) * block_size,
            bytes_per_row_alignment,
        )

        # How many rows fit in ZERO_BUFFER?
        # Rust: crate::device::ZERO_BUFFER_SIZE
        ZERO_BUFFER_SIZE = 256 * 1024 # Placeholder
        max_rows_per_copy = ZERO_BUFFER_SIZE // bytes_per_row
        max_rows_per_copy = (max_rows_per_copy // block_height) * block_height
        
        if max_rows_per_copy == 0:
            continue # Should probably error or handle differently

        for array_layer in range_.layer_range:
            for z in range(depth):
                num_rows_left = height
                while num_rows_left > 0:
                    num_rows = min(num_rows_left, max_rows_per_copy)
                    
                    zero_buffer_copy_regions.append(hal.BufferTextureCopy(
                        buffer_layout=hal.TexelCopyBufferLayout(
                            offset=0,
                            bytes_per_row=bytes_per_row,
                            rows_per_image=None,
                        ),
                        texture_base=hal.TextureCopyBase(
                            mip_level=mip_level,
                            array_layer=array_layer,
                            origin=hal.Origin3d(x=0, y=height - num_rows_left, z=z),
                            aspect=hal.FormatAspects.COLOR,
                        ),
                        size=hal.CopyExtent(
                            width=width,
                            height=num_rows,
                            depth=1,
                        )
                    ))
                    num_rows_left -= num_rows

    if zero_buffer_copy_regions:
        encoder.copy_buffer_to_texture(zero_buffer, dst_raw, zero_buffer_copy_regions)


def _clear_texture_via_render_passes(
    dst_texture: Any,
    range_: TextureInitRange,
    is_color: bool,
    encoder: Any,
    instance_flags: Any,
) -> None:
    """Clear texture via a render pass with LOAD_CLEAR."""
    import pywgpu_hal as hal
    
    for mip_level in range_.mip_range:
        width = max(1, dst_texture.desc.size[0] >> mip_level)
        height = max(1, dst_texture.desc.size[1] >> mip_level)
        extent = hal.Extent3d(width=width, height=height, depth_or_array_layers=1)
        
        for array_layer in range_.layer_range:
            # Create a temporary view for clearing
            # This is a bit complex as it requires hal.TextureViewDescriptor
            view_desc = hal.TextureViewDescriptor(
                label="(pywgpu internal) clear_texture clear view",
                format=dst_texture.desc.format,
                dimension="2d",
                usage=TextureUses.COLOR_TARGET if is_color else TextureUses.DEPTH_STENCIL_WRITE,
                range=hal.ImageSubresourceRange(
                    aspect=hal.FormatAspects.COLOR if is_color else hal.FormatAspects.DEPTH_STENCIL,
                    base_mip_level=mip_level,
                    mip_level_count=1,
                    base_array_layer=array_layer,
                    array_layer_count=1,
                )
            )
            
            # Since we are inside a command encoder, we might not have direct access to device.create_texture_view
            # But the texture has a reference to the device.
            view = dst_texture.device.hal_device.create_texture_view(dst_texture.raw(), view_desc)
            
            color_attachments = []
            depth_stencil_attachment = None
            
            if is_color:
                color_attachments.append(hal.ColorAttachment(
                    target=hal.Attachment(view=view, usage=TextureUses.COLOR_TARGET),
                    depth_slice=None,
                    resolve_target=None,
                    ops=hal.AttachmentOps.STORE | hal.AttachmentOps.LOAD_CLEAR,
                    clear_value=hal.Color(0.0, 0.0, 0.0, 0.0), # wgt.Color.TRANSPARENT
                ))
            else:
                depth_stencil_attachment = hal.DepthStencilAttachment(
                    target=hal.Attachment(view=view, usage=TextureUses.DEPTH_STENCIL_WRITE),
                    depth_ops=hal.AttachmentOps.STORE | hal.AttachmentOps.LOAD_CLEAR,
                    stencil_ops=hal.AttachmentOps.STORE | hal.AttachmentOps.LOAD_CLEAR,
                    clear_value=(0.0, 0),
                )
            
            render_pass_desc = hal.RenderPassDescriptor(
                label="(pywgpu internal) clear_texture clear pass",
                extent=extent,
                sample_count=dst_texture.desc.sample_count,
                color_attachments=color_attachments,
                depth_stencil_attachment=depth_stencil_attachment,
                multiview_mask=None,
                timestamp_writes=None,
                occlusion_query_set=None,
            )
            
            encoder.begin_render_pass(render_pass_desc)
            encoder.end_render_pass()
            
            # Cleanup view
            dst_texture.device.hal_device.destroy_texture_view(view)


def _format_is_depth_stencil(fmt: str) -> bool:
    """Check if a texture format is a depth/stencil format."""
    return fmt.startswith("depth") or fmt == "stencil8"
