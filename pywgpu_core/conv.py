"""
Type conversion between wgpu and HAL.

This module provides conversion utilities between wgpu types and HAL
(Hardware Abstraction Layer) types. These conversions are used to
translate WebGPU API calls into HAL-specific operations.

The module handles:
- Buffer usage conversions
- Texture usage conversions
- Format conversions
- Dimension checks
"""

from __future__ import annotations

from typing import Any, Optional

# Import HAL and wgpu_types
try:
    import sys
    import os

    _hal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pywgpu-hal")
    if _hal_path not in sys.path:
        sys.path.insert(0, _hal_path)
    import lib as hal
except ImportError:
    hal = None  # type: ignore

try:
    import pywgpu_types as wgt
except ImportError:
    wgt = None  # type: ignore


def is_valid_external_image_copy_dst_texture_format(format: Any) -> bool:
    """
    Check if a texture format is valid for external image copy destination.

    Args:
        format: The texture format to check.

    Returns:
        True if the format is valid, False otherwise.
    """
    if wgt is None:
        return False

    # Valid formats for external image copy destination
    # Based on WebGPU spec and wgpu-core implementation
    valid_formats = {
        "r8unorm",
        "r16float",
        "r32float",
        "rg8unorm",
        "rg16float",
        "rg32float",
        "rgba8unorm",
        "rgba8unorm-srgb",
        "bgra8unorm",
        "bgra8unorm-srgb",
        "rgb10a2unorm",
        "rgba16float",
        "rgba32float",
    }

    # Handle both string and enum formats
    if isinstance(format, str):
        return format.lower() in valid_formats
    elif hasattr(format, "name"):
        return format.name.lower() in valid_formats

    return False


def map_buffer_usage(usage: Any) -> Any:
    """
    Map wgpu buffer usages to HAL buffer uses.

    Args:
        usage: The wgpu buffer usages (wgt.BufferUsages).

    Returns:
        The HAL buffer uses (wgt.BufferUses).
    """
    if wgt is None:
        return 0

    # Start with empty uses
    u = 0

    # Helper to check if usage contains a flag
    def has_flag(flag_name: str) -> bool:
        if hasattr(wgt, "BufferUsages") and hasattr(usage, "__contains__"):
            flag = getattr(wgt.BufferUsages, flag_name, None)
            if flag is not None:
                return flag in usage or (
                    hasattr(usage, "value") and (usage.value & flag) != 0
                )
        return False

    # Helper to set a use flag
    def set_use(use_name: str, condition: bool):
        nonlocal u
        if condition and hasattr(wgt, "BufferUses"):
            use_flag = getattr(wgt.BufferUses, use_name, 0)
            u |= use_flag

    # Map each usage flag
    set_use("MAP_READ", has_flag("MAP_READ"))
    set_use("MAP_WRITE", has_flag("MAP_WRITE"))
    set_use("COPY_SRC", has_flag("COPY_SRC"))
    set_use("COPY_DST", has_flag("COPY_DST"))
    set_use("INDEX", has_flag("INDEX"))
    set_use("VERTEX", has_flag("VERTEX"))
    set_use("UNIFORM", has_flag("UNIFORM"))

    # STORAGE maps to both READ_ONLY and READ_WRITE
    if has_flag("STORAGE"):
        set_use("STORAGE_READ_ONLY", True)
        set_use("STORAGE_READ_WRITE", True)

    set_use("INDIRECT", has_flag("INDIRECT"))
    set_use("QUERY_RESOLVE", has_flag("QUERY_RESOLVE"))
    set_use("BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT", has_flag("BLAS_INPUT"))
    set_use("TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT", has_flag("TLAS_INPUT"))

    return u


def map_texture_usage(
    usage: Any,
    aspect: Any,
    flags: Any,
) -> Any:
    """
    Map wgpu texture usages to HAL texture uses.

    Args:
        usage: The wgpu texture usages (wgt.TextureUsages).
        aspect: The format aspects (hal.FormatAspects).
        flags: The texture format feature flags (wgt.TextureFormatFeatureFlags).

    Returns:
        The HAL texture uses (wgt.TextureUses).
    """
    if wgt is None:
        return 0

    u = 0

    # Helper functions
    def has_usage(flag_name: str) -> bool:
        if hasattr(wgt, "TextureUsages") and hasattr(usage, "__contains__"):
            flag = getattr(wgt.TextureUsages, flag_name, None)
            if flag is not None:
                return flag in usage or (
                    hasattr(usage, "value") and (usage.value & flag) != 0
                )
        return False

    def has_feature_flag(flag_name: str) -> bool:
        if hasattr(wgt, "TextureFormatFeatureFlags") and hasattr(flags, "__contains__"):
            flag = getattr(wgt.TextureFormatFeatureFlags, flag_name, None)
            if flag is not None:
                return flag in flags or (
                    hasattr(flags, "value") and (flags.value & flag) != 0
                )
        return False

    def set_use(use_name: str, condition: bool):
        nonlocal u
        if condition and hasattr(wgt, "TextureUses"):
            use_flag = getattr(wgt.TextureUses, use_name, 0)
            u |= use_flag

    # Basic copy operations
    set_use("COPY_SRC", has_usage("COPY_SRC"))
    set_use("COPY_DST", has_usage("COPY_DST"))
    set_use("RESOURCE", has_usage("TEXTURE_BINDING"))

    # Storage binding with read/write variants
    if has_usage("STORAGE_BINDING"):
        set_use("STORAGE_READ_ONLY", has_feature_flag("STORAGE_READ_ONLY"))
        set_use("STORAGE_WRITE_ONLY", has_feature_flag("STORAGE_WRITE_ONLY"))
        set_use("STORAGE_READ_WRITE", has_feature_flag("STORAGE_READ_WRITE"))

    # Determine if this is a color or depth/stencil format
    is_color = False
    if hal and hasattr(aspect, "__and__"):
        color_aspects = 0
        if hasattr(hal.FormatAspects, "COLOR"):
            color_aspects |= hal.FormatAspects.COLOR
        if hasattr(hal.FormatAspects, "PLANE_0"):
            color_aspects |= hal.FormatAspects.PLANE_0
        if hasattr(hal.FormatAspects, "PLANE_1"):
            color_aspects |= hal.FormatAspects.PLANE_1
        if hasattr(hal.FormatAspects, "PLANE_2"):
            color_aspects |= hal.FormatAspects.PLANE_2
        is_color = (aspect & color_aspects) != 0

    # Render attachment usage
    if has_usage("RENDER_ATTACHMENT"):
        if is_color:
            set_use("COLOR_TARGET", True)
        else:
            set_use("DEPTH_STENCIL_READ", True)
            set_use("DEPTH_STENCIL_WRITE", True)

    # Additional uses
    set_use("STORAGE_ATOMIC", has_usage("STORAGE_ATOMIC"))
    set_use("TRANSIENT", has_usage("TRANSIENT"))

    return u


def map_texture_usage_for_texture(
    desc: Any,
    format_features: Any,
) -> Any:
    """
    Map texture usage for a specific texture.

    This function enforces having COPY_DST/DEPTH_STENCIL_WRITE/COLOR_TARGET
    otherwise we wouldn't be able to initialize the texture.

    Args:
        desc: The texture descriptor (TextureDescriptor).
        format_features: The format features (TextureFormatFeatures).

    Returns:
        The HAL texture uses (wgt.TextureUses).
    """
    if wgt is None or not hasattr(desc, "usage") or not hasattr(desc, "format"):
        return 0

    # Get format aspects
    format_aspect = hal.FormatAspects.COLOR if hal else 0
    if hasattr(desc.format, "into"):
        format_aspect = desc.format.into()

    # Get format feature flags
    flags = 0
    if hasattr(format_features, "flags"):
        flags = format_features.flags

    # Map the basic usage
    uses = map_texture_usage(desc.usage, format_aspect, flags)

    # Enforce initialization capability
    if (
        hasattr(desc.format, "is_depth_stencil_format")
        and desc.format.is_depth_stencil_format()
    ):
        # Depth/stencil formats need DEPTH_STENCIL_WRITE
        if wgt and hasattr(wgt, "TextureUses"):
            uses |= getattr(wgt.TextureUses, "DEPTH_STENCIL_WRITE", 0)
    elif hasattr(desc.usage, "__contains__") and hasattr(wgt, "TextureUsages"):
        copy_dst = getattr(wgt.TextureUsages, "COPY_DST", None)
        if copy_dst and copy_dst in desc.usage:
            # Already has COPY_DST
            pass
        else:
            # Try to use COLOR_TARGET if available, otherwise COPY_DST
            can_use_color_target = False
            if hasattr(format_features, "allowed_usages"):
                render_attachment = getattr(
                    wgt.TextureUsages, "RENDER_ATTACHMENT", None
                )
                if (
                    render_attachment
                    and render_attachment in format_features.allowed_usages
                ):
                    if hasattr(desc, "dimension"):
                        # Render targets must be 2D
                        d2 = (
                            getattr(wgt.TextureDimension, "D2", None)
                            if hasattr(wgt, "TextureDimension")
                            else None
                        )
                        can_use_color_target = d2 is not None and desc.dimension == d2

            if can_use_color_target and hasattr(wgt, "TextureUses"):
                uses |= getattr(wgt.TextureUses, "COLOR_TARGET", 0)
            elif hasattr(wgt, "TextureUses"):
                uses |= getattr(wgt.TextureUses, "COPY_DST", 0)

    return uses


def map_texture_usage_from_hal(uses: Any) -> Any:
    """
    Map HAL texture uses to wgpu texture usages.

    Args:
        uses: The HAL texture uses (wgt.TextureUses).

    Returns:
        The wgpu texture usages (wgt.TextureUsages).
    """
    if wgt is None:
        return 0

    u = 0

    # Helper functions
    def has_use(use_name: str) -> bool:
        if hasattr(wgt, "TextureUses") and hasattr(uses, "__contains__"):
            use_flag = getattr(wgt.TextureUses, use_name, None)
            if use_flag is not None:
                return use_flag in uses or (
                    hasattr(uses, "value") and (uses.value & use_flag) != 0
                )
        return False

    def set_usage(usage_name: str, condition: bool):
        nonlocal u
        if condition and hasattr(wgt, "TextureUsages"):
            usage_flag = getattr(wgt.TextureUsages, usage_name, 0)
            u |= usage_flag

    # Map each use to usage
    set_usage("COPY_SRC", has_use("COPY_SRC"))
    set_usage("COPY_DST", has_use("COPY_DST"))
    set_usage("TEXTURE_BINDING", has_use("RESOURCE"))

    # STORAGE_BINDING if any storage use is present
    storage_uses = (
        has_use("STORAGE_READ_ONLY")
        or has_use("STORAGE_WRITE_ONLY")
        or has_use("STORAGE_READ_WRITE")
    )
    set_usage("STORAGE_BINDING", storage_uses)

    set_usage("RENDER_ATTACHMENT", has_use("COLOR_TARGET"))
    set_usage("STORAGE_ATOMIC", has_use("STORAGE_ATOMIC"))
    set_usage("TRANSIENT", has_use("TRANSIENT"))

    return u


def check_texture_dimension_size(
    dimension: Any,
    extent: Any,
    sample_size: int,
    limits: Any,
) -> Optional[Exception]:
    """
    Check the requested texture size against the supported limits.

    This function implements the texture size and sample count checks in
    the WebGPU specification.

    Args:
        dimension: The texture dimension (wgt.TextureDimension).
        extent: The texture extent (wgt.Extent3d).
        sample_size: The sample count.
        limits: The device limits (wgt.Limits).

    Returns:
        None if validation passes, otherwise an error exception.
    """
    if wgt is None or not hasattr(extent, "width"):
        return None

    # Extract extent dimensions
    width = getattr(extent, "width", 0)
    height = getattr(extent, "height", 0)
    depth_or_array_layers = getattr(extent, "depth_or_array_layers", 0)

    # Determine limits based on dimension
    extent_limits = [0, 0, 0]
    sample_limit = 1

    if hasattr(wgt, "TextureDimension"):
        D1 = getattr(wgt.TextureDimension, "D1", None)
        D2 = getattr(wgt.TextureDimension, "D2", None)
        D3 = getattr(wgt.TextureDimension, "D3", None)

        if dimension == D1:
            extent_limits = [getattr(limits, "max_texture_dimension_1d", 8192), 1, 1]
            sample_limit = 1
        elif dimension == D2:
            extent_limits = [
                getattr(limits, "max_texture_dimension_2d", 8192),
                getattr(limits, "max_texture_dimension_2d", 8192),
                getattr(limits, "max_texture_array_layers", 256),
            ]
            sample_limit = 32
        elif dimension == D3:
            max_3d = getattr(limits, "max_texture_dimension_3d", 2048)
            extent_limits = [max_3d, max_3d, max_3d]
            sample_limit = 1

    # Check each dimension
    dimensions = [
        ("X", width, extent_limits[0]),
        ("Y", height, extent_limits[1]),
        ("Z", depth_or_array_layers, extent_limits[2]),
    ]

    for dim_name, given, limit in dimensions:
        if given == 0:
            return ValueError(f"Texture dimension {dim_name} cannot be zero")
        if given > limit:
            return ValueError(
                f"Texture dimension {dim_name} ({given}) exceeds limit ({limit})"
            )

    # Check sample count
    if sample_size == 0:
        return ValueError("Sample count cannot be zero")
    if sample_size > sample_limit:
        return ValueError(
            f"Sample count ({sample_size}) exceeds limit ({sample_limit})"
        )
    if not (sample_size & (sample_size - 1)) == 0:  # Check if power of 2
        return ValueError(f"Sample count ({sample_size}) must be a power of 2")

    return None


def bind_group_layout_flags(features: Any) -> Any:
    """
    Convert wgpu features to HAL bind group layout flags.

    Args:
        features: The wgpu features (wgt.Features).

    Returns:
        The HAL bind group layout flags (hal.BindGroupLayoutFlags).
    """
    if hal is None or not hasattr(hal, "BindGroupLayoutFlags"):
        return 0

    flags = 0

    # Check if PARTIALLY_BOUND_BINDING_ARRAY feature is enabled
    if wgt and hasattr(wgt, "Features") and hasattr(features, "__contains__"):
        partially_bound = getattr(wgt.Features, "PARTIALLY_BOUND_BINDING_ARRAY", None)
        if partially_bound and partially_bound in features:
            flags |= getattr(hal.BindGroupLayoutFlags, "PARTIALLY_BOUND", 0)

    return flags
