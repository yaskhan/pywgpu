"""
Enable extension definitions and validation.

Translated from wgpu-trunk/naga/src/front/wgsl/parse/directive/enable_extension.rs

This module defines extensions that can be enabled with the 'enable' directive.
"""

from enum import Enum
from typing import Set, Optional


class ImplementedEnableExtension(Enum):
    """
    Extensions that are implemented and can be enabled.
    
    These extensions add new features to WGSL.
    """
    # f16 support
    F16 = "f16"
    
    # Dual source blending
    DUAL_SOURCE_BLENDING = "dual_source_blending"
    
    # Mesh shaders (wgpu extension)
    WGPU_MESH_SHADER = "wgpu_mesh_shader"


class UnimplementedEnableExtension(Enum):
    """
    Extensions that are recognized but not yet implemented.
    
    These will produce an error if enabled.
    """
    # Chromium experimental extensions
    CHROMIUM_DISABLE_UNIFORMITY_ANALYSIS = "chromium_disable_uniformity_analysis"
    CHROMIUM_EXPERIMENTAL_DP4A = "chromium_experimental_dp4a"
    CHROMIUM_EXPERIMENTAL_FRAMEBUFFER_FETCH = "chromium_experimental_framebuffer_fetch"
    CHROMIUM_EXPERIMENTAL_PIXEL_LOCAL = "chromium_experimental_pixel_local"
    CHROMIUM_EXPERIMENTAL_READ_WRITE_STORAGE_TEXTURE = "chromium_experimental_read_write_storage_texture"
    CHROMIUM_EXPERIMENTAL_SUBGROUPS = "chromium_experimental_subgroups"
    CHROMIUM_INTERNAL_DUAL_SOURCE_BLENDING = "chromium_internal_dual_source_blending"
    CHROMIUM_INTERNAL_GRAPHITE = "chromium_internal_graphite"
    CHROMIUM_INTERNAL_INPUT_ATTACHMENTS = "chromium_internal_input_attachments"
    CHROMIUM_INTERNAL_RELAXED_UNIFORM_LAYOUT = "chromium_internal_relaxed_uniform_layout"


class EnableExtensionSet:
    """
    Set of enabled extensions.
    
    Tracks which extensions are currently enabled during parsing.
    """
    
    def __init__(self):
        """Initialize with no extensions enabled."""
        self._enabled: Set[ImplementedEnableExtension] = set()
    
    def enable(self, extension: ImplementedEnableExtension) -> None:
        """
        Enable an extension.
        
        Args:
            extension: Extension to enable
        """
        self._enabled.add(extension)
    
    def is_enabled(self, extension: ImplementedEnableExtension) -> bool:
        """
        Check if an extension is enabled.
        
        Args:
            extension: Extension to check
            
        Returns:
            True if enabled
        """
        return extension in self._enabled
    
    def contains(self, extension: ImplementedEnableExtension) -> bool:
        """
        Check if an extension is enabled (alias for is_enabled).
        
        Args:
            extension: Extension to check
            
        Returns:
            True if enabled
        """
        return self.is_enabled(extension)
    
    def __contains__(self, extension: ImplementedEnableExtension) -> bool:
        """Support 'in' operator."""
        return self.is_enabled(extension)
    
    def get_all(self) -> Set[ImplementedEnableExtension]:
        """
        Get all enabled extensions.
        
        Returns:
            Set of enabled extensions
        """
        return self._enabled.copy()


def parse_enable_extension(name: str, span: tuple[int, int]) -> ImplementedEnableExtension:
    """
    Parse an enable extension name.
    
    Args:
        name: Extension name
        span: Source location
        
    Returns:
        ImplementedEnableExtension
        
    Raises:
        ParseError: If extension is unknown or unimplemented
    """
    from ..error import ParseError
    
    # Try implemented extensions
    try:
        return ImplementedEnableExtension(name)
    except ValueError:
        pass
    
    # Try unimplemented extensions
    try:
        unimpl = UnimplementedEnableExtension(name)
        raise ParseError(
            message=f"extension '{name}' is not implemented",
            labels=[(span[0], span[1], "")],
            notes=[f"This extension is recognized but not yet supported"]
        )
    except ValueError:
        pass
    
    # Unknown extension
    valid_extensions = [e.value for e in ImplementedEnableExtension]
    raise ParseError(
        message=f"unknown extension: '{name}'",
        labels=[(span[0], span[1], "")],
        notes=[f"Valid extensions: {', '.join(valid_extensions)}"]
    )


def validate_extension_requirements(
    extension: ImplementedEnableExtension,
    enabled_extensions: EnableExtensionSet,
    span: tuple[int, int]
) -> None:
    """
    Validate that extension requirements are met.
    
    Some extensions may require other extensions to be enabled first.
    """
    from ..error import ParseError
    
    # Define dependencies
    # Key: Extension being enabled, Value: List of required extensions
    dependencies = {
        # Example: ImplementedEnableExtension.SOME_EXT: [ImplementedEnableExtension.F16]
    }
    
    requirements = dependencies.get(extension, [])
    for req in requirements:
        if not enabled_extensions.is_enabled(req):
            raise ParseError(
                message=f"extension '{extension.value}' requires extension '{req.value}' to be enabled",
                labels=[(span[0], span[1], "")],
                notes=[f"Add 'enable {req.value};' before this directive"]
            )


def get_extension_features(extension: ImplementedEnableExtension) -> Set[str]:
    """
    Get the features provided by an extension.
    
    Args:
        extension: Extension to query
        
    Returns:
        Set of feature names
    """
    features = {
        ImplementedEnableExtension.F16: {
            'f16_type',
            'f16_literals',
        },
        ImplementedEnableExtension.DUAL_SOURCE_BLENDING: {
            'blend_src_attribute',
            'dual_source_blending',
        },
        ImplementedEnableExtension.WGPU_MESH_SHADER: {
            'mesh_shader_stage',
            'task_shader_stage',
            'per_primitive_attribute',
        },
    }
    
    return features.get(extension, set())
