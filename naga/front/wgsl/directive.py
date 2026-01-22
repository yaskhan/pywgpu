"""
WGSL directive handling (enable, requires, diagnostic).

Translated from wgpu-trunk/naga/src/front/wgsl/parse/directive.rs

This module handles parsing and processing of WGSL directives.
"""

from typing import Set, List, Optional
from enum import Enum
from dataclasses import dataclass


class EnableExtension(Enum):
    """
    Extensions that can be enabled with the 'enable' directive.
    
    These correspond to WGSL language extensions that add new features.
    """
    # Implemented extensions
    F16 = "f16"
    DUAL_SOURCE_BLENDING = "dual_source_blending"
    WGPU_MESH_SHADER = "wgpu_mesh_shader"
    
    # Unimplemented extensions (for future support)
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


class LanguageExtension(Enum):
    """
    Language extensions that affect parsing behavior.
    
    These are different from enable extensions - they change
    how the language itself is parsed.
    """
    # Implemented language extensions
    POINTER_COMPOSITE_ACCESS = "pointer_composite_access"
    
    # Unimplemented language extensions
    PACKED_4X8_INTEGER_DOT_PRODUCT = "packed_4x8_integer_dot_product"
    UNRESTRICTED_POINTER_PARAMETERS = "unrestricted_pointer_parameters"
    SUBGROUPS = "subgroups"
    SUBGROUPS_F16 = "subgroups_f16"


@dataclass
class DirectiveKind:
    """Type of directive."""
    pass


@dataclass
class EnableDirective(DirectiveKind):
    """
    Enable directive: enable extension_name;
    
    Attributes:
        extensions: List of extensions to enable
    """
    extensions: List[EnableExtension]


@dataclass
class RequiresDirective(DirectiveKind):
    """
    Requires directive: requires feature_name;
    
    Attributes:
        features: List of required features
    """
    features: List[LanguageExtension]


@dataclass
class DiagnosticDirective(DirectiveKind):
    """
    Diagnostic directive: diagnostic(severity, rule);
    
    Attributes:
        severity: Diagnostic severity (off, info, warning, error)
        rule: Triggering rule name
    """
    severity: str
    rule: str


class EnableExtensions:
    """
    Tracks which extensions are currently enabled.
    
    This is used during parsing to determine which features
    are available.
    """
    
    def __init__(self):
        """Initialize with no extensions enabled."""
        self._enabled: Set[EnableExtension] = set()
    
    def enable(self, extension: EnableExtension) -> None:
        """
        Enable an extension.
        
        Args:
            extension: Extension to enable
        """
        self._enabled.add(extension)
    
    def is_enabled(self, extension: EnableExtension) -> bool:
        """
        Check if an extension is enabled.
        
        Args:
            extension: Extension to check
            
        Returns:
            True if enabled
        """
        return extension in self._enabled
    
    def contains(self, extension: EnableExtension) -> bool:
        """
        Check if an extension is enabled (alias for is_enabled).
        
        Args:
            extension: Extension to check
            
        Returns:
            True if enabled
        """
        return self.is_enabled(extension)
    
    def __contains__(self, extension: EnableExtension) -> bool:
        """Support 'in' operator."""
        return self.is_enabled(extension)


class LanguageExtensions:
    """
    Tracks which language extensions are active.
    
    Language extensions affect how the parser behaves.
    """
    
    def __init__(self):
        """Initialize with no extensions active."""
        self._active: Set[LanguageExtension] = set()
    
    def activate(self, extension: LanguageExtension) -> None:
        """
        Activate a language extension.
        
        Args:
            extension: Extension to activate
        """
        self._active.add(extension)
    
    def is_active(self, extension: LanguageExtension) -> bool:
        """
        Check if a language extension is active.
        
        Args:
            extension: Extension to check
            
        Returns:
            True if active
        """
        return extension in self._active
    
    def __contains__(self, extension: LanguageExtension) -> bool:
        """Support 'in' operator."""
        return self.is_active(extension)


def parse_enable_directive(extensions_text: List[str]) -> EnableDirective:
    """
    Parse an enable directive.
    
    Args:
        extensions_text: List of extension names
        
    Returns:
        EnableDirective with parsed extensions
        
    Raises:
        ParseError: If extension name is unknown
    """
    from .error import ParseError
    
    extensions = []
    for ext_name in extensions_text:
        try:
            ext = EnableExtension(ext_name)
            extensions.append(ext)
        except ValueError:
            # Unknown extension
            raise ParseError(
                message=f"unknown extension: '{ext_name}'",
                labels=[(0, 0, "")],  # TODO: Proper span
                notes=["Valid extensions: " + ", ".join(e.value for e in EnableExtension)]
            )
    
    return EnableDirective(extensions)


def parse_requires_directive(features_text: List[str]) -> RequiresDirective:
    """
    Parse a requires directive.
    
    Args:
        features_text: List of feature names
        
    Returns:
        RequiresDirective with parsed features
        
    Raises:
        ParseError: If feature name is unknown
    """
    from .error import ParseError
    
    features = []
    for feat_name in features_text:
        try:
            feat = LanguageExtension(feat_name)
            features.append(feat)
        except ValueError:
            # Unknown feature
            raise ParseError(
                message=f"unknown language feature: '{feat_name}'",
                labels=[(0, 0, "")],  # TODO: Proper span
                notes=["Valid features: " + ", ".join(f.value for f in LanguageExtension)]
            )
    
    return RequiresDirective(features)


def parse_diagnostic_directive(severity: str, rule: str) -> DiagnosticDirective:
    """
    Parse a diagnostic directive.
    
    Args:
        severity: Severity level (off, info, warning, error)
        rule: Triggering rule name
        
    Returns:
        DiagnosticDirective
        
    Raises:
        ParseError: If severity is invalid
    """
    from .error import ParseError
    
    valid_severities = ['off', 'info', 'warning', 'error']
    if severity not in valid_severities:
        raise ParseError(
            message=f"invalid diagnostic severity: '{severity}'",
            labels=[(0, 0, "")],  # TODO: Proper span
            notes=[f"Valid severities: {', '.join(valid_severities)}"]
        )
    
    return DiagnosticDirective(severity, rule)
