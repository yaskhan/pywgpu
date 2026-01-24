"""
Main GLSL parser implementation.

This module provides the main parser functionality for GLSL shaders,
handling preprocessing directives, declarations, and expression parsing.
"""

from typing import Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass


class DirectiveKind(Enum):
    """Types of preprocessing directives."""
    VERSION = "version"
    EXTENSION = "extension"
    PRAGMA = "pragma"
    DEFINE = "define"
    UNDEF = "undef"
    IFDEF = "ifdef"
    IFNDEF = "ifndef"
    IF = "if"
    ELSE = "else"
    ELIF = "elif"
    ENDIF = "endif"
    ERROR = "error"
    LINE = "line"


class ExtensionBehavior(Enum):
    """Extension behavior modes."""
    REQUIRE = "require"
    ENABLE = "enable"
    WARN = "warn"
    DISABLE = "disable"


class PreprocessorError(Enum):
    """Preprocessor error types."""
    UNEXPECTED_TOKEN = "unexpected_token"
    UNEXPECTED_NEW_LINE = "unexpected_new_line"
    UNEXPECTED_EOF = "unexpected_eof"
    INVALID_DIRECTIVE = "invalid_directive"


@dataclass
class Directive:
    """Preprocessing directive information."""
    kind: DirectiveKind
    name: Optional[str] = None
    behavior: Optional[ExtensionBehavior] = None
    tokens: List[Any] = None


class Parser:
    """Main GLSL parser implementation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.tokens: List[Any] = []
        self.current_token_index: int = 0
        self.directives: List[Directive] = []
        self.extensions: Dict[str, ExtensionBehavior] = {}
        self.optimize: bool = True
        self.debug: bool = False
    
    def parse(self, tokens: List[Any], options: Optional[Any] = None) -> List[Any]:
        """
        Process tokens through the preprocessing and directive handling pipeline.
        """
        # This implementation processes a list of tokens and handles 
        # preprocessing directives like #version, #extension, #pragma.
        self.tokens = tokens
        self.current_token_index = 0
        
        # Directive handling is typically done by the GlslParser through callbacks,
        # but here we can implement a base pass that filters out or processes them.
        processed_tokens = []
        while self.current_token_index < len(self.tokens):
             token = self.tokens[self.current_token_index]
             # In our lexer, directives are often already separated or marked
             if hasattr(token, 'value') and str(token.value).lower() == "directive":
                  self.handle_directive(token.data, token.meta)
             else:
                  processed_tokens.append(token)
             self.current_token_index += 1
             
        return processed_tokens
    
    def handle_directive(self, directive: Directive, meta: Any) -> None:
        """
        Handle preprocessing directives.
        
        Args:
            directive: The directive to handle
            meta: Metadata about directive location
        """
        if directive.kind == DirectiveKind.VERSION:
            self._handle_version_directive(directive, meta)
        elif directive.kind == DirectiveKind.EXTENSION:
            self._handle_extension_directive(directive, meta)
        elif directive.kind == DirectiveKind.PRAGMA:
            self._handle_pragma_directive(directive, meta)
    
    def _handle_version_directive(self, directive: Directive, meta: Any) -> None:
        """
        Handle #version directive.
        
        Args:
            directive: Version directive
            meta: Metadata
        """
        if not directive.tokens:
            self.errors.append("Version number required")
            return

        try:
            version_val = int(directive.tokens[0].value) if hasattr(directive.tokens[0], 'value') else int(directive.tokens[0])
        except (ValueError, AttributeError):
            self.errors.append("Invalid version number")
            return

        profile = None
        if len(directive.tokens) > 1:
            raw_profile = directive.tokens[1].value if hasattr(directive.tokens[1], 'value') else str(directive.tokens[1])
            try:
                from .parser import Profile
                profile = Profile(raw_profile)
            except ValueError:
                self.errors.append(f"Invalid profile: {raw_profile}")
                return

        # Store these in metadata via a callback or direct access if possible
        # For now, we store locally in the main parser instance which GlslParser can query
        self.version = version_val
        self.profile = profile
    
    def _handle_extension_directive(self, directive: Directive, meta: Any) -> None:
        """
        Handle #extension directive.
        """
        # Verify extension name and behavior are present
        if not directive.name:
            self.errors.append("Extension name required")
            return
        
        if not directive.behavior:
            self.errors.append("Extension behavior required")
            return
        
        # Extension behaviors:
        # - require: Error if not supported
        # - enable: Enable if supported, warn (log) if not
        # - warn: Enable if supported, warn (log) if not
        # - disable: Disable extension
        
        if directive.name == "all":
            self._handle_all_extension(directive.behavior, meta)
        else:
            self._handle_specific_extension(directive.name, directive.behavior, meta)

    def _handle_all_extension(self, behavior: ExtensionBehavior, meta: Any) -> None:
        """Handle #extension all directive."""
        # '#extension all : behavior' applies to all extensions.
        if behavior == ExtensionBehavior.DISABLE:
             # Disabling 'all' effectively resets enabled extensions to empty set.
             self.extensions.clear()
        elif behavior == ExtensionBehavior.WARN:
             # This mode would cause warnings on any extension usage.
             # We track this state to report during later stages.
             self.warn_on_all_extensions = True
        elif behavior in [ExtensionBehavior.ENABLE, ExtensionBehavior.REQUIRE]:
             # GLSL spec: "#extension all : enable" is not allowed.
             self.errors.append("'all' extension cannot be enabled or required")
    
    def _handle_specific_extension(self, name: str, behavior: ExtensionBehavior, meta: Any) -> None:
        """
        Handle specific extension directive.
        """
        if not self.is_extension_supported(name):
             if behavior == ExtensionBehavior.REQUIRE:
                 self.errors.append(f"Extension {name} is not supported")
             elif behavior in [ExtensionBehavior.ENABLE, ExtensionBehavior.WARN]:
                 # Just a warning if we had a warning system, but for now we'll just track it
                 pass
        
        if behavior == ExtensionBehavior.DISABLE:
             if name in self.extensions:
                 del self.extensions[name]
        else:
             self.extensions[name] = behavior
    
    def _handle_pragma_directive(self, directive: Directive, meta: Any) -> None:
        """
        Handle #pragma directive.
        """
        # Handle common pragmas:
        # - #pragma optimize(on/off)
        # - #pragma debug(on/off)
        # - vendor/backend specific pragmas
        
        if not directive.tokens:
            return
        
        pragma_name = str(directive.tokens[0])
        
        if pragma_name == "optimize":
            self._handle_optimize_pragma(directive.tokens[1:] if len(directive.tokens) > 1 else [])
        elif pragma_name == "debug":
            self._handle_debug_pragma(directive.tokens[1:] if len(directive.tokens) > 1 else [])
        else:
            # Unknown or vendor-specific pragma
            self._handle_unknown_pragma(directive.tokens)
    
    def _handle_optimize_pragma(self, args: List[Any]) -> None:
        """
        Handle #pragma optimize directive.
        """
        if not args:
            return
        
        state = str(args[0]).lower()
        if state == "on":
             self.optimize = True
        elif state == "off":
             self.optimize = False
    
    def _handle_debug_pragma(self, args: List[Any]) -> None:
        """
        Handle #pragma debug directive.
        """
        if not args:
            return
        
        state = str(args[0]).lower()
        if state == "on":
             self.debug = True
        elif state == "off":
             self.debug = False
    
    def _handle_unknown_pragma(self, tokens: List[Any]) -> None:
        """
        Handle unknown or vendor-specific pragmas.
        """
        # Unknown pragmas are typically ignored by the core compiler.
        # However, we store them for potential backend-specific processing.
        pragma_str = " ".join([str(t) for t in tokens])
        # Log unknown pragma for debugging/extension purposes
        # print(f"Unknown pragma: {pragma_str}")
        pass
        
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported extensions.
        """
        # Standard GLSL extensions that Naga might support or bypass
        return [
            "GL_ARB_compute_shader",
            "GL_ARB_shading_language_420pack",
            "GL_ARB_shader_image_load_store",
            "GL_EXT_gpu_shader4",
            "GL_EXT_shader_image_load_store",
        ]
    
    def is_extension_supported(self, name: str) -> bool:
        """
        Check if an extension is supported.
        """
        return name in self.get_supported_extensions()
    
    def get_enabled_extensions(self) -> List[str]:
        """
        Get list of enabled extensions.
        """
        return list(self.extensions.keys())
    
    def get_errors(self) -> List[str]:
        """Get parsing errors."""
        return self.errors.copy()
    
    def clear_errors(self) -> None:
        """Clear parsing errors."""
        self.errors.clear()