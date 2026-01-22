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
    
    def parse(self, source: str) -> Any:
        """
        Parse GLSL source code.
        
        Args:
            source: GLSL source code to parse
            
        Returns:
            Parsed module or AST
        """
        # TODO: Implement full parsing pipeline
        # 1. Lexical analysis
        # 2. Preprocessing
        # 3. Parsing declarations
        # 4. Building AST
        
        return None
    
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
        # TODO: Implement version directive handling
        # Should parse version number and profile (core, compatibility, es)
        pass
    
    def _handle_extension_directive(self, directive: Directive, meta: Any) -> None:
        """
        Handle #extension directive.
        
        Args:
            directive: Extension directive
            meta: Metadata
        """
        # TODO: Proper extension handling
        # - Checking for extension support in the compiler
        # - Handle behaviors such as warn
        # - Handle the all extension
        
        # Extension directive format:
        # #extension extension_name : behavior
        # #extension all : behavior
        
        if not directive.name:
            self.errors.append("Extension name required")
            return
        
        if not directive.behavior:
            self.errors.append("Extension behavior required")
            return
        
        # TODO: Implement extension support checking
        # This should:
        # 1. Check if the extension is supported by the compiler
        # 2. Apply the specified behavior (require, enable, warn, disable)
        # 3. Track enabled extensions
        # 4. Report errors for unsupported extensions
        
        # Handle special cases
        if directive.name == "all":
            # "all" extension affects all extensions
            self._handle_all_extension(directive.behavior, meta)
        else:
            # Specific extension
            self._handle_specific_extension(directive.name, directive.behavior, meta)
    
    def _handle_all_extension(self, behavior: ExtensionBehavior, meta: Any) -> None:
        """
        Handle #extension all directive.
        
        Args:
            behavior: Extension behavior
            meta: Metadata
        """
        # TODO: Implement "all" extension handling
        # This should apply the behavior to all available extensions
        # - require: All extensions must be supported
        # - enable: Enable all extensions
        # - warn: Warn about all extensions
        # - disable: Disable all extensions
        
        pass
    
    def _handle_specific_extension(self, name: str, behavior: ExtensionBehavior, meta: Any) -> None:
        """
        Handle specific extension directive.
        
        Args:
            name: Extension name
            behavior: Extension behavior
            meta: Metadata
        """
        # TODO: Implement specific extension handling
        # This should:
        # 1. Check if extension is known/supported
        # 2. Apply the behavior:
        #    - require: Extension must be supported, error if not
        #    - enable: Enable the extension, warn if not supported
        #    - warn: Enable with warning if not supported
        #    - disable: Disable the extension
        # 3. Track extension state
        # 4. Report appropriate messages
        
        pass
    
    def _handle_pragma_directive(self, directive: Directive, meta: Any) -> None:
        """
        Handle #pragma directive.
        
        Args:
            directive: Pragma directive
            meta: Metadata
        """
        # TODO: handle some common pragmas?
        # Common GLSL pragmas that could be handled include:
        # - #pragma optimize(on/off)
        # - #pragma debug(on/off)
        # - vendor-specific pragmas
        
        # TODO: Implement common pragma handling
        # This should recognize and handle common pragmas:
        
        if not directive.tokens:
            return
        
        pragma_name = directive.tokens[0] if directive.tokens else None
        
        if pragma_name == "optimize":
            self._handle_optimize_pragma(directive.tokens[1:] if len(directive.tokens) > 1 else [])
        elif pragma_name == "debug":
            self._handle_debug_pragma(directive.tokens[1:] if len(directive.tokens) > 1 else [])
        else:
            # Unknown pragma - could be vendor-specific
            self._handle_unknown_pragma(directive.tokens)
    
    def _handle_optimize_pragma(self, args: List[Any]) -> None:
        """
        Handle #pragma optimize directive.
        
        Args:
            args: Pragma arguments
        """
        # TODO: Implement optimize pragma
        # #pragma optimize(on) - enable optimizations
        # #pragma optimize(off) - disable optimizations
        
        if not args:
            return
        
        state = str(args[0]).lower()
        if state in ["on", "off"]:
            # Track optimization state
            pass
    
    def _handle_debug_pragma(self, args: List[Any]) -> None:
        """
        Handle #pragma debug directive.
        
        Args:
            args: Pragma arguments
        """
        # TODO: Implement debug pragma
        # #pragma debug(on) - enable debugging
        # #pragma debug(off) - disable debugging
        
        if not args:
            return
        
        state = str(args[0]).lower()
        if state in ["on", "off"]:
            # Track debug state
            pass
    
    def _handle_unknown_pragma(self, tokens: List[Any]) -> None:
        """
        Handle unknown or vendor-specific pragmas.
        
        Args:
            tokens: Pragma tokens
        """
        # TODO: Implement unknown pragma handling
        # Vendor-specific pragmas should be preserved or ignored
        # depending on the target backend
        
        # For now, just ignore unknown pragmas
        pass
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported extensions.
        
        Returns:
            List of supported extension names
        """
        # TODO: Implement extension support database
        # This should return all extensions supported by the compiler
        return []
    
    def is_extension_supported(self, name: str) -> bool:
        """
        Check if an extension is supported.
        
        Args:
            name: Extension name
            
        Returns:
            True if extension is supported
        """
        # TODO: Implement extension support checking
        return name in self.get_supported_extensions()
    
    def get_enabled_extensions(self) -> List[str]:
        """
        Get list of enabled extensions.
        
        Returns:
            List of enabled extension names
        """
        # TODO: Track enabled extensions
        return []
    
    def get_errors(self) -> List[str]:
        """Get parsing errors."""
        return self.errors.copy()
    
    def clear_errors(self) -> None:
        """Clear parsing errors."""
        self.errors.clear()