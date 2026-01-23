from typing import Any, Optional, Dict
from .. import Parser
from ...ir import Module
from .recursive_parser import WgslRecursiveParser
from .lexer import TokenKind


from .error import ParseError


class Options:
    """WGSL parsing options."""

    def __init__(self):
        self.entry_points: list[str] = []
        self.shader_stage: Optional[str] = None
        self.workgroup_size: tuple[int, int, int] = (1, 1, 1)
        self.diagnostics: bool = True
        self.max_recursion_depth: int = 128

    @classmethod
    def new(cls) -> "Options":
        """Create new options with default values."""
        return cls()

    def set_shader_stage(self, stage: str) -> None:
        """Set the shader stage for parsing."""
        self.shader_stage = stage

    def add_entry_point(self, name: str) -> None:
        """Add an entry point to include in parsing."""
        self.entry_points.append(name)


class Frontend:
    """
    Main WGSL frontend for parsing and converting WGSL source to NAGA IR.

    This class orchestrates the parsing pipeline:
    1. Lexical analysis and parsing (Lexer + Parser)
    2. Semantic analysis and indexing (Index)
    3. Lowering to NAGA IR (Lowerer)
    """

    def __init__(self, options: Optional[Options] = None):
        self.options = options or Options.new()
        self.errors: list[ParseError] = []
        self.diagnostics: list[str] = []

    @classmethod
    def new(cls) -> "Frontend":
        """Create a new WGSL frontend with default options."""
        return cls()

    @classmethod
    def new_with_options(cls, options: Options) -> "Frontend":
        """Create a new WGSL frontend with custom options."""
        return cls(options)

    def set_options(self, options: Options) -> None:
        """Update the parsing options."""
        self.options = options

    def parse(self, source: str) -> Module:
        """
        Parse WGSL source code into a NAGA IR module.
        """
        try:
            return self._inner_parse(source)
        except ParseError as e:
            if self.options.diagnostics:
                self.diagnostics.append(e.emit_to_string(source))
            raise

    def _inner_parse(self, source: str) -> Module:
        """Internal parsing implementation."""
        # Step 1: Parse source into intermediate representation
        tu = self._parse_to_tu(source)

        # Step 2: Generate index for semantic analysis
        from .index import Index
        index = Index.generate(tu)

        # Step 3: Lower to NAGA IR
        module = self._lower_to_ir(tu, index)

        return module

    def _parse_to_tu(self, source: str) -> Any:
        """Parse source code into a translation unit (AST)."""
        from .lexer import Lexer
        from .ast import TranslationUnit
        from .enable_extension import EnableExtensionSet
        from .language_extension import LanguageExtensionSet
        
        lexer = Lexer(source)
        enable_extensions = EnableExtensionSet()
        language_extensions = LanguageExtensionSet()
        
        parser = WgslRecursiveParser(
            lexer,
            enable_extensions,
            language_extensions,
            self.options
        )
        
        decls = []
        directives = []
        
        try:
            while True:
                token = parser.peek()
                if token is None or token.kind == TokenKind.EOF:
                    break
                
                if parser.is_directive():
                    directive = parser.parse_directive()
                    directives.append(directive)
                    continue
                
                decl = parser.parse_global_decl()
                if decl is not None:
                    decls.append(decl)
        
        except Exception as e:
            if not isinstance(e, ParseError):
                raise ParseError(message=str(e), labels=[], notes=[])
            raise
        
        return TranslationUnit(decls=decls, directives=directives)

    def _lower_to_ir(self, tu: Any, index: Any) -> Module:
        """Lower the parsed representation to NAGA IR."""
        from .lower import Lowerer
        lowerer = Lowerer(index)
        return lowerer.lower(tu)

    def add_diagnostic(self, message: str, severity: str = "error") -> None:
        """Add a diagnostic message during parsing."""
        self.diagnostics.append(f"[{severity}] {message}")

    def get_diagnostics(self) -> list[str]:
        """Get all diagnostic messages."""
        return self.diagnostics.copy()


    def add_diagnostic(self, message: str, severity: str = "error") -> None:
        """Add a diagnostic message during parsing."""
        self.diagnostics.append(f"[{severity}] {message}")

    def get_diagnostics(self) -> list[str]:
        """Get all diagnostic messages."""
        return self.diagnostics.copy()


class WgslParser(Parser):
    """
    WGSL parser implementation - convenience interface.

    This provides a simple interface for parsing WGSL source code,
    delegating to the full Frontend implementation.
    """

    def __init__(self, options: Optional[Options] = None) -> None:
        self.frontend = Frontend.new_with_options(options or Options.new())

    def parse(self, source: str) -> Module:
        """
        Parse WGSL source code into a NAGA IR module.

        Args:
            source: WGSL shader source code

        Returns:
            A NAGA IR Module representing the parsed shader
        """
        return self.frontend.parse(source)

    @staticmethod
    def parse_str(source: str) -> Module:
        """
        Static convenience method to parse WGSL source.

        Args:
            source: WGSL shader source code

        Returns:
            A NAGA IR Module representing the parsed shader
        """
        frontend = Frontend.new()
        return frontend.parse(source)


def parse_str(source: str) -> Module:
    """
    Convenience function to parse WGSL source code.

    Args:
        source: WGSL shader source code

    Returns:
        A NAGA IR Module representing the parsed shader

    Raises:
        ParseError: If parsing fails
    """
    frontend = Frontend.new()
    return frontend.parse(source)
