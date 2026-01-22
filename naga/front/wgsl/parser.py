from typing import Any, Optional, Dict
from .. import Parser
from ...ir import Module
from .recursive_parser import WgslRecursiveParser


class ParseError(Exception):
    """WGSL parsing error with detailed diagnostics."""

    def __init__(self, message: str, source: str, offset: int = 0):
        super().__init__(message)
        self.message = message
        self.source = source
        self.offset = offset

    def emit_to_string(self, source: str) -> str:
        """Format the error with context from source code."""
        lines = source.split("\n")
        line_num = 0
        char_count = 0

        for i, line in enumerate(lines):
            if char_count + len(line) >= self.offset:
                line_num = i
                break
            char_count += len(line) + 1  # +1 for newline

        error_line = lines[line_num] if line_num < len(lines) else ""
        column = self.offset - char_count

        return f"Error at line {line_num + 1}, column {column}: {self.message}\n{error_line}\n{' ' * column}^"


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

        Args:
            source: WGSL shader source code as a string

        Returns:
            A NAGA IR Module representing the parsed shader

        Raises:
            ParseError: If parsing fails with syntax or semantic errors
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
        index = self._generate_index(tu)

        # Step 3: Lower to NAGA IR
        module = self._lower_to_ir(tu, index)

        return module

    def _parse_to_tu(self, source: str) -> Any:
        """
        Parse source code into a translation unit (intermediate representation).
        
        This creates a lexer and parser to build the AST.
        
        Args:
            source: WGSL source code
            
        Returns:
            TranslationUnit (AST)
            
        Raises:
            ParseError: If parsing fails
        """
        from .lexer import Lexer
        from .ast import TranslationUnit, GlobalDecl
        from .enable_extension import EnableExtensionSet
        from .language_extension import LanguageExtensionSet
        
        # Create lexer
        lexer = Lexer(source)
        
        # Create extension tracking
        enable_extensions = EnableExtensionSet()
        language_extensions = LanguageExtensionSet()
        
        # Create parser
        parser = WgslRecursiveParser(
            lexer,
            enable_extensions,
            language_extensions,
            self.options
        )
        
        # Parse global declarations
        decls = []
        directives = []
        
        try:
            while True:
                token = parser.peek()
                if token is None:
                    break
                
                # Check for directives
                if parser.is_directive():
                    directive = parser.parse_directive()
                    directives.append(directive)
                    continue
                
                # Parse global declaration
                decl = parser.parse_global_decl()
                if decl is not None:
                    decls.append(decl)
        
        except Exception as e:
            # Convert to ParseError if needed
            if not isinstance(e, ParseError):
                raise ParseError(
                    message=str(e),
                    labels=[(0, 0, "")],
                    notes=[]
                )
            raise
        
        # Create translation unit
        tu = TranslationUnit(decls=decls, directives=directives)
        return tu

    def _generate_index(self, tu: Any) -> Any:
        """Generate semantic index from parsed translation unit."""
        # This would analyze the TU and build symbol tables, type information, etc.
        # For now, return empty index
        return {}

    def _lower_to_ir(self, tu: Any, index: Any) -> Module:
        """
        Lower the parsed representation to NAGA IR.
        
        This is the final stage of WGSL parsing that converts the analyzed
        AST into NAGA's intermediate representation.
        
        Args:
            tu: Translation unit (parsed AST)
            index: Semantic index with dependency ordering
            
        Returns:
            Complete NAGA IR Module
        """
        module = Module()

        # The lowering process follows these steps (from Rust Lowerer):
        
        # 1. Initialize lowering context
        #    - Create expression arena for global scope
        #    - Set up type resolution tables
        #    - Initialize handle maps for declarations
        
        # 2. Process global declarations in dependency order
        #    for decl_handle in index.visit_ordered():
        #        decl = tu.decls[decl_handle]
        #        
        #        Match on declaration kind:
        #        - Struct: Lower struct type definition
        #          * Process struct members with bindings
        #          * Add to module.types
        #          * Register in type lookup table
        #        
        #        - Type: Lower type alias
        #          * Resolve aliased type
        #          * Register in type lookup table
        #        
        #        - Const: Lower constant declaration
        #          * Evaluate constant expression
        #          * Add to module.constants
        #          * Register in constant lookup table
        #        
        #        - Override: Lower pipeline-overridable constant
        #          * Create override with default value
        #          * Add to module.overrides
        #          * Register in override lookup table
        #        
        #        - Var: Lower global variable
        #          * Determine address space and access mode
        #          * Process bindings (group, binding, location, etc.)
        #          * Add to module.global_variables
        #          * Register in variable lookup table
        #        
        #        - Fn: Lower function declaration
        #          * Process function signature (parameters, return type)
        #          * Lower function body statements
        #          * Handle local variables
        #          * Convert expressions to IR
        #          * Add to module.functions
        #          * If entry point, add to module.entry_points
        #        
        #        - ConstAssert: Evaluate and verify const assertion
        #          * Evaluate assertion expression
        #          * Verify it evaluates to true
        #          * Emit error if false
        
        # 3. Type conversion (lower/conversion.rs)
        #    - Scalar types: bool, i32, u32, f32, f16
        #    - Vector types: vec2, vec3, vec4
        #    - Matrix types: mat2x2, mat3x3, mat4x4, etc.
        #    - Array types: array<T, N> or array<T>
        #    - Struct types: user-defined structs
        #    - Pointer types: ptr<address_space, T>
        #    - Atomic types: atomic<T>
        #    - Sampler types: sampler, sampler_comparison
        #    - Texture types: texture_1d, texture_2d, etc.
        #    - Image types: texture_storage_*
        
        # 4. Expression lowering (lower/mod.rs)
        #    - Literals: bool, int, float
        #    - Identifiers: resolve to variables, constants, functions
        #    - Binary operations: +, -, *, /, %, &, |, ^, <<, >>, etc.
        #    - Unary operations: -, !, ~
        #    - Function calls: built-in and user-defined
        #    - Member access: struct.field, vector.x, etc.
        #    - Index access: array[i], vector[i]
        #    - Construction: vec3(1.0, 2.0, 3.0), MyStruct(...)
        #    - Type casts: f32(x), i32(y)
        
        # 5. Statement lowering (lower/mod.rs)
        #    - Variable declarations: var x: i32 = 0;
        #    - Assignments: x = y;
        #    - Compound assignments: x += y;
        #    - If statements: if (cond) { } else { }
        #    - Switch statements: switch (x) { case 0: { } default: { } }
        #    - Loops: loop { }, while (cond) { }, for (...) { }
        #    - Break and continue
        #    - Return statements: return expr;
        #    - Discard: discard;
        #    - Function calls as statements
        
        # 6. Entry point handling
        #    - Detect @vertex, @fragment, @compute attributes
        #    - Process workgroup_size for compute shaders
        #    - Handle entry point inputs/outputs
        #    - Create EntryPoint with proper stage
        
        # 7. Validation during lowering
        #    - Type checking
        #    - Address space validation
        #    - Binding validation
        #    - Expression scope validation
        #    - Control flow validation

        return module

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
