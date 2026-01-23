from typing import Any, Optional, Dict, List
from enum import Enum
from .. import Parser
from ...ir import Module
from . import ast, builtins, functions, offset, parser_main, types, variables
from .lexer import Lexer
from .token import TokenValue
from .sub_parsers import declarations, functions as parser_functions, types as parser_types, expressions, statements

# Import ShaderStage if it exists, otherwise define it locally
try:
    from ...module import ShaderStage
except ImportError:
    from enum import Enum

    class ShaderStage(Enum):
        VERTEX = "vertex"
        FRAGMENT = "fragment"
        COMPUTE = "compute"

        def compute_like(self) -> bool:
            """Check if this stage is compute-like."""
            return self == ShaderStage.COMPUTE


class Profile(Enum):
    """GLSL profile types."""

    CORE = "core"
    COMPATIBILITY = "compatibility"
    ES = "es"


class Precision(Enum):
    """GLSL precision qualifiers."""

    LOW = "lowp"
    MEDIUM = "mediump"
    HIGH = "highp"


class Options:
    """
    Per-shader options passed to parse method.

    Attributes:
        stage: The shader stage in the pipeline
        defines: Preprocessor definitions to be used
    """

    def __init__(self, stage: ShaderStage, defines: Optional[Dict[str, str]] = None):
        self.stage = stage
        self.defines = defines or {}

    @classmethod
    def from_stage(cls, stage: ShaderStage) -> "Options":
        """Create options from shader stage."""
        return cls(stage)


class ShaderMetadata:
    """
    Additional information about the GLSL shader.

    Attributes:
        version: The GLSL version specified in the shader
        profile: The GLSL profile specified in the shader
        stage: The shader stage in the pipeline
        workgroup_size: The workgroup size for compute shaders
        early_fragment_tests: Whether early fragment tests were requested
        extensions: Required or requested extensions
    """

    def __init__(self):
        self.version: int = 0
        self.profile: Profile = Profile.CORE
        self.stage: ShaderStage = ShaderStage.VERTEX
        self.workgroup_size: list[int] = [0, 0, 0]
        self.early_fragment_tests: bool = False
        self.extensions: set[str] = set()

    def reset(self, stage: ShaderStage) -> None:
        """Reset metadata for new parsing."""
        self.version = 0
        self.profile = Profile.CORE
        self.stage = stage
        self.workgroup_size = [1 if stage.compute_like() else 0] * 3
        self.early_fragment_tests = False
        self.extensions.clear()


class GlslParser(Parser):
    """
    GLSL parser implementation.

    This parser handles GLSL (OpenGL Shading Language) shaders and converts
    them into NAGA IR modules.
    """

    def __init__(self, options: Optional[Options] = None) -> None:
        self.options = options
        self.metadata = ShaderMetadata()
        self.errors: List[str] = []
        
        # Initialize sub-parsers
        self.builtins = builtins.Builtins()
        self.functions = functions.FunctionHandler()
        self.offset_calculator = offset.OffsetCalculator()
        self.type_parser = types.TypeParser()
        self.variable_handler = variables.VariableHandler()
        self.declaration_parser = declarations.DeclarationParser(None)
        self.function_parser = parser_functions.FunctionParser(None)
        self.type_parser_impl = parser_types.TypeParser(None)
        self.main_parser = parser_main.Parser()
        self.expression_parser = expressions.ExpressionParser(None)
        self.statement_parser = statements.StatementParser(None)
        
        # Internal lookup tables
        self.lookup_function: Dict[str, Any] = {}
        self.lookup_type: Dict[str, Any] = {}
        self.global_variables: List[Any] = []
        self.entry_args: List[Any] = []
        self.module: Optional[Module] = None

    def reset(self, stage: ShaderStage) -> None:
        """Reset the parser state for new parsing."""
        self.metadata.reset(stage)
        self.errors.clear()
        self.lookup_function.clear()
        self.lookup_type.clear()
        self.global_variables.clear()
        self.entry_args.clear()
        
        # Reset sub-parsers
        self.builtins = builtins.Builtins()
        self.functions = functions.FunctionHandler()
        self.offset_calculator = offset.OffsetCalculator()
        self.type_parser = types.TypeParser()
        self.variable_handler = variables.VariableHandler()
        self.declaration_parser = declarations.DeclarationParser(None)
        self.function_parser = parser_functions.FunctionParser(None)
        self.type_parser_impl = parser_types.TypeParser(None)
        self.main_parser = parser_main.Parser()
        self.expression_parser = expressions.ExpressionParser(None)
        self.statement_parser = statements.StatementParser(None)

    def parse(self, source: str, options: Optional[Options] = None) -> Module:
        """
        Parse a GLSL shader source string into a NAGA IR module.

        Args:
            source: The GLSL shader source code
            options: Parsing options including shader stage and defines

        Returns:
            A NAGA IR Module representing the parsed shader

        Raises:
            ValueError: If parsing fails due to errors
        """
        # Use provided options or stored options
        parse_options = options or self.options
        if parse_options is None:
            raise ValueError("No parsing options provided")

        # Reset parser state
        self.reset(parse_options.stage)
        self.module = Module()

        # Store options
        self.options = parse_options

        # 1. Lexical analysis - Create lexer from source and defines
        lexer = Lexer(source, parse_options.defines)
        tokens = lexer.tokenize()
        
        # 2. Main Parsing Loop
        ctx = ParsingContext(tokens)
        
        # 3. Handle directives at the start
        while not ctx.at_end(lexer):
            token = ctx.peek(self)
            if token and token.value == TokenValue.DIRECTIVE:
                # In real GLSL, directives can only be at start of lines
                # For now, let's just skip them or handle if they are #version
                ctx.bump(self)
                continue
            break
            
        while not ctx.at_end(lexer):
            try:
                # Parse global declarations
                decl = self.declaration_parser.parse_declaration(ctx, self)
                if decl is None:
                    if not ctx.at_end(lexer):
                        # Attempt recovery
                        ctx.bump(self)
                        continue
                    break
            except Exception as e:
                self.errors.append(f"Parse error: {str(e)}")
                if not ctx.at_end(lexer):
                    ctx.bump(self)
                else:
                    break
        
        if self.errors:
            # Report errors
            pass
            
        return self.module
        #    - Collect errors during parsing
        #    - Return errors if parsing fails
        
        # Use the module populated during parsing
        module = self.module
        
        # Initialize builtin functions
        self._initialize_builtin_functions(module)
        
        # TODO: Parse the source and build the module
        # This would involve:
        # - Tokenizing the source
        # - Parsing declarations
        # - Building the AST
        # - Converting to IR
        
        # Check for errors
        if self.errors:
            error_messages = [str(e) for e in self.errors]
            raise ValueError(f"GLSL parsing failed with {len(self.errors)} error(s): {error_messages}")

        return module

    def _initialize_builtin_functions(self, module: Module) -> None:
        """Initialize builtin functions in the module."""
        # TODO: Initialize all builtin functions
        # This should populate the lookup_function table with:
        # - Texture functions (texture, textureSize, etc.)
        # - Math functions (sin, cos, sqrt, etc.)
        # - Vector functions (dot, cross, normalize, etc.)
        # - Matrix functions (matrixCompMult, transpose, etc.)
        
        pass

    def handle_directive(self, directive: Any, meta: Any) -> None:
        """
        Handle preprocessing directives like #version, #extension, #pragma.

        Args:
            directive: The directive to handle
            meta: Metadata about the directive location
        """
        # TODO: Implement directive handling
        # This should handle:
        # - #version: Set metadata.version and metadata.profile
        # - #extension: Add to metadata.extensions
        # - #pragma: Handle pragma directives
        
        # Use the main parser for directive handling
        self.main_parser.handle_directive(directive, meta)
        
        # Sync metadata
        if hasattr(self.main_parser, 'version'):
            self.metadata.version = self.main_parser.version
        if hasattr(self.main_parser, 'profile') and self.main_parser.profile:
            self.metadata.profile = self.main_parser.profile

    def add_entry_point(self, function_handle: Any, ctx: Any) -> None:
        """
        Add an entry point to the module.

        Args:
            function_handle: Handle to the function to use as entry point
            ctx: Parsing context
        """
        # TODO: Implement entry point addition
        # This should:
        # 1. Create EntryPoint from function
        # 2. Set up entry point arguments from self.entry_args
        # 3. Add to module.entry_points

        pass

    def add_global_var(self, ctx: Any, declaration: Any) -> Any:
        """
        Add a global variable or constant to the module.

        Args:
            ctx: Parsing context
            declaration: Variable declaration

        Returns:
            GlobalOrConstant enum indicating what was added
        """
        from ...ir import GlobalVariable
        handle = self.variable_handler.add_global_variable(ctx, declaration)
        if isinstance(handle, GlobalVariable):
            idx = len(self.module.global_variables)
            self.module.global_variables.append(handle)
            return idx
        return handle

    def add_local_var(self, ctx: Any, declaration: Any) -> Any:
        """
        Add a local variable to the current function.

        Args:
            ctx: Parsing context
            declaration: Variable declaration

        Returns:
            Expression handle for the local variable
        """
        # TODO: Implement local variable addition
        # This should handle:
        # - Local variable declarations
        # - Function parameter handling
        # - Variable initialization

        return None

    def get_metadata(self) -> ShaderMetadata:
        """
        Get metadata about the parsed shader.

        Returns:
            ShaderMetadata containing version, profile, extensions, etc.
        """
        return self.metadata

    def resolve_builtin_function(self, name: str, args: List[Any]) -> Optional[Any]:
        """
        Resolve a builtin function call.

        Args:
            name: Function name
            args: Function arguments

        Returns:
            Resolved builtin function or None
        """
        return self.builtins.resolve_builtin_call(name, args)

    def check_type_conversion(self, expected_type: str, actual_type: str) -> Optional[Any]:
        """
        Check if conversion between types is possible.

        Args:
            expected_type: Expected type
            actual_type: Actual type

        Returns:
            Conversion function or None
        """
        return self.functions.check_conversion_compatibility(expected_type, actual_type)

    def calculate_type_offset(self, ty: Any, layout: ast.StructLayout) -> Optional[Any]:
        """
        Calculate type offset for layout.

        Args:
            ty: Type handle
            layout: Struct layout

        Returns:
            Type offset information
        """
        return self.offset_calculator.calculate_offset(ty, None, layout, None)

    def parse_image_type(self, tokens: List[str]) -> Optional[Any]:
        """
        Parse image type from tokens.

        Args:
            tokens: Image type tokens

        Returns:
            Parsed image type or None
        """
        return self.type_parser.parse_image_type(tokens)

    def add_error(self, message: str, location: Any = None) -> None:
        """
        Add an error to the parser.

        Args:
            message: Error message
            location: Error location
        """
        error_msg = f"Error at {location}: {message}" if location else f"Error: {message}"
        self.errors.append(error_msg)


class ParsingContext:
    """
    Context for parsing GLSL source code.
    
    Manages token stream, backtracking, and parsing state.
    """

    def __init__(self, tokens: List[Any]):
        """
        Initialize parsing context.

        Args:
            tokens: The list of tokens to parse
        """
        self.tokens = tokens
        self.cursor = 0
        self.backtracked_token: Optional[Any] = None
        self.last_meta: Any = None

    def at_end(self, lexer: Any = None) -> bool:
        """Check if parser is at end of token stream."""
        if self.backtracked_token is not None:
            return False
        return self.cursor >= len(self.tokens)

    def backtrack(self, token: Any) -> None:
        """
        Backtrack to a previously consumed token.

        Args:
            token: Token to backtrack to

        Raises:
            ValueError: If already backtracked without bumping
        """
        if self.backtracked_token is not None:
            raise ValueError("Parser tried to backtrack twice in a row")
        self.backtracked_token = token

    def bump(self, frontend: GlslParser) -> Any:
        """
        Consume and return the next token.

        Args:
            frontend: The parser frontend

        Returns:
            The next token

        Raises:
            ValueError: If end of file is reached
        """
        token = self.next(frontend)
        if token is None:
            raise ValueError("Unexpected end of file")
        return token

    def next(self, frontend: GlslParser) -> Optional[Any]:
        """
        Get the next token, handling directives and errors.

        Args:
            frontend: The parser frontend

        Returns:
            The next token or None if end of file
        """
        if self.backtracked_token is not None:
            token = self.backtracked_token
            self.backtracked_token = None
            return token
        
        if self.cursor >= len(self.tokens):
            return None
        
        token = self.tokens[self.cursor]
        self.cursor += 1
        self.last_meta = token.meta
        return token

    def peek(self, frontend: GlslParser, offset: int = 0) -> Optional[Any]:
        """
        Peek at a token at an offset from the current cursor.

        Args:
            frontend: The parser frontend
            offset: Offset from current cursor (default 0)

        Returns:
            The token at the given offset or None if eof
        """
        cursor_offset = offset
        if self.backtracked_token is not None:
            if offset == 0:
                return self.backtracked_token
            cursor_offset -= 1
            
        if self.cursor + cursor_offset >= len(self.tokens):
            return None
        
        return self.tokens[self.cursor + cursor_offset]

    def expect(self, frontend: GlslParser, value: Any) -> Any:
        """
        Expect a specific token value.

        Args:
            frontend: The parser frontend
            value: Expected token value

        Returns:
            The token if it matches

        Raises:
            ValueError: If token doesn't match expected value
        """
        token = self.bump(frontend)
        if token.value != value:
            raise ValueError(f"Expected {value}, got {token.value}")
        return token

    def expect_ident(self, frontend: GlslParser) -> tuple[str, Any]:
        """
        Expect an identifier token.

        Args:
            frontend: The parser frontend

        Returns:
            Tuple of (identifier name, metadata)

        Raises:
            ValueError: If next token is not an identifier
        """
        from .token import TokenValue
        token = self.bump(frontend)
        if token.value != TokenValue.IDENTIFIER:
            raise ValueError(f"Expected identifier, got {token.value}")
        return str(token.data), token.meta

    def parse_external_declaration(self, frontend: GlslParser, ctx: Any) -> None:
        """
        Parse a top-level declaration.

        Args:
            frontend: The parser frontend
            ctx: Parsing context
        """
        token = self.peek(frontend)
        if token is None:
            return
        
        from .token import TokenValue
        if token.value == TokenValue.DIRECTIVE:
            self.bump(frontend)
            return
            
        # Dispatch to sub-parsers based on lookahead
        # For now, we'll try to parse as a declaration
        result = frontend.declaration_parser.parse_declaration(ctx, frontend)
        # If result is None, it means it couldn't parse as a declaration
        # We might want to report an error or skip
        if result is None:
             # Skip one token to avoid infinite loop if unsure
             self.bump(frontend)
