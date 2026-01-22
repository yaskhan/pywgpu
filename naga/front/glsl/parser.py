from typing import Any, Optional, Dict
from enum import Enum
from .. import Parser
from ...ir import Module

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
        self.errors: list[Any] = []
        # Internal lookup tables
        self.lookup_function: Dict[str, Any] = {}
        self.lookup_type: Dict[str, Any] = {}
        self.global_variables: list[Any] = []
        self.entry_args: list[Any] = []

    def reset(self, stage: ShaderStage) -> None:
        """Reset the parser state for new parsing."""
        self.metadata.reset(stage)
        self.errors.clear()
        self.lookup_function.clear()
        self.lookup_type.clear()
        self.global_variables.clear()
        self.entry_args.clear()

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

        # Store options
        self.options = parse_options

        # Create the module
        module = Module()

        # TODO: Implement full GLSL parsing pipeline
        # The complete implementation requires:
        # 1. Lexical analysis - Create lexer from source and defines
        #    lexer = Lexer(source, parse_options.defines)
        # 2. Create parsing context with the lexer
        #    ctx = ParsingContext(lexer)
        # 3. Parse external declarations in a loop
        #    while ctx.peek() is not None:
        #        ctx.parse_external_declaration(self, module_ctx)
        # 4. Find and add entry point (main function)
        #    - Look for 'main' function in lookup_function
        #    - Verify it has no parameters and is defined
        #    - Add it as an entry point to the module
        # 5. Handle preprocessing directives:
        #    - #version: Parse version number and profile
        #    - #extension: Track required/enabled extensions
        #    - #pragma: Handle pragma directives
        # 6. Error handling and collection
        #    - Collect errors during parsing
        #    - Return errors if parsing fails

        # Check for errors
        if self.errors:
            error_messages = [str(e) for e in self.errors]
            raise ValueError(f"GLSL parsing failed with {len(self.errors)} error(s): {error_messages}")

        return module

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
        pass

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
        # TODO: Implement global variable addition
        # This should handle:
        # - Global variables
        # - Constants
        # - Overrides
        pass

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
        pass

    def get_metadata(self) -> ShaderMetadata:
        """
        Get metadata about the parsed shader.

        Returns:
            ShaderMetadata containing version, profile, extensions, etc.
        """
        return self.metadata


class ParsingContext:
    """
    Context for parsing GLSL source code.
    
    Manages token stream, backtracking, and parsing state.
    """

    def __init__(self, lexer: Any):
        """
        Initialize parsing context.

        Args:
            lexer: The lexer providing tokens
        """
        self.lexer = lexer
        self.backtracked_token: Optional[Any] = None
        self.last_meta: Any = None

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
        # TODO: Implement token retrieval with directive handling
        # This should:
        # 1. Check for backtracked token first
        # 2. Get next token from lexer
        # 3. Handle directives by calling frontend.handle_directive
        # 4. Handle lexer errors by adding to frontend.errors
        # 5. Return token or None
        return None

    def peek(self, frontend: GlslParser) -> Optional[Any]:
        """
        Peek at the next token without consuming it.

        Args:
            frontend: The parser frontend

        Returns:
            The next token or None if end of file
        """
        # TODO: Implement token peeking
        return None

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
        token = self.bump(frontend)
        # TODO: Check if token is identifier and return (name, meta)
        raise NotImplementedError("expect_ident not fully implemented")

    def parse_external_declaration(self, frontend: GlslParser, ctx: Any) -> None:
        """
        Parse a top-level declaration.

        Args:
            frontend: The parser frontend
            ctx: Parsing context
        """
        # TODO: Implement external declaration parsing
        # This should handle:
        # - Function declarations/definitions
        # - Global variable declarations
        # - Struct definitions
        # - Type definitions
        pass
