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
    def from_stage(cls, stage: ShaderStage) -> 'Options':
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
        
        # TODO: Implement actual GLSL parsing logic
        # This involves:
        # 1. Lexical analysis (tokenization)
        # 2. Preprocessing (handle defines)
        # 3. Parsing (build AST)
        # 4. Semantic analysis (type checking, validation)
        # 5. IR generation (convert to NAGA IR)
        
        # For now, return empty module
        return module
    
    def get_metadata(self) -> ShaderMetadata:
        """
        Get metadata about the parsed shader.
        
        Returns:
            ShaderMetadata containing version, profile, extensions, etc.
        """
        return self.metadata
