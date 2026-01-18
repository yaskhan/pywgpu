from typing import Any, Protocol, TypeVar, Generic, Optional
from ..ir import Module

T = TypeVar('T')

class Parser(Protocol):
    """
    Base trait for all parsers.
    """
    def parse(self, source: Any) -> Module:
        ...

class Typifier:
    """
    A table of types for an Arena[Expression].
    
    A front end can use a Typifier to get types for an arena's expressions
    while it is still contributing expressions to it. At any point, you can call
    typifier.grow(expr, arena, ctx), where expr is a Handle[Expression]
    referring to something in arena, and the Typifier will resolve the types
    of all the expressions up to and including expr.
    """
    
    def __init__(self) -> None:
        self.resolutions: dict[str, Any] = {}  # Simplified representation
    
    def reset(self) -> None:
        """Reset the typifier state."""
        self.resolutions.clear()
    
    def get(self, expr_handle: str, types: Any) -> Any:
        """Get the type for an expression handle."""
        resolution = self.resolutions.get(expr_handle)
        if resolution is None:
            raise KeyError(f"No type resolution for {expr_handle}")
        return resolution

class SymbolTable(Generic[T]):
    """
    Structure responsible for managing variable lookups and keeping track of
    lexical scopes.
    """
    
    def __init__(self) -> None:
        self.scopes: list[dict[str, T]] = [{}]
        self.cursor: int = 1
    
    def push_scope(self) -> None:
        """Add a new lexical scope."""
        if len(self.scopes) == self.cursor:
            self.scopes.append({})
        else:
            self.scopes[self.cursor].clear()
        self.cursor += 1
    
    def pop_scope(self) -> None:
        """Remove the current lexical scope."""
        if self.cursor == 1:
            raise RuntimeError("Tried to pop the root scope")
        self.cursor -= 1
    
    def lookup(self, name: str) -> Optional[T]:
        """Look up a variable by name in the current scope chain."""
        for scope in reversed(self.scopes[:self.cursor]):
            if name in scope:
                return scope[name]
        return None
    
    def add(self, name: str, var: T) -> Optional[T]:
        """Add a variable to the current scope."""
        current_scope = self.scopes[self.cursor - 1]
        return current_scope.get(name)
    
    def add_root(self, name: str, var: T) -> Optional[T]:
        """Add a variable to the root scope."""
        return self.scopes[0].get(name)

# Additional utility classes and functions for all frontends
class ParseErrors(Exception):
    """Collection of parsing errors."""
    
    def __init__(self, errors: list[Any], message: str = "Parsing failed with multiple errors"):
        super().__init__(message)
        self.errors = errors
        self.message = message

def get_shader_stage_from_source(source: str, default_stage: str = "vertex") -> str:
    """
    Attempt to determine shader stage from source code patterns.
    
    Args:
        source: Shader source code
        default_stage: Default stage if cannot be determined
        
    Returns:
        Detected shader stage
    """
    source_lower = source.lower()
    
    if "fragment" in source_lower or "@fragment" in source:
        return "fragment"
    elif "vertex" in source_lower or "@vertex" in source:
        return "vertex"
    elif "compute" in source_lower or "@compute" in source or "workgroup_size" in source:
        return "compute"
    elif "void main()" in source:
        # For GLSL, this is ambiguous but we'll default to vertex
        return default_stage
    
    return default_stage

__all__ = [
    "Parser",
    "Typifier", 
    "SymbolTable",
    "ParseErrors",
    "get_shader_stage_from_source"
]
