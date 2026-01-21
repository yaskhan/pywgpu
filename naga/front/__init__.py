"""
Frontend parsers that consume binary and text shaders and load them into Modules.
"""

from typing import Any, Protocol, TypeVar, Generic, Optional, Union
import logging

from ..arena import Handle, HandleVec, Arena, UniqueArena
from ..proc import ResolveContext, ResolveError, TypeResolution
from ..ir import Module, Expression, Type, TypeInner

log = logging.getLogger(__name__)

T = TypeVar("T")
Name = TypeVar("Name")
Var = TypeVar("Var")


class Parser(Protocol):
    """
    Base trait for all parsers.
    """

    def parse(self, source: Any) -> Module: ...


class Typifier:
    """
    A table of types for an Arena[Expression].
    
    A front end can use a Typifier to get types for an arena's expressions
    while it is still contributing expressions to it. At any point, you can call
    typifier.grow(expr, arena, ctx), where expr is a Handle[Expression]
    referring to something in arena, and the Typifier will resolve the types
    of all the expressions up to and including expr. Then you can write
    typifier[handle] to get the type of any handle at or before expr.
    
    Note that Typifier does *not* build an Arena[Type] as a part of its
    usual operation. Ideally, a module's type arena should only contain types
    actually needed by Handle[Type]s elsewhere in the module — functions,
    variables, Compose expressions, other types, and so on — so we don't
    want every little thing that occurs as the type of some intermediate
    expression to show up there.
    
    Instead, Typifier accumulates a TypeResolution for each expression,
    which refers to the Arena[Type] in the ResolveContext passed to grow
    as needed. TypeResolution is a lightweight representation for
    intermediate types like this; see its documentation for details.
    
    If you do need to register a Typifier's conclusion in an Arena[Type]
    (say, for a LocalVariable whose type you've inferred), you can use
    register_type to do so.
    """

    def __init__(self) -> None:
        """Initialize a new Typifier."""
        self.resolutions: HandleVec[Expression, TypeResolution] = HandleVec()

    def reset(self) -> None:
        """Reset the typifier state."""
        self.resolutions.clear()

    def get(self, expr_handle: Handle[Expression], types: UniqueArena[Type]) -> TypeInner:
        """
        Get the type inner for an expression handle.
        
        Args:
            expr_handle: Handle to the expression
            types: The unique arena of types
            
        Returns:
            The type inner for the expression
        """
        return self.resolutions[expr_handle].inner_with(types)

    def register_type(
        self,
        expr_handle: Handle[Expression],
        types: UniqueArena[Type],
    ) -> Handle[Type]:
        """
        Add an expression's type to an Arena[Type].
        
        Add the type of expr_handle to types, and return a Handle[Type]
        referring to it.
        
        Note:
            If you just need a TypeInner for expr_handle's type, consider
            using typifier[expression].inner_with(types) instead. Calling
            TypeResolution.inner_with often lets us avoid adding anything to
            the arena, which can significantly reduce the number of types that end
            up in the final module.
        
        Args:
            expr_handle: Handle to the expression
            types: The unique arena of types
            
        Returns:
            Handle to the registered type
        """
        resolution = self[expr_handle]
        if isinstance(resolution, Handle):
            return resolution
        else:
            # TypeResolution.Value case
            return types.insert(Type(name=None, inner=resolution), span=None)

    def grow(
        self,
        expr_handle: Handle[Expression],
        expressions: Arena[Expression],
        ctx: ResolveContext,
    ) -> None:
        """
        Grow this typifier until it contains a type for expr_handle.
        
        Args:
            expr_handle: Handle to the expression to resolve
            expressions: Arena of expressions
            ctx: Resolve context
            
        Raises:
            ResolveError: If resolution fails
        """
        if len(self.resolutions) <= expr_handle.index():
            for eh, expr in expressions.iter_from(len(self.resolutions)):
                resolution = ctx.resolve(expr, lambda h: self.resolutions[h])
                log.debug(f"Resolving {eh!r} = {expr!r} : {resolution!r}")
                self.resolutions.insert(eh, resolution)

    def invalidate(
        self,
        expr_handle: Handle[Expression],
        expressions: Arena[Expression],
        ctx: ResolveContext,
    ) -> None:
        """
        Recompute the type resolution for expr_handle.
        
        If the type of expr_handle hasn't yet been calculated, call
        grow to ensure it is covered.
        
        In either case, when this returns, self[expr_handle] should be an
        updated type resolution for expr_handle.
        
        Args:
            expr_handle: Handle to the expression to invalidate
            expressions: Arena of expressions
            ctx: Resolve context
            
        Raises:
            ResolveError: If resolution fails
        """
        if len(self.resolutions) <= expr_handle.index():
            self.grow(expr_handle, expressions, ctx)
        else:
            expr = expressions[expr_handle]
            resolution = ctx.resolve(expr, lambda h: self.resolutions[h])
            self.resolutions[expr_handle] = resolution

    def __getitem__(self, handle: Handle[Expression]) -> TypeResolution:
        """Get the type resolution for an expression handle."""
        return self.resolutions[handle]


class SymbolTable(Generic[Name, Var]):
    """
    Structure responsible for managing variable lookups and keeping track of
    lexical scopes.
    
    The symbol table is generic over the variable representation and its name
    to allow larger flexibility on the frontends on how they might represent them.
    
    Example:
        >>> # Create a new symbol table with u32s representing the variable
        >>> symbol_table = SymbolTable()
        >>> 
        >>> # Add two variables named 'var1' and 'var2' with 0 and 2 respectively
        >>> symbol_table.add("var1", 0)
        >>> symbol_table.add("var2", 2)
        >>> 
        >>> # Check that 'var1' exists and is 0
        >>> assert symbol_table.lookup("var1") == 0
        >>> 
        >>> # Push a new scope and add a variable to it named 'var1' shadowing the
        >>> # variable of our previous scope
        >>> symbol_table.push_scope()
        >>> symbol_table.add("var1", 1)
        >>> 
        >>> # Check that 'var1' now points to the new value of 1 and 'var2' still
        >>> # exists with its value of 2
        >>> assert symbol_table.lookup("var1") == 1
        >>> assert symbol_table.lookup("var2") == 2
        >>> 
        >>> # Pop the scope
        >>> symbol_table.pop_scope()
        >>> 
        >>> # Check that 'var1' now refers to our initial variable with value 0
        >>> assert symbol_table.lookup("var1") == 0
    
    Scopes are ordered as a LIFO stack so a variable defined in a later scope
    with the same name as another variable defined in a earlier scope will take
    precedence in the lookup. Scopes can be added with push_scope and
    removed with pop_scope.
    
    A root scope is added when the symbol table is created and must always be
    present. Trying to pop it will result in a panic.
    
    Variables can be added with add and looked up with lookup. Adding a
    variable will do so in the currently active scope and as mentioned
    previously a lookup will search from the current scope to the root scope.
    
    Attributes:
        scopes: Stack of lexical scopes. Not all scopes are active; see cursor.
        cursor: Limit of the scopes stack (exclusive). By using a separate value for
            the stack length instead of list's own internal length, the scopes can
            be reused to cache memory allocations.
    """

    def __init__(self) -> None:
        """Constructs a new symbol table with a root scope."""
        self.scopes: list[dict[Name, Var]] = [{}]
        self.cursor: int = 1

    def push_scope(self) -> None:
        """
        Adds a new lexical scope.
        
        All variables declared after this point will be added to this scope
        until another scope is pushed or pop_scope is called, causing this
        scope to be removed along with all variables added to it.
        """
        # If the cursor is equal to the scope's stack length then we need to
        # push another empty scope. Otherwise we can reuse the already existing
        # scope.
        if len(self.scopes) == self.cursor:
            self.scopes.append({})
        else:
            self.scopes[self.cursor].clear()
        self.cursor += 1

    def pop_scope(self) -> None:
        """
        Removes the current lexical scope and all its variables.
        
        Raises:
            AssertionError: If the current lexical scope is the root scope
        """
        # Despite the method title, the variables are only deleted when the
        # scope is reused. This is because while a clear is inevitable if the
        # scope needs to be reused, there are cases where the scope might be
        # popped and not reused, i.e. if another scope with the same nesting
        # level is never pushed again.
        assert self.cursor != 1, "Tried to pop the root scope"
        self.cursor -= 1

    def lookup(self, name: Name) -> Optional[Var]:
        """
        Perform a lookup for a variable named name.
        
        As stated in the class level documentation the lookup will proceed from
        the current scope to the root scope, returning Some when a variable is
        found or None if there doesn't exist a variable with name in any
        scope.
        
        Args:
            name: The name to look up
            
        Returns:
            The variable if found, None otherwise
        """
        # Iterate backwards through the scopes and try to find the variable
        for scope in reversed(self.scopes[: self.cursor]):
            if name in scope:
                return scope[name]
        return None

    def add(self, name: Name, var: Var) -> Optional[Var]:
        """
        Adds a new variable to the current scope.
        
        Returns the previous variable with the same name in this scope if it
        exists, so that the frontend might handle it in case variable shadowing
        is disallowed.
        
        Args:
            name: The name of the variable
            var: The variable
            
        Returns:
            The previous variable with the same name if it exists
        """
        current_scope = self.scopes[self.cursor - 1]
        old_var = current_scope.get(name)
        current_scope[name] = var
        return old_var

    def add_root(self, name: Name, var: Var) -> Optional[Var]:
        """
        Adds a new variable to the root scope.
        
        This is used in GLSL for builtins which aren't known in advance and only
        when used for the first time, so there must be a way to add those
        declarations to the root unconditionally from the current scope.
        
        Returns the previous variable with the same name in the root scope if it
        exists, so that the frontend might handle it in case variable shadowing
        is disallowed.
        
        Args:
            name: The name of the variable
            var: The variable
            
        Returns:
            The previous variable with the same name if it exists
        """
        old_var = self.scopes[0].get(name)
        self.scopes[0][name] = var
        return old_var


# Additional utility classes and functions for all frontends
class ParseErrors(Exception):
    """Collection of parsing errors."""

    def __init__(
        self, errors: list[Any], message: str = "Parsing failed with multiple errors"
    ):
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
    elif (
        "compute" in source_lower or "@compute" in source or "workgroup_size" in source
    ):
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
    "get_shader_stage_from_source",
    "ResolveContext",
    "ResolveError",
    "TypeResolution",
]
