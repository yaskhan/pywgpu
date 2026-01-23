"""
WGSL semantic indexing and dependency resolution.

Translated from wgpu-trunk/naga/src/front/wgsl/index.rs

This module performs topological sorting on global declarations,
ensuring that each definition appears before all its uses.
"""

from typing import Any, List, Dict, Set, Optional, Iterator
from dataclasses import dataclass


@dataclass
class ResolvedDependency:
    """
    An edge from a reference to its referent in the dependency graph.
    
    Attributes:
        decl_handle: Handle to the referenced declaration
        usage_span: Location where the reference occurs
    """
    decl_handle: Any
    usage_span: tuple[int, int]


class Index:
    """
    A GlobalDecl list in which each definition occurs before all its uses.
    
    This performs topological sorting on the translation unit's global
    declarations, placing referents before the definitions that refer to them.
    """
    
    def __init__(self, dependency_order: List[Any]):
        """
        Initialize the index with sorted declarations.
        
        Args:
            dependency_order: List of declaration handles in dependency order
        """
        self.dependency_order = dependency_order
    
    @classmethod
    def generate(cls, tu: Any) -> 'Index':
        """
        Generate an Index for the given translation unit.
        
        Performs a topological sort on tu's global declarations, placing
        referents before the definitions that refer to them.
        
        Args:
            tu: Translation unit with declarations
            
        Returns:
            Index with declarations in dependency order
            
        Raises:
            ParseError: If the graph contains cycles
        """
        # Build map from names to declaration handles
        globals: Dict[str, Any] = {}
        
        for handle, decl in enumerate(tu.decls):
            ident = _get_decl_ident(decl)
            if ident is not None:
                name = ident['name']
                if name in globals:
                    # Redefinition error
                    from .error import redefinition_error
                    old_decl = tu.decls[globals[name]]
                    old_ident = _get_decl_ident(old_decl)
                    raise redefinition_error(
                        name,
                        old_ident['span'],
                        ident['span']
                    )
                globals[name] = handle
        
        # Perform depth-first sort
        solver = DependencySolver(
            globals=globals,
            module=tu,
            visited=[False] * len(tu.decls),
            temp_visited=[False] * len(tu.decls),
            path=[],
            out=[]
        )
        
        dependency_order = solver.solve()
        return cls(dependency_order)
    
    def visit_ordered(self) -> Iterator[Any]:
        """
        Iterate over GlobalDecls, visiting each definition before all its uses.
        
        Yields:
            Declaration handles in dependency order
        """
        return iter(self.dependency_order)


class DependencySolver:
    """
    Local state for ordering a TranslationUnit's module-scope declarations.
    
    Performs depth-first search to detect cycles and produce topological ordering.
    """
    
    def __init__(
        self,
        globals: Dict[str, Any],
        module: Any,
        visited: List[bool],
        temp_visited: List[bool],
        path: List[ResolvedDependency],
        out: List[Any]
    ):
        """
        Initialize the dependency solver.
        
        Args:
            globals: Map from names to declaration handles
            module: Translation unit being sorted
            visited: Tracks which declarations have been processed
            temp_visited: Tracks current DFS path for cycle detection
            path: Current path in DFS traversal
            out: Output list of sorted declarations
        """
        self.globals = globals
        self.module = module
        self.visited = visited
        self.temp_visited = temp_visited
        self.path = path
        self.out = out
    
    def solve(self) -> List[Any]:
        """
        Produce the sorted list of declaration handles, and check for cycles.
        
        Returns:
            List of declaration handles in dependency order
            
        Raises:
            ParseError: If cycles are detected
        """
        for id in range(len(self.module.decls)):
            if self.visited[id]:
                continue
            self._dfs(id)
        
        return self.out
    
    def _dfs(self, id: int) -> None:
        """
        Ensure all declarations used by id have been added, then append id.
        
        Args:
            id: Declaration handle to process
            
        Raises:
            ParseError: If a cycle is detected
        """
        decl = self.module.decls[id]
        
        self.temp_visited[id] = True
        
        # Process dependencies
        for dep in getattr(decl, 'dependencies', []):
            dep_name = dep.get('ident')
            if dep_name in self.globals:
                dep_id = self.globals[dep_name]
                
                self.path.append(ResolvedDependency(
                    decl_handle=dep_id,
                    usage_span=dep.get('usage', (0, 0))
                ))
                
                if self.temp_visited[dep_id]:
                    # Found a cycle
                    if dep_id == id:
                        # Direct self-reference
                        from .error import ParseError
                        ident = _get_decl_ident(decl)
                        raise ParseError(
                            message="recursive declaration",
                            labels=[(ident['span'][0], ident['span'][1], "recursive reference")],
                            notes=[]
                        )
                    else:
                        # Indirect cycle through other declarations
                        from .error import ParseError
                        
                        # Find start of cycle in path
                        start_at = 0
                        for i, path_dep in enumerate(reversed(self.path)):
                            if path_dep.decl_handle == dep_id:
                                start_at = len(self.path) - i - 1
                                break
                        
                        # Build cycle path for error message
                        cycle_labels = []
                        for path_dep in self.path[start_at:]:
                            curr_decl = self.module.decls[path_dep.decl_handle]
                            curr_ident = _get_decl_ident(curr_decl)
                            cycle_labels.append((
                                curr_ident['span'][0],
                                curr_ident['span'][1],
                                "used here"
                            ))
                        
                        dep_ident = _get_decl_ident(self.module.decls[dep_id])
                        raise ParseError(
                            message="cyclic declaration dependency",
                            labels=[(dep_ident['span'][0], dep_ident['span'][1], "cycle starts here")] + cycle_labels,
                            notes=[]
                        )
                
                elif not self.visited[dep_id]:
                    self._dfs(dep_id)
                
                # Remove edge from current path
                self.path.pop()
        
        # Remove node from current path
        self.temp_visited[id] = False
        
        # Add to output ordering
        self.out.append(id)
        self.visited[id] = True


def _get_decl_ident(decl: Any) -> Optional[Dict[str, Any]]:
    """
    Extract identifier from a global declaration.
    
    Args:
        decl: Global declaration
        
    Returns:
        Dictionary with 'name' and 'span' keys, or None if no identifier
    """
    kind = getattr(decl, 'kind', None)
    if kind is None:
        return None
    
    # Match on declaration kind
    kind_type = type(kind).__name__
    
    if hasattr(kind, 'name'):
        name_obj = kind.name
        from .ast import Ident
        if isinstance(name_obj, Ident):
            return {
                'name': name_obj.name,
                'span': name_obj.span
            }
        return {
            'name': str(name_obj),
            'span': getattr(kind, 'span', (0, 0))
        }
    
    # ConstAssert has no identifier
    return None
