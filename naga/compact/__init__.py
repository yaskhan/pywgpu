"""
Module compaction logic for removing unused items from a module.

This module provides the `compact` function for removing unused objects from a module,
along with supporting types and logic for tracing usage and adjusting handles.
"""

from typing import Any, Optional
import logging

from .handle_set_map import Handle, HandleSet, HandleMap
from .functions import FunctionCompactor, FunctionTracer
from .expressions import ExpressionCompactor, ExpressionTracer
from .statements import StatementCompactor, StatementTracer
from .types import TypeCompactor, TypeTracer

log = logging.getLogger(__name__)


class KeepUnused:
    """
    Configuration option for `compact`. See `compact` for details.
    
    Attributes:
        NO: Remove unused items
        YES: Keep unused items
    """

    NO = False
    YES = True


def compact(module: Any, keep_unused: bool = False) -> None:
    """
    Remove most unused objects from `module`, which must be valid.
    
    Always removes the following unused objects:
    - anonymous types, overrides, and constants
    - abstract-typed constants
    - expressions
    
    If `keep_unused` is `True`, the following are never considered unused,
    otherwise, they will also be removed if unused:
    - functions
    - global variables
    - named types and overrides
    
    The following are never removed:
    - named constants with a concrete type
    - special types
    - entry points
    - within an entry point or a used function:
        - arguments
        - local variables
        - named expressions
    
    After removing items according to the rules above, all handles in the
    remaining objects are adjusted as necessary. When `keep_unused` is `True`, the
    resulting module should have all the named objects (except abstract-typed
    constants) present in the original, and those objects should be functionally
    identical. When `keep_unused` is `False`, the resulting module should have the
    entry points present in the original, and those entry points should be
    functionally identical.
    
    Args:
        module: The module to compact (will be modified in place)
        keep_unused: If True, keep unused functions, global variables, and named types
        
    Raises:
        Panic if module would not pass validation
    """
    log.trace("creating module tracer")
    module_tracer = ModuleTracer(module)

    # Observe what each entry point actually uses
    log.trace("tracing entry points")
    entry_point_maps = []
    for entry_point in module.entry_points:
        log.trace(f"tracing entry point {entry_point.function.name!r}")

        if hasattr(entry_point, "workgroup_size_overrides") and entry_point.workgroup_size_overrides:
            for size in entry_point.workgroup_size_overrides:
                if size is not None:
                    module_tracer.global_expressions_used.insert(size)

        if hasattr(entry_point, "task_payload") and entry_point.task_payload is not None:
            module_tracer.global_variables_used.insert(entry_point.task_payload)

        if hasattr(entry_point, "mesh_info") and entry_point.mesh_info is not None:
            mesh_info = entry_point.mesh_info
            module_tracer.global_variables_used.insert(mesh_info.output_variable)
            module_tracer.types_used.insert(mesh_info.vertex_output_type)
            module_tracer.types_used.insert(mesh_info.primitive_output_type)
            if mesh_info.max_vertices_override is not None:
                module_tracer.global_expressions_used.insert(mesh_info.max_vertices_override)
            if mesh_info.max_primitives_override is not None:
                module_tracer.global_expressions_used.insert(mesh_info.max_primitives_override)

        if entry_point.stage == "Task" or entry_point.stage == "Mesh":
            # u32 should always be there if the module is valid
            u32_type = None
            for handle, ty in module.types.items():
                if hasattr(ty.inner, "scalar") and ty.inner.scalar.kind == "Uint" and ty.inner.scalar.width == 4:
                    u32_type = handle
                    break
            if u32_type is not None:
                module_tracer.types_used.insert(u32_type)

        used = module_tracer.as_function(entry_point.function)
        used.trace()
        entry_point_maps.append(FunctionMap.from_tracer(used))

    # Trace functions
    log.trace("tracing functions")
    function_maps = HandleMap()
    if keep_unused:
        module_tracer.functions_used.add_all()
        module_tracer.functions_pending.add_all()

    while True:
        handle = module_tracer.functions_pending.pop()
        if handle is None:
            break
        function = module.functions[handle]
        log.trace(f"tracing function {function!r}")
        function_tracer = module_tracer.as_function(function)
        function_tracer.trace()
        function_maps.insert(handle, FunctionMap.from_tracer(function_tracer))

    # Trace special types
    log.trace("tracing special types")
    module_tracer.trace_special_types(module.special_types)

    # Trace global variables
    log.trace("tracing global variables")
    if keep_unused:
        module_tracer.global_variables_used.add_all()
    for global_handle in module_tracer.global_variables_used.iter():
        log.trace(f"tracing global {module.global_variables[global_handle].name!r}")
        module_tracer.types_used.insert(module.global_variables[global_handle].ty)
        if hasattr(module.global_variables[global_handle], "init") and module.global_variables[global_handle].init is not None:
            module_tracer.global_expressions_used.insert(module.global_variables[global_handle].init)

    # Trace named constants
    log.trace("tracing named constants")
    for handle, constant in module.constants.items():
        if constant.name is None:
            continue
        # Skip abstract-typed constants
        ty = module.types[constant.ty]
        if hasattr(ty.inner, "is_abstract") and ty.inner.is_abstract(module.types):
            continue

        log.trace(f"tracing constant {constant.name!r}")
        module_tracer.constants_used.insert(handle)
        module_tracer.types_used.insert(constant.ty)
        module_tracer.global_expressions_used.insert(constant.init)

    if keep_unused:
        # Treat all named overrides as used
        for handle, override in module.overrides.items():
            if override.name is not None and module_tracer.overrides_used.insert(handle):
                module_tracer.types_used.insert(override.ty)
                if override.init is not None:
                    module_tracer.global_expressions_used.insert(override.init)

        # Treat all named types as used
        for handle, ty in module.types.items():
            if ty.name is not None:
                module_tracer.types_used.insert(handle)

    # Trace types and expressions in tandem
    module_tracer.type_expression_tandem()

    # Create module map from tracer
    module_map = ModuleMap.from_tracer(module_tracer)

    # Compact types
    log.trace("compacting types")
    new_types = []  # UniqueArena equivalent
    for old_handle, ty, span in module.types.drain_all():
        expected_new_handle = module_map.types.try_adjust(old_handle)
        if expected_new_handle is not None:
            module_map.adjust_type(ty)
            actual_new_handle = len(new_types)
            new_types.append((ty, span))
            assert actual_new_handle == expected_new_handle
    module.types = new_types

    log.trace("adjusting special types")
    module_map.adjust_special_types(module.special_types)

    # Compact constant expressions
    log.trace("adjusting constant expressions")
    module.global_expressions.retain_mut(
        lambda handle, expr: _retain_expression(handle, expr, module_map)
    )

    # Compact constants
    log.trace("adjusting constants")
    module.constants.retain_mut(
        lambda handle, constant: _retain_constant(handle, constant, module_map)
    )

    # Compact overrides
    log.trace("adjusting overrides")
    module.overrides.retain_mut(
        lambda handle, override: _retain_override(handle, override, module_map)
    )

    # Adjust workgroup_size_overrides
    log.trace("adjusting workgroup_size_overrides")
    for entry_point in module.entry_points:
        if hasattr(entry_point, "workgroup_size_overrides") and entry_point.workgroup_size_overrides is not None:
            for i, size in enumerate(entry_point.workgroup_size_overrides):
                if size is not None:
                    module_map.global_expressions.adjust(size)
                    entry_point.workgroup_size_overrides[i] = size

    # Compact global variables
    log.trace("adjusting global variables")
    module.global_variables.retain_mut(
        lambda handle, global_var: _retain_global_variable(handle, global_var, module_map)
    )

    # Adjust doc comments
    if hasattr(module, "doc_comments") and module.doc_comments is not None:
        module_map.adjust_doc_comments(module.doc_comments)

    # Reused storage for named expressions
    reused_named_expressions = {}

    # Compact functions
    for handle, function in module.functions.items():
        function_map = function_maps.get(handle)
        if function_map is not None:
            log.trace(f"retaining and compacting function {function.name!r}")
            function_map.compact(function, module_map, reused_named_expressions)
        else:
            log.trace(f"dropping function {function.name!r}")
            module.functions.remove(handle)

    # Compact entry points
    for entry_point, func_map in zip(module.entry_points, entry_point_maps):
        log.trace(f"compacting entry point {entry_point.function.name!r}")
        func_map.compact(entry_point.function, module_map, reused_named_expressions)
        if hasattr(entry_point, "task_payload") and entry_point.task_payload is not None:
            module_map.globals.adjust(entry_point.task_payload)
        if hasattr(entry_point, "mesh_info") and entry_point.mesh_info is not None:
            mesh_info = entry_point.mesh_info
            module_map.globals.adjust(mesh_info.output_variable)
            module_map.types.adjust(mesh_info.vertex_output_type)
            module_map.types.adjust(mesh_info.primitive_output_type)
            if mesh_info.max_vertices_override is not None:
                module_map.global_expressions.adjust(mesh_info.max_vertices_override)
            if mesh_info.max_primitives_override is not None:
                module_map.global_expressions.adjust(mesh_info.max_primitives_override)


def _retain_expression(handle: Handle, expr: Any, module_map: "ModuleMap") -> bool:
    """Helper to retain and adjust expression."""
    if module_map.global_expressions.used(handle):
        module_map.adjust_expression(expr, module_map.global_expressions)
        return True
    return False


def _retain_constant(handle: Handle, constant: Any, module_map: "ModuleMap") -> bool:
    """Helper to retain and adjust constant."""
    if module_map.constants.used(handle):
        module_map.types.adjust(constant.ty)
        module_map.global_expressions.adjust(constant.init)
        return True
    return False


def _retain_override(handle: Handle, override: Any, module_map: "ModuleMap") -> bool:
    """Helper to retain and adjust override."""
    if module_map.overrides.used(handle):
        module_map.types.adjust(override.ty)
        if override.init is not None:
            module_map.global_expressions.adjust(override.init)
        return True
    return False


def _retain_global_variable(handle: Handle, global_var: Any, module_map: "ModuleMap") -> bool:
    """Helper to retain and adjust global variable."""
    if module_map.globals.used(handle):
        log.trace(f"retaining global variable {global_var.name!r}")
        module_map.types.adjust(global_var.ty)
        if hasattr(global_var, "init") and global_var.init is not None:
            module_map.global_expressions.adjust(global_var.init)
        return True
    else:
        log.trace(f"dropping global variable {global_var.name!r}")
        return False


class ModuleTracer:
    """
    Traces module usage to determine which items are used.
    
    Attributes:
        module: The module being traced
        functions_pending: Subset of functions_used that have not yet been traced
        functions_used: Used functions
        types_used: Used types
        global_variables_used: Used global variables
        constants_used: Used constants
        overrides_used: Used overrides
        global_expressions_used: Used global expressions
    """

    def __init__(self, module: Any) -> None:
        """
        Initialize a new module tracer.
        
        Args:
            module: The module to trace
        """
        self.module = module
        self.functions_pending = HandleSet.for_arena(module.functions)
        self.functions_used = HandleSet.for_arena(module.functions)
        self.types_used = HandleSet.for_arena(module.types)
        self.global_variables_used = HandleSet.for_arena(module.global_variables)
        self.constants_used = HandleSet.for_arena(module.constants)
        self.overrides_used = HandleSet.for_arena(module.overrides)
        self.global_expressions_used = HandleSet.for_arena(module.global_expressions)

    def trace_special_types(self, special_types: Any) -> None:
        """
        Trace special types.
        
        Args:
            special_types: The special types object
        """
        if hasattr(special_types, "ray_desc") and special_types.ray_desc is not None:
            self.types_used.insert(special_types.ray_desc)
        if hasattr(special_types, "ray_intersection") and special_types.ray_intersection is not None:
            self.types_used.insert(special_types.ray_intersection)
        if hasattr(special_types, "ray_vertex_return") and special_types.ray_vertex_return is not None:
            self.types_used.insert(special_types.ray_vertex_return)
        if hasattr(special_types, "external_texture_params") and special_types.external_texture_params is not None:
            self.types_used.insert(special_types.external_texture_params)
        if hasattr(special_types, "external_texture_transfer_function") and special_types.external_texture_transfer_function is not None:
            self.types_used.insert(special_types.external_texture_transfer_function)
        if hasattr(special_types, "predeclared_types"):
            for handle in special_types.predeclared_types.values():
                self.types_used.insert(handle)

    def type_expression_tandem(self) -> None:
        """
        Traverse types and global expressions in tandem to determine which are used.
        
        Assuming that all types and global expressions used by other parts of
        the module have been added to `types_used` and `global_expressions_used`,
        expand those sets to include all types and global expressions reachable
        from those.
        """
        # For each type T, compute the latest global expression E that T and
        # its predecessors refer to
        max_dep = []
        previous = None
        for handle, ty in self.module.types.items():
            if hasattr(ty.inner, "array") or hasattr(ty.inner, "binding_array"):
                # Check array size
                size = getattr(ty.inner, "size", None)
                if size is not None and hasattr(size, "pending"):
                    override_handle = size.pending
                    override = self.module.overrides[override_handle]
                    previous = max(previous, override.init) if previous is not None else override.init
            max_dep.append(previous)

        # Visit types and global expressions from youngest to oldest
        exprs = list(reversed(list(self.module.global_expressions.items())))
        expr_iter = iter(exprs)

        for (ty_handle, ty), dep in zip(
            reversed(list(self.module.types.items())),
            reversed(max_dep)
        ):
            # Visit expressions that could refer to this type
            while True:
                try:
                    expr_pair = next(expr_iter)
                    expr_handle, expr = expr_pair
                    if dep is None or expr_handle > dep:
                        if self.global_expressions_used.contains(expr_handle):
                            self.as_const_expression().trace_expression(expr)
                    else:
                        # Put it back, we're done with expressions for this type
                        expr_iter = iter([expr_pair] + list(expr_iter))
                        break
                except StopIteration:
                    break

            # Trace the type if used
            if self.types_used.contains(ty_handle):
                self.as_type().trace_type(ty)

        # Visit any remaining expressions
        for expr_handle, expr in expr_iter:
            if self.global_expressions_used.contains(expr_handle):
                self.as_const_expression().trace_expression(expr)

    def as_type(self) -> TypeTracer:
        """Get a type tracer."""
        return TypeTracer(
            overrides=self.module.overrides,
            types_used=self.types_used,
            expressions_used=self.global_expressions_used,
            overrides_used=self.overrides_used,
        )

    def as_const_expression(self) -> ExpressionTracer:
        """Get an expression tracer for constant expressions."""
        return ExpressionTracer(
            constants=self.module.constants,
            overrides=self.module.overrides,
            expressions=self.module.global_expressions,
            types_used=self.types_used,
            global_variables_used=self.global_variables_used,
            constants_used=self.constants_used,
            overrides_used=self.overrides_used,
            expressions_used=self.global_expressions_used,
            global_expressions_used=None,
        )

    def as_function(self, function: Any) -> FunctionTracer:
        """Get a function tracer."""
        return FunctionTracer(
            function=function,
            constants=self.module.constants,
            overrides=self.module.overrides,
            functions_pending=self.functions_pending,
            functions_used=self.functions_used,
            types_used=self.types_used,
            global_variables_used=self.global_variables_used,
            constants_used=self.constants_used,
            overrides_used=self.overrides_used,
            global_expressions_used=self.global_expressions_used,
        )


class ModuleMap:
    """
    Maps old handles to new handles for a module.
    
    Attributes:
        functions: Function handle map
        types: Type handle map
        globals: Global variable handle map
        constants: Constant handle map
        overrides: Override handle map
        global_expressions: Global expression handle map
    """

    def __init__(
        self,
        functions: HandleMap,
        types: HandleMap,
        globals: HandleMap,
        constants: HandleMap,
        overrides: HandleMap,
        global_expressions: HandleMap,
    ) -> None:
        """
        Initialize a new module map.
        
        Args:
            functions: Function handle map
            types: Type handle map
            globals: Global variable handle map
            constants: Constant handle map
            overrides: Override handle map
            global_expressions: Global expression handle map
        """
        self.functions = functions
        self.types = types
        self.globals = globals
        self.constants = constants
        self.overrides = overrides
        self.global_expressions = global_expressions

    @classmethod
    def from_tracer(cls, tracer: ModuleTracer) -> "ModuleMap":
        """
        Create a module map from a module tracer.
        
        Args:
            tracer: The module tracer
            
        Returns:
            The created module map
        """
        return cls(
            functions=HandleMap.from_set(tracer.functions_used),
            types=HandleMap.from_set(tracer.types_used),
            globals=HandleMap.from_set(tracer.global_variables_used),
            constants=HandleMap.from_set(tracer.constants_used),
            overrides=HandleMap.from_set(tracer.overrides_used),
            global_expressions=HandleMap.from_set(tracer.global_expressions_used),
        )

    def adjust_special_types(self, special_types: Any) -> None:
        """
        Adjust special types.
        
        Args:
            special_types: The special types object to adjust
        """
        if hasattr(special_types, "ray_desc") and special_types.ray_desc is not None:
            self.types.adjust(special_types.ray_desc)
        if hasattr(special_types, "ray_intersection") and special_types.ray_intersection is not None:
            self.types.adjust(special_types.ray_intersection)
        if hasattr(special_types, "ray_vertex_return") and special_types.ray_vertex_return is not None:
            self.types.adjust(special_types.ray_vertex_return)
        if hasattr(special_types, "external_texture_params") and special_types.external_texture_params is not None:
            self.types.adjust(special_types.external_texture_params)
        if hasattr(special_types, "external_texture_transfer_function") and special_types.external_texture_transfer_function is not None:
            self.types.adjust(special_types.external_texture_transfer_function)
        if hasattr(special_types, "predeclared_types"):
            for key in special_types.predeclared_types:
                self.types.adjust(special_types.predeclared_types[key])

    def adjust_doc_comments(self, doc_comments: Any) -> None:
        """
        Adjust doc comments.
        
        Args:
            doc_comments: The doc comments object to adjust
        """
        # Adjust doc comments for types
        log.trace("adjusting doc comments for types")
        if hasattr(doc_comments, "types"):
            new_doc_types = {}
            for ty, doc_comment in doc_comments.types.items():
                if self.types.used(ty):
                    self.types.adjust(ty)
                    new_doc_types[ty] = doc_comment
            doc_comments.types = new_doc_types

        # Adjust doc comments for struct members
        log.trace("adjusting doc comments for struct members")
        if hasattr(doc_comments, "struct_members"):
            new_doc_struct_members = {}
            for (ty, index), doc_comment in doc_comments.struct_members.items():
                if self.types.used(ty):
                    self.types.adjust(ty)
                    new_doc_struct_members[(ty, index)] = doc_comment
            doc_comments.struct_members = new_doc_struct_members

        # Adjust doc comments for functions
        log.trace("adjusting doc comments for functions")
        if hasattr(doc_comments, "functions"):
            new_doc_functions = {}
            for handle, doc_comment in doc_comments.functions.items():
                if self.functions.used(handle):
                    self.functions.adjust(handle)
                    new_doc_functions[handle] = doc_comment
            doc_comments.functions = new_doc_functions

        # Adjust doc comments for constants
        log.trace("adjusting doc comments for constants")
        if hasattr(doc_comments, "constants"):
            new_doc_constants = {}
            for constant, doc_comment in doc_comments.constants.items():
                if self.constants.used(constant):
                    self.constants.adjust(constant)
                    new_doc_constants[constant] = doc_comment
            doc_comments.constants = new_doc_constants

        # Adjust doc comments for globals
        log.trace("adjusting doc comments for globals")
        if hasattr(doc_comments, "global_variables"):
            new_doc_globals = {}
            for handle, doc_comment in doc_comments.global_variables.items():
                if self.globals.used(handle):
                    self.globals.adjust(handle)
                    new_doc_globals[handle] = doc_comment
            doc_comments.global_variables = new_doc_globals

    def adjust_type(self, ty: Any) -> None:
        """
        Adjust all handles in a type.
        
        Args:
            ty: The type to adjust
        """
        # This is implemented in types.py TypeCompactor
        # For now, placeholder
        pass

    def adjust_expression(self, expr: Any, operand_map: HandleMap) -> None:
        """
        Fix up all handles in an expression.
        
        Use the expression handle remappings in `operand_map`, and all
        other mappings from `self`.
        
        Args:
            expr: The expression to adjust
            operand_map: The handle map for expression operands
        """
        # This is implemented in expressions.py ExpressionCompactor
        # For now, placeholder
        pass


class FunctionMap:
    """
    Maps old handles to new handles for a function.
    
    Attributes:
        expressions: Expression handle map
    """

    def __init__(self, expressions: HandleMap) -> None:
        """
        Initialize a new function map.
        
        Args:
            expressions: Expression handle map
        """
        self.expressions = expressions

    @classmethod
    def from_tracer(cls, tracer: FunctionTracer) -> "FunctionMap":
        """
        Create a function map from a function tracer.
        
        Args:
            tracer: The function tracer
            
        Returns:
            The created function map
        """
        return cls(expressions=HandleMap.from_set(tracer.expressions_used))

    def compact(self, function: Any, module_map: ModuleMap, reuse: dict) -> None:
        """
        Compact a function.
        
        Args:
            function: The function to compact
            module_map: The module map for adjusting handles
            reuse: Reused named expressions storage
        """
        # This is implemented in functions.py FunctionCompactor
        # For now, placeholder
        pass
