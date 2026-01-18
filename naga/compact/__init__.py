from typing import Any, Optional
from .handle_set_map import Handle, HandleSet, HandleMap
from .functions import FunctionCompactor, FunctionTracer
from .expressions import ExpressionCompactor, ExpressionTracer
from .statements import StatementCompactor, StatementTracer
from .types import TypeCompactor, TypeTracer


class KeepUnused:
    """Configuration option for compaction."""
    NO = False
    YES = True


class ModuleTracer:
    """
    Traces module usage to determine which items are used.
    """
    def __init__(self, module: Any) -> None:
        self.module = module
        self.functions_used = HandleSet()
        self.functions_pending = HandleSet()
        self.types_used = HandleSet()
        self.global_variables_used = HandleSet()
        self.constants_used = HandleSet()
        self.overrides_used = HandleSet()
        self.global_expressions_used = HandleSet()

    def trace_entry_points(self) -> None:
        """Trace entry points to determine usage."""
        for entry_point in self.module.entry_points:
            # Trace workgroup size overrides
            if hasattr(entry_point, "workgroup_size_overrides"):
                for size in entry_point.workgroup_size_overrides:
                    if size:
                        self.global_expressions_used.insert(size)

            # Trace task payload
            if hasattr(entry_point, "task_payload") and entry_point.task_payload:
                self.global_variables_used.insert(entry_point.task_payload)

            # Trace mesh info
            if hasattr(entry_point, "mesh_info") and entry_point.mesh_info:
                mesh_info = entry_point.mesh_info
                self.global_variables_used.insert(mesh_info.output_variable)
                self.types_used.insert(mesh_info.vertex_output_type)
                self.types_used.insert(mesh_info.primitive_output_type)
                if mesh_info.max_vertices_override:
                    self.global_expressions_used.insert(mesh_info.max_vertices_override)
                if mesh_info.max_primitives_override:
                    self.global_expressions_used.insert(mesh_info.max_primitives_override)

            # Trace u32 type for task/mesh shaders
            if entry_point.stage in ["Task", "Mesh"]:
                for handle, ty in self.module.types:
                    if hasattr(ty.inner, "scalar") and ty.inner.scalar.kind == "Uint":
                        self.types_used.insert(handle)
                        break

            # Trace function
            function_tracer = self.as_function(entry_point.function)
            function_tracer.trace()

    def trace_functions(self) -> None:
        """Trace functions to determine usage."""
        while self.functions_pending:
            function_handle = self.functions_pending.pop()
            if not self.functions_used.contains(function_handle):
                continue

            function = self.module.functions.get(function_handle)
            if function:
                function_tracer = self.as_function(function)
                function_tracer.trace()

    def trace_global_expressions_and_types(self) -> None:
        """Trace global expressions and types."""
        # Get max dependencies for types
        max_dep = []
        for handle, ty in self.module.types:
            # Placeholder: calculate max dependency
            max_dep.append(None)

        # Trace types and expressions in reverse order
        exprs = list(self.module.global_expressions.items())
        exprs.reverse()
        expr_iter = iter(exprs)

        for (ty_handle, ty), dep in zip(self.module.types.items(), max_dep):
            # Trace expressions before this type
            while True:
                try:
                    next_expr = next(expr_iter)
                    if next_expr[0] > dep:
                        if self.global_expressions_used.contains(next_expr[0]):
                            self.as_const_expression().trace_expression(next_expr[1])
                    else:
                        break
                except StopIteration:
                    break

            # Trace the type
            if self.types_used.contains(ty_handle):
                self.as_type().trace_type(ty)

        # Trace remaining expressions
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
    """
    def __init__(self) -> None:
        self.functions = HandleMap()
        self.types = HandleMap()
        self.globals = HandleMap()
        self.constants = HandleMap()
        self.overrides = HandleMap()
        self.global_expressions = HandleMap()

    def adjust_special_types(self, special: Any) -> None:
        """Adjust special types."""
        if hasattr(special, "ray_desc") and special.ray_desc:
            self.types.adjust(special.ray_desc)
        if hasattr(special, "ray_intersection") and special.ray_intersection:
            self.types.adjust(special.ray_intersection)
        if hasattr(special, "ray_vertex_return") and special.ray_vertex_return:
            self.types.adjust(special.ray_vertex_return)
        if hasattr(special, "external_texture_params") and special.external_texture_params:
            self.types.adjust(special.external_texture_params)
        if hasattr(special, "external_texture_transfer_function") and special.external_texture_transfer_function:
            self.types.adjust(special.external_texture_transfer_function)

        if hasattr(special, "predeclared_types"):
            for handle in special.predeclared_types.values():
                self.types.adjust(handle)

    def adjust_doc_comments(self, doc_comments: Any) -> None:
        """Adjust doc comments."""
        # Placeholder implementation
        pass


class FunctionMap:
    """
    Maps old handles to new handles for a function.
    """
    def __init__(self) -> None:
        self.expressions = HandleMap()

    def compact(self, function: Any, module_map: ModuleMap, reuse: Any) -> None:
        """Compact a function."""
        # Placeholder implementation
        pass


class Compact:
    """
    Module compaction logic.
    """
    def __init__(self) -> None:
        self.module_tracer = None
        self.module_map = None

    def compact(self, module: Any, keep_unused: bool = False) -> Any:
        """
        Compact a module by removing unused items.

        Args:
            module: The module to compact
            keep_unused: If True, keep unused functions, global variables, and named types

        Returns:
            The compacted module
        """
        # Create module tracer
        self.module_tracer = ModuleTracer(module)

        # Trace entry points
        self.module_tracer.trace_entry_points()

        # If keeping unused, mark all functions as used
        if keep_unused:
            for handle in self.module_tracer.module.functions:
                self.module_tracer.functions_used.insert(handle)
                self.module_tracer.functions_pending.insert(handle)

        # Trace functions
        self.module_tracer.trace_functions()

        # Trace global expressions and types
        self.module_tracer.trace_global_expressions_and_types()

        # Create module map
        self.module_map = ModuleMap()

        # Placeholder: adjust handles in the module
        # This would involve:
        # 1. Removing unused items from arenas
        # 2. Adjusting handles in remaining items
        # 3. Updating special types and doc comments

        return module
