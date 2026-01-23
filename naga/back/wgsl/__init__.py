"""
Backend for WGSL (WebGPU Shading Language).

This module contains the writer implementation for converting Naga IR
to WGSL code.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntFlag
import io

from .expression_writer import WGSLExpressionWriter
from .statement_writer import WGSLStatementWriter
from ...error import ShaderError
from ...common.diagnostic_debug import DiagnosticDebug


class WriterFlags(IntFlag):
    """Flags controlling WGSL writing behavior."""

    EXPLICIT_TYPES = 0x1  # Always annotate type information instead of inferring


class Attribute:
    """WGSL attribute representation."""

    Binding: Any
    BuiltIn: Any
    Group: Any
    Invariant: Any
    Interpolate: Any
    Location: Any
    BlendSrc: Any
    Stage: Any
    WorkGroupSize: Any
    MeshStage: Any
    TaskPayload: Any
    PerPrimitive: Any


class Indirection:
    """Controls how pointer expressions are rendered in WGSL."""

    Ordinary = "ordinary"  # Render as ptr-typed expressions
    Reference = "reference"  # Render as reference-typed expressions


class Writer:
    """
    Writer for converting Naga IR modules to WGSL code.

    Maintains internal state to output a Module into WGSL format.
    """

    def __init__(
        self, out: Union[str, io.StringIO], flags: WriterFlags = WriterFlags(0)
    ):
        """Initialize the writer with output stream and flags."""
        if isinstance(out, str):
            self.out = io.StringIO(out)
        else:
            self.out = out
        self.flags = flags
        self.names: Dict[str, str] = {}
        self.named_expressions: Dict[str, Any] = {}
        self.required_polyfills: Set[str] = set()

    def write(self, module: Any, info: Any) -> str:
        """
        Write the complete module to WGSL.

        Args:
            module: The Naga IR module to write
            info: Module validation information

        Returns:
            The generated WGSL code as string

        Raises:
            ShaderError: If writing fails
        """
        try:
            self.reset(module)

            # Write enable declarations
            self._write_enable_declarations(module)

            # Write structs
            self._write_structs(module)

            # Write constants
            self._write_constants(module)

            # Write overrides
            self._write_overrides(module)

            # Write globals
            self._write_globals(module)

            # Write functions
            self._write_functions(module, info)

            # Write entry points
            self._write_entry_points(module, info)

            # Write polyfills
            self._write_polyfills()

            return self.out.getvalue()

        except Exception as e:
            raise ShaderError(f"WGSL writing failed: {e}") from e

    def reset(self, module: Any) -> None:
        """Reset writer state for a new module."""
        self.names.clear()
        self.named_expressions.clear()
        self.required_polyfills.clear()

        # Initialize naming - simplified version
        self._initialize_names(module)

    def _initialize_names(self, module: Any) -> None:
        """Initialize name mappings for module elements."""
        # Placeholder implementation - would need full IR access
        pass

    def _write_enable_declarations(self, module: Any) -> None:
        """Write enable declarations needed by the module."""
        needs_f16 = False
        needs_dual_source = False
        needs_clip_distances = False
        needs_mesh_shaders = False

        # Analyze module for required enables
        # This would analyze types, entry points, etc.

        written = False
        if needs_f16:
            self.out.write("enable f16;\n\n")
            written = True
        if needs_dual_source:
            self.out.write("enable dual_source_blending;\n\n")
            written = True
        if needs_clip_distances:
            self.out.write("enable clip_distances;\n\n")
            written = True
        if needs_mesh_shaders:
            self.out.write("enable wgpu_mesh_shader;\n\n")
            written = True

    def _write_structs(self, module: Any) -> None:
        """Write all struct type definitions."""
        for handle, ty in module.types.items():
            if hasattr(ty, "inner") and ty.inner and hasattr(ty.inner, "members"):
                self._write_struct(module, handle, ty.inner.members)
                self.out.write("\n")

    def _write_struct(self, module: Any, handle: Any, members: List[Any]) -> None:
        """Write a single struct definition."""
        struct_name = self.names.get(str(handle), f"Struct{handle}")
        self.out.write(f"struct {struct_name} {{\n")

        for i, member in enumerate(members):
            member_name = self.names.get(f"{handle}_{i}", f"field_{i}")
            self.out.write(f"    {member_name}: {self._type_to_string(member.ty)},\n")

        self.out.write("};\n")

    def _write_constants(self, module: Any) -> None:
        """Write named constants."""
        for handle, constant in module.constants.items():
            if constant.name:
                const_name = self.names.get(str(handle), constant.name)
                value = self._constant_value_to_string(constant)
                self.out.write(
                    f"let {const_name}: {self._type_to_string(constant.ty)} = {value};\n"
                )
        self.out.write("\n")

    def _write_overrides(self, module: Any) -> None:
        """Write pipeline overrides."""
        for handle, override in module.overrides.items():
            if override.name:
                override_name = self.names.get(str(handle), override.name)
                self.out.write(f"override {override_name}")
                if override.ty:
                    self.out.write(f": {self._type_to_string(override.ty)}")
                self.out.write(";\n")
        self.out.write("\n")

    def _write_globals(self, module: Any) -> None:
        """Write global variables."""
        for ty, global_var in module.global_variables.items():
            var_name = self.names.get(
                str(id(global_var)), global_var.name or f"var_{id(global_var)}"
            )
            var_decl = (
                f"var<{self._address_space_to_string(global_var.space)}> {var_name}"
            )
            if global_var.ty:
                var_decl += f": {self._type_to_string(global_var.ty)}"
            if global_var.binding:
                var_decl += f" @{self._binding_to_string(global_var.binding)}"
            self.out.write(f"{var_decl};\n")
        self.out.write("\n")

    def _write_functions(self, module: Any, info: Any) -> None:
        """Write regular functions."""
        for handle, function in module.functions.items():
            self._write_function(module, function, info[handle], "function")
            self.out.write("\n")

    def _write_entry_points(self, module: Any, info: Any) -> None:
        """Write entry point functions."""
        for index, ep in enumerate(module.entry_points):
            self._write_function(
                module, ep.function, info.get_entry_point(index), "entry_point", index
            )
            if index < len(module.entry_points) - 1:
                self.out.write("\n")

    def _write_function(
        self,
        module: Any,
        func: Any,
        func_info: Any,
        func_type: str,
        index: Optional[int] = None,
    ) -> None:
        """Write a single function."""
        func_name_key = (
            f"entry_point_{index}" if index is not None else f"function_{id(func)}"
        )
        func_name = self.names.get(func_name_key, f"func_{func_name_key}")

        # Write function signature
        self.out.write(f"fn {func_name}(")

        # Write arguments
        for i, arg in enumerate(func.arguments):
            if arg.binding:
                self._write_attributes([self._binding_to_attribute(arg.binding)])

            arg_name = self.names.get(f"{func_name_key}_arg_{i}", f"arg_{i}")
            self.out.write(f"{arg_name}: {self._type_to_string(arg.ty)}")

            if i < len(func.arguments) - 1:
                self.out.write(", ")

        self.out.write(")")

        # Write return type
        if func.result:
            self.out.write(f" -> {self._type_to_string(func.result.ty)}")

        self.out.write(" {\n")

        # Create writers for this function
        expr_writer = WGSLExpressionWriter(module, self.names, func.expressions)
        stmt_writer = WGSLStatementWriter(module, self.names, expr_writer)

        # Write function body
        if hasattr(func, "body") and func.body:
            body_code = stmt_writer.write_block(func.body, 1)
            self.out.write(body_code)

        self.out.write("}\n")


    def _type_to_string(self, ty: Any) -> str:
        """Convert a type to WGSL string representation."""
        if ty is None:
            return "void"

        if hasattr(ty, "inner"):
            inner = ty.inner
        else:
            inner = ty

        if hasattr(inner, "ty"):
            inner = inner.ty

        if str(inner).startswith("Scalar."):
            scalar_type = str(inner).split(".")[1]
            return {
                "F16": "f16",
                "F32": "f32",
                "F64": "f64",
                "I32": "i32",
                "U32": "u32",
                "Bool": "bool",
            }.get(scalar_type, str(inner).lower())
        elif str(inner).startswith("Vector"):
            # Vector<count, scalar>
            return f"vec{inner.size}<{self._type_to_string(inner.scalar)}>"
        elif str(inner).startswith("Matrix"):
            # Matrix<columns, rows, scalar>
            return (
                f"mat{inner.columns}x{inner.rows}<{self._type_to_string(inner.scalar)}>"
            )
        elif str(inner).startswith("Array"):
            # Array<element, count>
            count = getattr(inner, "size", None)
            if count is None:
                return f"array<{self._type_to_string(inner.element)}>"
            else:
                return f"array<{self._type_to_string(inner.element)}, {count}>"
        else:
            return str(inner).lower()

    def _constant_value_to_string(self, constant: Any) -> str:
        """Convert a constant value to string representation."""
        if hasattr(constant, "value"):
            return str(constant.value)
        return "0"

    def _address_space_to_string(self, space: Any) -> str:
        """Convert address space to WGSL string."""
        space_map = {
            "Function": "function",
            "Private": "private",
            "WorkGroup": "workgroup",
            "Uniform": "uniform",
            "Storage": "storage",
            "Handle": "handle",
            "TaskPayload": "task_payload",
        }
        return space_map.get(str(space), "function")

    def _binding_to_string(self, binding: Any) -> str:
        """Convert binding to attribute string."""
        if hasattr(binding, "location"):
            return f"@location({binding.location})"
        elif hasattr(binding, "builtin"):
            return f"@builtin({binding.builtin})"
        elif hasattr(binding, "group") and hasattr(binding, "binding"):
            return f"@group({binding.group}) @binding({binding.binding})"
        else:
            return ""

    def _binding_to_attribute(self, binding: Any) -> Any:
        """Convert binding to Attribute object."""
        attr = Attribute()
        if hasattr(binding, "location"):
            attr.Location = binding.location
        elif hasattr(binding, "builtin"):
            attr.BuiltIn = binding.builtin
        elif hasattr(binding, "group") and hasattr(binding, "binding"):
            attr.Group = binding.group
            attr.Binding = binding.binding
        return attr

    def _write_attributes(self, attributes: List[Any]) -> None:
        """Write attribute list."""
        for attr in attributes:
            if hasattr(attr, "Location"):
                self.out.write(f"@location({attr.Location}) ")
            elif hasattr(attr, "BuiltIn"):
                self.out.write(f"@builtin({attr.BuiltIn}) ")
            elif hasattr(attr, "Group") and hasattr(attr, "Binding"):
                self.out.write(f"@group({attr.Group}) @binding({attr.Binding}) ")
            elif hasattr(attr, "Stage"):
                self.out.write(f"@stage({attr.Stage}) ")

    def _write_polyfills(self) -> None:
        """Write required polyfill functions."""
        for polyfill in self.required_polyfills:
            self.out.write(f"\n{polyfill}\n")

    def finish(self) -> str:
        """Finish writing and return the complete output."""
        return self.out.getvalue()


def write_string(module: Any, info: Any, flags: WriterFlags = WriterFlags(0)) -> str:
    """
    Write a module to WGSL string.

    Args:
        module: The Naga IR module
        info: Module validation info
        flags: Writer flags

    Returns:
        Generated WGSL code as string
    """
    writer = Writer("", flags)
    return writer.write(module, info)
