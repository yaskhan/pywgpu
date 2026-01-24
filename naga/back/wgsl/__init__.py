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
from ...ir.type import (
    Type, TypeInner, TypeInnerType, ScalarKind, VectorSize,
    ImageDimension, ImageClass, StorageFormat
)


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
        # Keep this mapping conservative; use provided IR names when present.
        if hasattr(module, "types"):
            for i, ty in enumerate(module.types):
                if getattr(ty, "name", None):
                    self.names[f"type_{i}"] = ty.name
        if hasattr(module, "constants"):
            for i, const in enumerate(module.constants):
                if getattr(const, "name", None):
                    self.names[f"const_{i}"] = const.name
        if hasattr(module, "global_variables"):
            for i, var in enumerate(module.global_variables):
                if getattr(var, "name", None):
                    self.names[f"global_{i}"] = var.name
        if hasattr(module, "functions"):
            for i, func in enumerate(module.functions):
                if getattr(func, "name", None):
                    self.names[f"function_{i}"] = func.name

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
        for i, global_var in enumerate(module.global_variables):
            var_name = self.names.get(
                f"global_{id(global_var)}", global_var.name or f"var_{i}"
            )
            
            # Write binding and group info if present
            if global_var.binding:
                self._write_attributes([self._binding_to_attribute(global_var.binding)])

            var_decl = (
                f"var<{self._address_space_to_string(global_var.space)}> {var_name}"
            )
            if global_var.ty:
                var_decl += f": {self._type_to_string(global_var.ty)}"
            
            self.out.write(f"{var_decl};\n")
        self.out.write("\n")

    def _write_functions(self, module: Any, info: Any) -> None:
        """Write regular functions."""
        for handle, function in module.functions.items():
            self._write_function(module, function, info[handle], "function")
            self.out.write("\n")

    def _write_entry_points(self, module: Any, info: Any) -> None:
        """Write entry point functions."""
        for ep in module.entry_points:
            self._write_function(
                module, ep.function, info.get_function_info(ep.function), "entry_point", ep
            )
            self.out.write("\n")

    def _write_function(
        self,
        module: Any,
        func: Any,
        func_info: Any,
        func_type: str,
        ep: Optional[Any] = None,
    ) -> None:
        """Write a single function or entry point."""
        if ep is not None:
            # Entry point specific attributes
            self.out.write(f"@{ep.stage}\n")
            if ep.stage == "compute":
                ws = ep.workgroup_size
                self.out.write(f"@workgroup_size({ws[0]}, {ws[1]}, {ws[2]})\n")
            
            func_name = self.names.get(f"entry_point_{ep.name}", ep.name)
        else:
            func_name_key = f"function_{id(func)}"
            func_name = self.names.get(func_name_key, f"func_{func_name_key}")

        # Write function signature
        self.out.write(f"fn {func_name}(")

        # Write arguments
        for i, arg in enumerate(func.arguments):
            if arg.binding:
                self._write_attributes([self._binding_to_attribute(arg.binding)])

            arg_name = self.names.get(f"arg_{id(arg)}", arg.name or f"arg_{i}")
            self.out.write(f"{arg_name}: {self._type_to_string(arg.ty)}")

            if i < len(func.arguments) - 1:
                self.out.write(", ")

        self.out.write(")")

        # Write return type
        if func.result:
            if hasattr(func.result, "binding") and func.result.binding:
                self.out.write(" ")
                self._write_attributes([self._binding_to_attribute(func.result.binding)])
            
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

        if isinstance(ty, int):
            # This is a type handle, lookup the type
            if ty < 0 or ty >= len(self._module.types):
                return f"/* Invalid Type Handle {ty} */"
            ty_obj = self._module.types[ty]
            if ty_obj.name:
                return self.names.get(f"type_{ty}", ty_obj.name)
            return self._type_inner_to_string(ty_obj.inner)

        if isinstance(ty, Type):
            if ty.name:
                return ty.name
            return self._type_inner_to_string(ty.inner)

        if isinstance(ty, TypeInner):
            return self._type_inner_to_string(ty)

        return str(ty).lower()

    def _type_inner_to_string(self, inner: TypeInner) -> str:
        """Convert a TypeInner to WGSL string representation."""
        match inner.type:
            case TypeInnerType.SCALAR:
                scalar = inner.scalar
                match scalar.kind:
                    case ScalarKind.F16: return "f16"
                    case ScalarKind.F32: return "f32"
                    case ScalarKind.F64: return "f32"  # WGSL doesn't have f64 yet, use f32 or polyfill?
                    case ScalarKind.I32: return "i32"
                    case ScalarKind.UINT: return "u32"
                    case ScalarKind.BOOL: return "bool"
                    case ScalarKind.ABSTRACT_INT: return "abstract-int"
                    case ScalarKind.ABSTRACT_FLOAT: return "abstract-float"
                    case _: return "/* Unknown Scalar */"
            
            case TypeInnerType.VECTOR:
                vec = inner.vector
                size = vec.size.value
                scalar_str = self._type_inner_to_string(TypeInner.new_scalar(vec.scalar))
                return f"vec{size}<{scalar_str}>"
            
            case TypeInnerType.MATRIX:
                mat = inner.matrix
                cols = mat.columns.value
                rows = mat.rows.value
                scalar_str = self._type_inner_to_string(TypeInner.new_scalar(mat.scalar))
                return f"mat{cols}x{rows}<{scalar_str}>"
            
            case TypeInnerType.ARRAY:
                arr = inner.array
                base_str = self._type_to_string(arr.base)
                if arr.size is None:
                    return f"array<{base_str}>"
                else:
                    # size is ArraySize, need to handle Constant/Literal/Override
                    size_val = self._array_size_to_string(arr.size)
                    return f"array<{base_str}, {size_val}>"
            
            case TypeInnerType.STRUCT:
                # Structs are referred to by name usually
                # If it's an anonymous struct in NAGA, we should have generated a name for it
                return "/* struct */"
            
            case TypeInnerType.IMAGE:
                img = inner.image
                dim_str = self._image_dim_to_string(img.dim)
                if img.class_ == ImageClass.DEPTH:
                    return f"texture_depth_{dim_str}"
                elif img.class_ == ImageClass.STORAGE:
                    format_str = img.storage_format.value if hasattr(img.storage_format, "value") else str(img.storage_format)
                    return f"texture_storage_{dim_str}<{format_str}, write>" # Handle access
                else:
                    # Sampled or multisampled
                    prefix = "texture_multisampled" if img.class_ == ImageClass.MULTISAMPLED else "texture"
                    # Handle base type if known
                    return f"{prefix}_{dim_str}<f32>"
            
            case TypeInnerType.SAMPLER:
                return "sampler_comparison" if inner.sampler.comparison else "sampler"
            
            case TypeInnerType.POINTER:
                ptr = inner.pointer
                base_str = self._type_to_string(ptr.base)
                space_str = self._address_space_to_string(ptr.space)
                return f"ptr<{space_str}, {base_str}>"
            
            case _:
                raise ShaderError(f"Unsupported WGSL type: {inner.type}")

    def _image_dim_to_string(self, dim: ImageDimension) -> str:
        match dim:
            case ImageDimension.D1: return "1d"
            case ImageDimension.D2: return "2d"
            case ImageDimension.D3: return "3d"
            case ImageDimension.CUBE: return "cube"
            case _: return "2d"

    def _array_size_to_string(self, size: Any) -> str:
        # ArraySize can be Constant, Dynamic, or Literal
        if hasattr(size, "value"):
            return str(size.value)
        return str(size)

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
