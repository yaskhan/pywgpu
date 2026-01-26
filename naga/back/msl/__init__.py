"""
Backend for MSL (Metal Shading Language).

This module contains the writer implementation for converting Naga IR
to MSL code with support for various Metal features and capabilities.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntEnum, IntFlag
import io

from .expression_writer import MSLExpressionWriter
from .statement_writer import MSLStatementWriter
from .keywords import RESERVED_KEYWORD_SET
from ...error import ShaderError
from ...ir.type import (
    Type, TypeInner, TypeInnerType, ScalarKind, VectorSize, Matrix, ArraySizeType
)


class ShaderStage(IntEnum):
    """Shader stage enumeration."""

    Vertex = 0
    Fragment = 1
    Compute = 2
    Task = 3
    Mesh = 4


class Options:
    """MSL writer options."""

    def __init__(
        self,
        lang_version: tuple[int, int] = (1, 0),
        per_entry_point_map: Optional[Dict[str, Any]] = None,
        inline_samplers: Optional[List[Any]] = None,
        spirv_cross_compatibility: bool = False,
        fake_missing_bindings: bool = True,
        bounds_check_policies: Optional[Any] = None,
        zero_initialize_workgroup_memory: bool = True,
        force_loop_bounding: bool = True,
    ):
        """
        Initialize MSL writer options.
        
        Args:
            lang_version: (Major, Minor) target version of the Metal Shading Language
            per_entry_point_map: Map of entry-point resources, indexed by entry point function name, to slots
            inline_samplers: Samplers to be inlined into the code
            spirv_cross_compatibility: Make it possible to link different stages via SPIRV-Cross
            fake_missing_bindings: Don't panic on missing bindings, instead generate invalid MSL
            bounds_check_policies: Bounds checking policies
            zero_initialize_workgroup_memory: Should workgroup variables be zero initialized (by polyfilling)
            force_loop_bounding: If set, loops will have code injected into them, forcing the compiler to think the number of iterations is bounded
        """
        self.lang_version = lang_version
        self.per_entry_point_map = per_entry_point_map or {}
        self.inline_samplers = inline_samplers or []
        self.spirv_cross_compatibility = spirv_cross_compatibility
        self.fake_missing_bindings = fake_missing_bindings
        self.bounds_check_policies = bounds_check_policies
        self.zero_initialize_workgroup_memory = zero_initialize_workgroup_memory
        self.force_loop_bounding = force_loop_bounding

    def supports_ray_tracing(self) -> bool:
        """Check if Metal version supports ray tracing."""
        return True  # Metal supports ray tracing

    def supports_mesh_shaders(self) -> bool:
        """Check if Metal version supports mesh/task shaders."""
        return True  # Metal supports mesh shaders


class BindingInfo:
    """Binding information for resources."""

    def __init__(self, resource_binding: str, bind_target: int):
        self.resource_binding = resource_binding
        self.bind_target = bind_target


class BindTarget:
    """Bind target for resources."""

    def __init__(
        self,
        buffer: Optional[int] = None,
        texture: Optional[int] = None,
        sampler: Optional[int] = None,
    ):
        self.buffer = buffer
        self.texture = texture
        self.sampler = sampler
        self.external_texture = None


class ReflectionInfo:
    """Reflection information for MSL shaders."""

    def __init__(self):
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.uniforms: Dict[str, Any] = {}
        self.textures: Dict[str, Any] = {}
        self.samplers: Dict[str, Any] = {}
        self.buffers: Dict[str, Any] = {}


class Writer:
    """
    Writer for converting Naga IR modules to MSL code.

    Maintains internal state to output a Module into MSL format.
    """

    def __init__(
        self,
        out: Union[str, io.StringIO],
        module: Any,
        info: Any,
        options: Options,
        entry_point: str,
        shader_stage: ShaderStage,
    ):
        """
        Initialize the MSL writer.

        Args:
            out: Output stream
            module: The Naga IR module
            info: Module validation information
            options: MSL writer options
            entry_point: Entry point function name
            shader_stage: Shader stage type
        """
        if isinstance(out, str):
            self.out = io.StringIO(out)
        else:
            self.out = out

        self.module = module
        self.info = info
        self.options = options
        self.entry_point_name = entry_point
        self.shader_stage = shader_stage

        # Internal state
        self.names: Dict[str, str] = {}
        self.namer = MSLNameGenerator()
        self.reflection_info = ReflectionInfo()
        self.required_features: Set[str] = set()

        # Entry point state
        self.entry_point = None

    def write(self) -> ReflectionInfo:
        """
        Write the complete module to MSL.

        Returns:
            Reflection information about the generated shader

        Raises:
            ShaderError: If writing fails
        """
        try:
            self._initialize_names()
            self._find_entry_point()

            # Write Metal includes and pragmas
            self._write_header()

            # Write struct definitions
            self._write_struct_definitions()

            # Write functions
            self._write_functions()

            # Write entry point
            self._write_entry_point()

            return self.reflection_info

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ShaderError(f"MSL writing failed: {e}") from e

    def _initialize_names(self) -> None:
        """Initialize name mappings for module elements."""
        self.names.clear()
        self.namer.reset()

        # Reserve MSL keywords
        reserved_keywords = [
            "device", "constant", "thread", "threadgroup",
            "vertex", "fragment", "kernel", "texture",
            "sampler", "float", "half", "double", "int", "uint", "bool",
            "float2", "float3", "float4",
            "half2", "half3", "half4",
            "int2", "int3", "int4",
            "uint2", "uint3", "uint4",
            "bool2", "bool3", "bool4",
            "float2x2", "float3x3", "float4x4",
            "using", "namespace", "metal", "struct", "template", "class"
        ]

        for keyword in reserved_keywords:
            self.namer.reserve_name(keyword)

    def _find_entry_point(self) -> None:
        """Find the entry point in the module."""
        self.entry_point = None
        for ep in self.module.entry_points:
            if ep.name == self.entry_point_name and ep.stage.value == self.shader_stage.value:
                self.entry_point = ep
                break

    def _write_header(self) -> None:
        """Write MSL file header."""
        stage_names = {
            ShaderStage.Vertex: "Vertex",
            ShaderStage.Fragment: "Fragment",
            ShaderStage.Compute: "Compute",
            ShaderStage.Mesh: "Mesh",
            ShaderStage.Task: "Task",
        }
        self.out.write(f"// {stage_names.get(self.shader_stage, 'Shader')} Shader\n\n")
        self.out.write("#include <metal_stdlib>\n")
        self.out.write("using namespace metal;\n\n")

    def _write_struct_definitions(self) -> None:
        """Write struct and type definitions."""
        for handle, ty in self.module.types.items():
            if hasattr(ty.inner, "members"):
                self._write_struct_definition(handle, ty.inner.members)
                self.out.write("\n")

    def _write_struct_definition(self, handle: Any, members: List[Any]) -> None:
        """Write a struct definition."""
        struct_name = self.names.get(f"type_{handle}", f"Struct{handle}")
        self.out.write(f"struct {struct_name} {{\n")

        for i, member in enumerate(members):
            member_name = self.names.get(f"member_{handle}_{i}", member.name or f"field_{i}")
            type_name = self._type_to_msl(member.ty)
            self.out.write(f"    {type_name} {member_name};\n")

        self.out.write("};\n")

    def _write_functions(self) -> None:
        """Write regular functions."""
        for handle, function in self.module.functions.items():
            self._write_function(handle, function)
            self.out.write("\n")

    def _write_function(self, handle: Any, function: Any) -> None:
        """Write a single function."""
        func_name = self.names.get(f"func_{handle}", function.name or f"func_{handle}")
        
        expr_writer = MSLExpressionWriter(self.module, self.names, function.expressions)
        stmt_writer = MSLStatementWriter(self.module, self.names, expr_writer)

        return_type = self._type_to_msl(function.result.ty if function.result else None)
        self.out.write(f"{return_type} {func_name}(")

        for i, arg in enumerate(function.arguments):
            arg_name = self.names.get(f"arg_{i}", arg.name or f"arg_{i}")
            type_name = self._type_to_msl(arg.ty)
            self.out.write(f"{type_name} {arg_name}")
            if i < len(function.arguments) - 1:
                self.out.write(", ")

        self.out.write(") {\n")
        
        if function.body:
            self.out.write(stmt_writer.write_block(function.body, 1))

        self.out.write("}\n")

    def _write_entry_point(self) -> None:
        """Write the entry point function."""
        if not self.entry_point:
            return

        ep = self.entry_point
        func = ep.function
        func_name = ep.name or "main0"
        
        # Write EP signature based on stage
        match self.shader_stage:
            case ShaderStage.Vertex:
                self.out.write(f"vertex float4 {func_name}() {{\n")
                self.out.write("    return float4(0.0);\n")
                self.out.write("}\n")
            case ShaderStage.Fragment:
                self.out.write(f"fragment float4 {func_name}() {{\n")
                self.out.write("    return float4(1.0, 0.0, 0.0, 1.0);\n")
                self.out.write("}\n")
            case ShaderStage.Compute:
                self.out.write(f"kernel void {func_name}() {{\n")
                self.out.write("}\n")
            case _:
                self.out.write(f"kernel void {func_name}() {{\n")
                self.out.write("}\n")

    def _type_to_msl(self, ty_handle: Any) -> str:
        """Convert a type to MSL string representation."""
        if ty_handle is None:
            return "void"
        
        if isinstance(ty_handle, int):
            ty = self.module.types[ty_handle]
            if ty.name:
                 return ty.name
            return self._type_inner_to_msl(ty.inner)
            
        return "void"

    def _type_inner_to_msl(self, inner: TypeInner) -> str:
        """Convert a TypeInner to MSL string representation."""
        match inner.type:
            case TypeInnerType.SCALAR:
                match inner.scalar.kind:
                    case ScalarKind.F16: return "half"
                    case ScalarKind.F32: return "float"
                    case ScalarKind.F64: return "double"
                    case ScalarKind.I32: return "int"
                    case ScalarKind.UINT: return "uint"
                    case ScalarKind.BOOL: return "bool"
                    case _: return "float"
            
            case TypeInnerType.VECTOR:
                vec = inner.vector
                size = vec.size.value
                scalar_str = self._type_inner_to_msl(TypeInner.new_scalar(vec.scalar))
                return f"{scalar_str}{size}"
            
            case TypeInnerType.MATRIX:
                mat = inner.matrix
                cols = mat.columns.value
                rows = mat.rows.value
                scalar_str = self._type_inner_to_msl(TypeInner.new_scalar(mat.scalar))
                return f"{scalar_str}{cols}x{rows}"
            
            case TypeInnerType.ARRAY:
                arr = inner.array
                element_str = self._type_to_msl(arr.base)
                if arr.size.type == ArraySizeType.DYNAMIC:
                    return f"device {element_str}*"
                elif arr.size.type == ArraySizeType.CONSTANT:
                    return f"array<{element_str}, {arr.size.constant.value}>"
                return f"device {element_str}*"

            case TypeInnerType.IMAGE:
                # MSL texture types: texture2d<float, access::read>
                img = inner.image
                dim = str(img.dim).lower().replace("dim", "")
                base = self._type_inner_to_msl(TypeInner.new_scalar(img.class_type.scalar))
                return f"texture{dim}<{base}, access::read>"

            case TypeInnerType.SAMPLER:
                return "sampler"

            case _:
                raise ShaderError(f"Unsupported MSL type: {inner.type}")

    def finish(self) -> str:
        """Finish writing and return the complete output."""
        return self.out.getvalue()


class MSLNameGenerator:
    """Generator for unique MSL names."""

    def __init__(self):
        self.used_names: Set[str] = set()
        self.name_map: Dict[str, str] = {}

    def reset(self) -> None:
        """Reset the name generator."""
        self.used_names.clear()
        self.name_map.clear()

    def reserve_name(self, name: str) -> None:
        """Reserve a name to avoid conflicts."""
        self.used_names.add(name)

    def get_name(self, original_name: str) -> str:
        """Get a unique name based on the original."""
        if original_name not in self.name_map:
            unique_name = original_name
            counter = 1
            while unique_name in self.used_names:
                unique_name = f"{original_name}_{counter}"
                counter += 1

            self.name_map[original_name] = unique_name
            self.used_names.add(unique_name)

        return self.name_map[original_name]


def write_string(
    module: Any,
    info: Any,
    options: Options,
    entry_point: str,
    shader_stage: ShaderStage,
) -> str:
    """
    Write a module to MSL string.

    Args:
        module: The Naga IR module
        info: Module validation info
        options: MSL writer options
        entry_point: Entry point function name
        shader_stage: Shader stage type

    Returns:
        Generated MSL code as string
    """
    writer = Writer("", module, info, options, entry_point, shader_stage)
    writer.write()
    return writer.finish()
