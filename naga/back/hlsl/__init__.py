"""
Backend for HLSL (High-Level Shading Language).

This module contains the writer implementation for converting Naga IR
to HLSL code with support for various shader model versions.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntEnum, IntFlag
import io

from .expression_writer import HLSLExpressionWriter
from .statement_writer import HLSLStatementWriter
from ...error import ShaderError
from ...ir.type import (
    Type, TypeInner, TypeInnerType, ScalarKind, VectorSize, Matrix, ArraySizeType
)


class ShaderModel(IntEnum):
    """HLSL shader model versions."""

    SM_5_0 = 50
    SM_5_1 = 51
    SM_6_0 = 60
    SM_6_1 = 61
    SM_6_2 = 62
    SM_6_3 = 63
    SM_6_4 = 64
    SM_6_5 = 65
    SM_6_6 = 66
    SM_6_7 = 67


class ShaderStage(IntEnum):
    """HLSL shader stage enumeration."""

    Vertex = 0
    Hull = 1
    Domain = 2
    Geometry = 3
    Pixel = 4
    Compute = 5


class Options:
    """HLSL writer options."""

    def __init__(self, shader_model: ShaderModel = ShaderModel.SM_5_0):
        """
        Initialize HLSL writer options.

        Args:
            shader_model: HLSL shader model version
        """
        self.shader_model = shader_model

    def supports_ray_tracing(self) -> bool:
        """Check if shader model supports ray tracing."""
        return self.shader_model >= ShaderModel.SM_6_5

    def supports_mesh_shaders(self) -> bool:
        """Check if shader model supports mesh/task shaders."""
        return self.shader_model >= ShaderModel.SM_6_5


class BindingMap:
    """Mapping between resources and bindings."""

    def __init__(self):
        self.bindings: Dict[str, int] = {}

    def insert(self, resource_binding: str, bind_target: int) -> None:
        """Insert a binding mapping."""
        self.bindings[resource_binding] = bind_target

    def get(self, resource_binding: str) -> Optional[int]:
        """Get binding target for resource."""
        return self.bindings.get(resource_binding)


class ReflectionInfo:
    """Reflection information for HLSL shaders."""

    def __init__(self):
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.uniforms: Dict[str, Any] = {}
        self.textures: Dict[str, Any] = {}
        self.samplers: Dict[str, Any] = {}


class Writer:
    """
    Writer for converting Naga IR modules to HLSL code.

    Maintains internal state to output a Module into HLSL format.
    """

    def __init__(
        self,
        out: Union[str, io.StringIO],
        module: Any,
        info: Any,
        options: Options,
        binding_map: Optional[BindingMap] = None,
    ):
        """
        Initialize the HLSL writer.

        Args:
            out: Output stream
            module: The Naga IR module
            info: Module validation information
            options: HLSL writer options
            binding_map: Optional binding map for resources
        """
        if isinstance(out, str):
            self.out = io.StringIO(out)
        else:
            self.out = out

        self.module = module
        self.info = info
        self.options = options
        self.binding_map = binding_map or BindingMap()

        # Internal state
        self.names: Dict[str, str] = {}
        self.namer = HLSLNameGenerator()
        self.reflection_info = ReflectionInfo()

        # Type information cache
        self.type_cache: Dict[str, str] = {}

    def write(self, entry_point_name: str, shader_stage: ShaderStage) -> ReflectionInfo:
        """
        Write the complete module to HLSL.

        Returns:
            Reflection information about the generated shader

        Raises:
            ShaderError: If writing fails
        """
        try:
            self._initialize_names()
            self._find_entry_point(entry_point_name, shader_stage)

            # Write HLSL header
            self._write_header(shader_stage)

            # Write struct definitions
            self._write_struct_definitions()

            # Write functions
            self._write_functions()

            # Write entry point
            self._write_entry_point(entry_point_name, shader_stage)

            return self.reflection_info

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ShaderError(f"HLSL writing failed: {e}") from e

    def _initialize_names(self) -> None:
        """Initialize name mappings."""
        self.names.clear()
        self.namer.reset()

        # Reserve HLSL keywords
        reserved_keywords = [
            "float", "double", "int", "uint", "bool", "half",
            "float2", "float3", "float4",
            "matrix", "vector", "sampler", "Texture2D", "RWTexture2D",
            "cbuffer", "tbuffer", "StructuredBuffer", "RWStructuredBuffer",
            "register", "return", "if", "else", "switch", "case", "default",
            "break", "continue", "discard", "while", "for", "do"
        ]

        for keyword in reserved_keywords:
            self.namer.reserve_name(keyword)

    def _find_entry_point(self, name: str, stage: ShaderStage) -> None:
        """Find the entry point."""
        self.entry_point = None
        for ep in self.module.entry_points:
            if ep.name == name and ep.stage.value == stage.value:
                self.entry_point = ep
                break

    def _write_header(self, stage: ShaderStage) -> None:
        """Write HLSL header."""
        self.out.write(f"// HLSL {stage.name} Shader - SM {self.options.shader_model}\n\n")

    def _write_struct_definitions(self) -> None:
        """Write struct definitions."""
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
            type_name = self._type_to_hlsl(member.ty)
            self.out.write(f"    {type_name} {member_name};\n")

        self.out.write("};\n")

    def _write_functions(self) -> None:
        """Write functions."""
        for handle, function in self.module.functions.items():
            self._write_function(handle, function)
            self.out.write("\n")

    def _write_function(self, handle: Any, function: Any) -> None:
        """Write a function."""
        func_name = self.names.get(f"func_{handle}", function.name or f"func_{handle}")
        
        expr_writer = HLSLExpressionWriter(self.module, self.names, function.expressions)
        stmt_writer = HLSLStatementWriter(self.module, self.names, expr_writer)

        return_type = self._type_to_hlsl(function.result.ty if function.result else None)
        self.out.write(f"{return_type} {func_name}(")

        for i, arg in enumerate(function.arguments):
            arg_name = self.names.get(f"arg_{i}", arg.name or f"arg_{i}")
            type_name = self._type_to_hlsl(arg.ty)
            self.out.write(f"{type_name} {arg_name}")
            if i < len(function.arguments) - 1:
                self.out.write(", ")

        self.out.write(") {\n")
        
        if function.body:
            self.out.write(stmt_writer.write_block(function.body, 1))

        self.out.write("}\n")

    def _write_entry_point(self, name: str, stage: ShaderStage) -> None:
        """Write the entry point."""
        if not self.entry_point:
            return

        match stage:
            case ShaderStage.Vertex:
                self.out.write(f"float4 {name}() : SV_Position {{\n")
                self.out.write("    return float4(0.0, 0.0, 0.0, 1.0);\n")
                self.out.write("}\n")
            case ShaderStage.Pixel:
                self.out.write(f"float4 {name}() : SV_Target {{\n")
                self.out.write("    return float4(1.0, 0.0, 0.0, 1.0);\n")
                self.out.write("}\n")
            case ShaderStage.Compute:
                self.out.write("[numthreads(1, 1, 1)]\n")
                self.out.write(f"void {name}() {{\n")
                self.out.write("}\n")
            case _:
                self.out.write(f"void {name}() {{\n")
                self.out.write("}\n")

    def _type_to_hlsl(self, ty_handle: Any) -> str:
        """Convert a type to HLSL string representation."""
        if ty_handle is None:
            return "void"
        
        if isinstance(ty_handle, int):
            ty = self.module.types[ty_handle]
            if ty.name:
                 return ty.name
            return self._type_inner_to_hlsl(ty.inner)
            
        return "void"

    def _type_inner_to_hlsl(self, inner: TypeInner) -> str:
        """Convert a TypeInner to HLSL string representation."""
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
                scalar_str = self._type_inner_to_hlsl(TypeInner.new_scalar(vec.scalar))
                return f"{scalar_str}{size}"
            
            case TypeInnerType.MATRIX:
                mat = inner.matrix
                cols = mat.columns.value
                rows = mat.rows.value
                scalar_str = self._type_inner_to_hlsl(TypeInner.new_scalar(mat.scalar))
                # HLSL matrix<float, 2, 2> or float2x2
                return f"{scalar_str}{cols}x{rows}"
            
            case TypeInnerType.ARRAY:
                arr = inner.array
                element_str = self._type_to_hlsl(arr.base)
                if arr.size.type == ArraySizeType.DYNAMIC:
                    return f"{element_str} " # StructuredBuffer handles this usually
                elif arr.size.type == ArraySizeType.CONSTANT:
                    return f"{element_str}[{arr.size.constant.value}]"
                return f"{element_str} "

            case TypeInnerType.IMAGE:
                # HLSL Texture2D<float>
                img = inner.image
                dim = str(img.dim).lower().replace("dim", "")
                base = self._type_inner_to_hlsl(TypeInner.new_scalar(img.class_type.scalar))
                return f"Texture{dim}<{base}>"

            case TypeInnerType.SAMPLER:
                return "SamplerState"

            case _:
                raise ShaderError(f"Unsupported HLSL type: {inner.type}")

    def finish(self) -> str:
        """Finish writing and return the complete output."""
        return self.out.getvalue()


class HLSLNameGenerator:
    """Generator for unique HLSL names."""

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
    binding_map: Optional[BindingMap] = None,
) -> str:
    """
    Write a module to HLSL string.

    Args:
        module: The Naga IR module
        info: Module validation info
        options: HLSL writer options
        entry_point: Entry point function name
        shader_stage: Shader stage type
        binding_map: Optional binding map for resources

    Returns:
        Generated HLSL code as string
    """
    writer = Writer("", module, info, options, binding_map)
    writer.write(entry_point, shader_stage)
    return writer.finish()
