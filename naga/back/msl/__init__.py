"""
Backend for MSL (Metal Shading Language).

This module contains the writer implementation for converting Naga IR
to MSL code with support for various Metal features and capabilities.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntEnum, IntFlag
import io

from ...error import ShaderError


class ShaderStage(IntEnum):
    """Shader stage enumeration."""
    Vertex = 0
    Fragment = 1
    Compute = 2
    Task = 3
    Mesh = 4


class Options:
    """MSL writer options."""
    
    def __init__(self):
        """Initialize MSL writer options."""
        pass
    
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
    
    def __init__(self, buffer: Optional[int] = None, texture: Optional[int] = None, 
                 sampler: Optional[int] = None):
        self.buffer = buffer
        self.texture = texture
        self.sampler = sampler
        self.external_texture = None


class ExternalTextureBinding:
    """External texture binding information."""
    
    def __init__(self, y_plane: int, cbcr_plane: int, alpha_plane: Optional[int] = None):
        self.y_plane = y_plane
        self.cbcr_plane = cbcr_plane
        self.alpha_plane = alpha_plane


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
    
    def __init__(self, out: Union[str, io.StringIO], module: Any, info: Any, 
                 options: Options, entry_point: str, shader_stage: ShaderStage):
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
        self.varyings: Dict[str, Any] = {}
        self.stage_in_struct = None
        self.stage_out_struct = None
    
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
            self._collect_required_features()
            
            # Write Metal includes and pragmas
            self._write_header()
            
            # Write resource bindings
            self._write_resource_bindings()
            
            # Write struct definitions
            self._write_struct_definitions()
            
            # Write sampler declarations
            self._write_samplers()
            
            # Write functions
            self._write_functions()
            
            # Write entry point
            self._write_entry_point()
            
            return self.reflection_info
            
        except Exception as e:
            raise ShaderError(f"MSL writing failed: {e}") from e
    
    def _initialize_names(self) -> None:
        """Initialize name mappings for module elements."""
        self.names.clear()
        self.namer.reset()
        
        # Reserve MSL keywords and built-in names
        reserved_keywords = [
            "device", "constant", "thread", "threadgroup", "vertex", "fragment", "kernel",
            "texture", "sampler", "threadgroup_imageblock", "threadgroup_memory",
            "intersection", "ray", "callable", "material", "object",
            "float", "half", "double", "int", "uint", "bool", "char", "short", "long",
            "float2", "float3", "float4", "half2", "half3", "half4",
            "double2", "double3", "double4",
            "int2", "int3", "int4", "uint2", "uint3", "uint4",
            "bool2", "bool3", "bool4", "char2", "char3", "char4",
            "short2", "short3", "short4", "long2", "long3", "long4",
            "float2x2", "float3x3", "float4x4", "float2x3", "float3x2",
            "float2x4", "float4x2", "float3x4", "float4x3",
            "half2x2", "half3x3", "half4x4", "half2x3", "half3x2",
            "half2x4", "half4x2", "half3x4", "half4x3",
            "texture2d", "texturecube", "texture2d_array", "texturecube_array",
            "texture2d_ms", "texture2d_array_ms", "depth2d", "depth2d_array",
            "sampler", "sampler_comparison",
            "vertex_id", "instance_id", "position", "point_size",
            "front_facing", "primitive_id", "sample_id", "sample_mask",
            "clip_distance", "cull_distance", "layer", "viewport_index",
            "thread_position_in_grid", "threadgroup_position_in_grid",
            "threadgroup_size", "grid_size", "dispatch_threads_per_threadgroup",
            "ray_id", "intersection_distance", "barycentric_coord",
            "kind", "world_position", "world_normal", "world_tangent",
            "world_bitangent", "world_velocity", "front_facing",
            "viewport_mask", "primitive_clip_mask", "layer_mask",
            "msaa_samples", "shading_rate",
            "vertex_descriptor", "primitive", "object_space_position",
            "object_space_normal", "object_space_tangent", "object_space_bitangent",
            "object_space_velocity", "world_space_position", "world_space_normal",
            "world_space_tangent", "world_space_bitangent", "world_space_velocity",
            "object_space_position_isosphere", "world_space_position_isosphere"
        ]
        
        for keyword in reserved_keywords:
            self.namer.reserve_name(keyword)
    
    def _find_entry_point(self) -> None:
        """Find the entry point in the module."""
        self.entry_point = None
        for ep in self.module.entry_points:
            if ep.name == self.entry_point_name and ep.stage == self.shader_stage:
                self.entry_point = ep
                break
    
    def _collect_required_features(self) -> None:
        """Analyze module and collect required MSL features."""
        # Placeholder implementation
        # Would analyze module types, expressions, etc.
        pass
    
    def _write_header(self) -> None:
        """Write MSL file header."""
        if self.shader_stage == ShaderStage.Vertex:
            self.out.write("// Vertex Shader\n")
        elif self.shader_stage == ShaderStage.Fragment:
            self.out.write("// Fragment Shader\n")
        elif self.shader_stage == ShaderStage.Compute:
            self.out.write("// Compute Shader\n")
        elif self.shader_stage == ShaderStage.Mesh:
            self.out.write("// Mesh Shader\n")
        elif self.shader_stage == ShaderStage.Task:
            self.out.write("// Task Shader\n")
        else:
            self.out.write("// Shader\n")
        
        self.out.write("\n")
        self.out.write("#include <metal_stdlib>\n")
        self.out.write("using namespace metal;\n\n")
    
    def _write_resource_bindings(self) -> None:
        """Write resource binding declarations."""
        self.out.write("// Resource bindings\n")
        self.out.write("struct naga_bindings {\n")
        self.out.write("    // Add resource bindings here\n")
        self.out.write("};\n\n")
    
    def _write_struct_definitions(self) -> None:
        """Write struct and type definitions."""
        for handle, ty in self.module.types.items():
            if hasattr(ty, 'inner') and ty.inner and hasattr(ty.inner, 'members'):
                self._write_struct_definition(handle, ty.inner.members)
                self.out.write("\n")
    
    def _write_struct_definition(self, handle: Any, members: List[Any]) -> None:
        """Write a struct definition."""
        struct_name = self.names.get(str(handle), f"Struct{handle}")
        self.out.write(f"struct {struct_name} {{\n")
        
        for i, member in enumerate(members):
            member_name = self.names.get(f"{handle}_{i}", f"field_{i}")
            type_name = self._type_to_msl(member.ty)
            self.out.write(f"    {type_name} {member_name}")
            # Add Metal-specific annotations if needed
            if hasattr(member, 'binding') and member.binding:
                # Add [[attribute]] annotations for MSL
                if hasattr(member.binding, 'location'):
                    self.out.write(f" [[attribute({member.binding.location})]]")
                elif hasattr(member.binding, 'builtin'):
                    builtin_name = self._builtin_to_attribute(member.binding.builtin)
                    if builtin_name:
                        self.out.write(f" [[{builtin_name}]]")
            self.out.write(";\n")
        
        self.out.write("};\n")
    
    def _write_samplers(self) -> None:
        """Write sampler state declarations."""
        self.out.write("// Sampler states\n")
        # Placeholder for sampler declarations
        # Would analyze module for sampler usage
        pass
    
    def _write_functions(self) -> None:
        """Write regular functions."""
        for handle, function in self.module.functions.items():
            self._write_function(function)
            self.out.write("\n")
    
    def _write_entry_point(self) -> None:
        """Write the main entry point function."""
        if not self.entry_point:
            return
        
        # Determine entry point signature based on shader stage
        if self.shader_stage == ShaderStage.Vertex:
            self._write_vertex_entry_point()
        elif self.shader_stage == ShaderStage.Fragment:
            self._write_fragment_entry_point()
        elif self.shader_stage == ShaderStage.Compute:
            self._write_compute_entry_point()
        elif self.shader_stage == ShaderStage.Mesh:
            self._write_mesh_entry_point()
        elif self.shader_stage == ShaderStage.Task:
            self._write_task_entry_point()
    
    def _write_vertex_entry_point(self) -> None:
        """Write vertex shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        
        # Write stage input structure
        self.out.write(f"struct VertexInput {{\n")
        if hasattr(self.entry_point, 'function'):
            for i, arg in enumerate(self.entry_point.function.arguments):
                param_name = f"param{i}"
                param_type = self._type_to_msl(arg.ty)
                self.out.write(f"    {param_type} {param_name}")
                if hasattr(arg, 'binding') and arg.binding:
                    if hasattr(arg.binding, 'location'):
                        self.out.write(f" [[attribute({arg.binding.location})]]")
                    elif hasattr(arg.binding, 'builtin'):
                        builtin_name = self._builtin_to_attribute(arg.binding.builtin)
                        if builtin_name:
                            self.out.write(f" [[{builtin_name}]]")
                self.out.write(";\n")
        self.out.write("};\n\n")
        
        # Write stage output structure
        self.out.write(f"struct VertexOutput {{\n")
        self.out.write(f"    float4 position [[position]];\n")
        self.out.write(f"    // Add other output parameters as needed\n")
        self.out.write(f"}};\n\n")
        
        # Write vertex function
        self.out.write(f"vertex VertexOutput {func_name}(VertexInput input [[stage_in]]) {{\n")
        self.out.write(f"    VertexOutput output;\n")
        self.out.write(f"    // TODO: Implement vertex shader logic\n")
        self.out.write(f"    output.position = float4(0.0, 0.0, 0.0, 1.0);\n")
        self.out.write(f"    return output;\n")
        self.out.write(f"}}\n")
    
    def _write_fragment_entry_point(self) -> None:
        """Write fragment shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        
        # Write stage input structure
        self.out.write(f"struct FragmentInput {{\n")
        self.out.write(f"    float4 position [[position]];\n")
        self.out.write(f"    // Add other input parameters as needed\n")
        self.out.write(f"}};\n\n")
        
        # Write fragment function
        self.out.write(f"fragment float4 {func_name}(FragmentInput input [[stage_in]]) {{\n")
        self.out.write(f"    // TODO: Implement fragment shader logic\n")
        self.out.write(f"    return float4(1.0, 0.0, 0.0, 1.0);\n")
        self.out.write(f"}}\n")
    
    def _write_compute_entry_point(self) -> None:
        """Write compute shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"kernel void {func_name}(\n")
        self.out.write(f"    uint3 thread_position_in_grid [[thread_position_in_grid]],\n")
        self.out.write(f"    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n")
        self.out.write(f"    uint3 threadgroup_size [[threads_per_threadgroup]]) {{\n")
        self.out.write(f"    // TODO: Implement compute shader logic\n")
        self.out.write(f"}}\n")
    
    def _write_mesh_entry_point(self) -> None:
        """Write mesh shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"kernel void {func_name}(\n")
        self.out.write(f"    uint3 thread_position_in_grid [[thread_position_in_grid]],\n")
        self.out.write(f"    device meshlet_t* meshlets [[buffer(0)]],\n")
        self.out.write(f"    device primitive_t* primitives [[buffer(1)]],\n")
        self.out.write(f"    device vertex_t* vertices [[buffer(2)]],\n")
        self.out.write(f"    device uint* indices [[buffer(3)]],\n")
        self.out.write(f"    device atomic_uint* primitiveCounter [[buffer(4)]]) {{\n")
        self.out.write(f"    // TODO: Implement mesh shader logic\n")
        self.out.write(f"}}\n")
    
    def _write_task_entry_point(self) -> None:
        """Write task shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"kernel void {func_name}(\n")
        self.out.write(f"    uint3 thread_position_in_grid [[thread_position_in_grid]],\n")
        self.out.write(f"    device meshlet_t* meshlets [[buffer(0)]]) {{\n")
        self.out.write(f"    // TODO: Implement task shader logic\n")
        self.out.write(f"}}\n")
    
    def _write_function(self, function: Any) -> None:
        """Write a single function."""
        func_name = self._get_function_name(function)
        
        # Write function signature
        return_type = self._type_to_msl(function.result.ty if function.result else None)
        self.out.write(f"{return_type} {func_name}(")
        
        # Write arguments
        for i, arg in enumerate(function.arguments):
            arg_name = f"arg_{i}"
            type_name = self._type_to_msl(arg.ty)
            self.out.write(f"{type_name} {arg_name}")
            if i < len(function.arguments) - 1:
                self.out.write(", ")
        
        self.out.write(") {\n")
        
        # Write function body
        if hasattr(function, 'body') and function.body:
            self._write_function_body(function.body, 1)
        
        self.out.write("}\n")
    
    def _write_function_body(self, body: Any, indent_level: int) -> None:
        """Write function body statements."""
        indent = "    " * indent_level
        
        # Placeholder function body implementation
        for stmt in body:
            stmt_type = type(stmt).__name__
            if stmt_type == "LocalVariable":
                var_name = f"local_{id(stmt)}"
                self.out.write(f"{indent}auto {var_name}")
                if hasattr(stmt, 'ty') and stmt.ty:
                    self.out.write(f" = {self._type_to_msl(stmt.ty)}")
                if hasattr(stmt, 'init') and stmt.init:
                    self.out.write(f" = {self._expression_to_msl(stmt.init)}")
                self.out.write(";\n")
            elif stmt_type == "Store":
                if hasattr(stmt, 'pointer') and hasattr(stmt, 'value'):
                    self.out.write(f"{indent}{self._expression_to_msl(stmt.pointer)} = {self._expression_to_msl(stmt.value)};\n")
            elif stmt_type == "Return":
                if hasattr(stmt, 'value') and stmt.value:
                    self.out.write(f"{indent}return {self._expression_to_msl(stmt.value)};\n")
                else:
                    self.out.write(f"{indent}return;\n")
            else:
                self.out.write(f"{indent}// TODO: Implement {stmt_type}\n")
    
    def _expression_to_msl(self, expr: Any) -> str:
        """Convert an expression to MSL string representation."""
        expr_type = type(expr).__name__
        
        if expr_type == "Literal":
            return str(expr.value)
        elif expr_type == "Variable":
            return self._get_variable_name(expr)
        elif expr_type == "BinaryOperation":
            left = self._expression_to_msl(expr.left)
            right = self._expression_to_msl(expr.right)
            op = self._binary_op_to_msl(expr.op)
            return f"({left} {op} {right})"
        elif expr_type == "Call":
            func_name = self._get_function_name(expr.function)
            args = [self._expression_to_msl(arg) for arg in expr.arguments]
            return f"{func_name}({', '.join(args)})"
        else:
            return f"/* TODO: {expr_type} */"
    
    def _binary_op_to_msl(self, op: Any) -> str:
        """Convert binary operation to MSL operator."""
        op_map = {
            "Add": "+",
            "Subtract": "-",
            "Multiply": "*",
            "Divide": "/",
            "Modulo": "%",
            "Equal": "==",
            "NotEqual": "!=",
            "Less": "<",
            "LessEqual": "<=",
            "Greater": ">",
            "GreaterEqual": ">=",
            "And": "&&",
            "Or": "||",
            "LogicalAnd": "&&",
            "LogicalOr": "||"
        }
        return op_map.get(str(op), "?")
    
    def _type_to_msl(self, ty: Any) -> str:
        """Convert a type to MSL string representation."""
        if ty is None:
            return "void"
        
        if hasattr(ty, 'inner'):
            inner = ty.inner
        else:
            inner = ty
        
        if hasattr(inner, 'ty'):
            inner = inner.ty
        
        if str(inner).startswith("Scalar."):
            scalar_type = str(inner).split('.')[1]
            type_map = {
                "F16": "half",
                "F32": "float",
                "F64": "double",
                "I32": "int",
                "U32": "uint",
                "Bool": "bool"
            }
            return type_map.get(scalar_type, str(inner).lower())
        elif str(inner).startswith("Vector"):
            scalar = self._type_to_msl(inner.scalar) if hasattr(inner, 'scalar') else "float"
            size = getattr(inner, 'size', 2)
            if scalar == "float":
                return f"float{size}"
            elif scalar == "half":
                return f"half{size}"
            elif scalar == "int":
                return f"int{size}"
            elif scalar == "uint":
                return f"uint{size}"
            elif scalar == "bool":
                return f"bool{size}"
            else:
                return f"{scalar}{size}"
        elif str(inner).startswith("Matrix"):
            scalar = self._type_to_msl(inner.scalar) if hasattr(inner, 'scalar') else "float"
            columns = getattr(inner, 'columns', 2)
            rows = getattr(inner, 'rows', 2)
            if scalar == "float":
                return f"float{columns}x{rows}"
            elif scalar == "half":
                return f"half{columns}x{rows}"
            else:
                return f"{scalar}{columns}x{rows}"
        elif str(inner).startswith("Array"):
            element = self._type_to_msl(inner.element) if hasattr(inner, 'element') else "float"
            count = getattr(inner, 'size', None)
            if count is None:
                return f"{element}[]"
            else:
                return f"{element}[{count}]"
        else:
            return str(inner).lower()
    
    def _builtin_to_attribute(self, builtin: Any) -> Optional[str]:
        """Convert builtin to MSL attribute."""
        builtin_map = {
            "Position": "position",
            "VertexIndex": "vertex_id",
            "InstanceIndex": "instance_id",
            "FrontFacing": "front_facing",
            "FragCoord": "position",
            "PointSize": "point_size",
            "ClipDistance": "clip_distance",
            "CullDistance": "cull_distance",
            "PrimitiveIndex": "primitive_id",
            "SampleIndex": "sample_id",
            "Layer": "layer",
            "ViewportIndex": "viewport_index",
            "ThreadPositionInGrid": "thread_position_in_grid",
            "ThreadGroupPositionInGrid": "threadgroup_position_in_grid",
            "ThreadGroupSize": "threadgroup_size",
            "GridSize": "grid_size"
        }
        return builtin_map.get(str(builtin))
    
    def _get_variable_name(self, var: Any) -> str:
        """Get the MSL variable name."""
        if hasattr(var, 'name') and var.name:
            return self.namer.get_name(var.name)
        return f"var_{id(var)}"
    
    def _get_function_name(self, func: Any) -> str:
        """Get the MSL function name."""
        if hasattr(func, 'name') and func.name:
            return self.namer.get_name(func.name)
        return f"func_{id(func)}"
    
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


def write_string(module: Any, info: Any, options: Options, 
                entry_point: str, shader_stage: ShaderStage) -> str:
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