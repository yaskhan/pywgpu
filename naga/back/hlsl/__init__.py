"""
Backend for HLSL (High-Level Shading Language).

This module contains the writer implementation for converting Naga IR
to HLSL code with support for various shader model versions.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntEnum, IntFlag
import io

from ...error import ShaderError


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
    """Shader stage enumeration."""
    Vertex = 0
    Hull = 1
    Domain = 2
    Geometry = 3
    Pixel = 4
    Compute = 5
    RayGeneration = 6
    Intersection = 7
    AnyHit = 8
    ClosestHit = 9
    Miss = 10
    Callable = 11
    Task = 12
    Mesh = 13


class Options:
    """HLSL writer options."""
    
    def __init__(self, shader_model: ShaderModel):
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
    
    def supports_samplers(self) -> bool:
        """Check if shader model supports sampler state objects."""
        return self.shader_model >= ShaderModel.SM_5_0


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
        self.constant_buffers: Dict[str, Any] = {}
        self.structured_buffers: Dict[str, Any] = {}


class Writer:
    """
    Writer for converting Naga IR modules to HLSL code.
    
    Maintains internal state to output a Module into HLSL format.
    """
    
    def __init__(self, out: Union[str, io.StringIO], module: Any, info: Any, 
                 options: Options, binding_map: Optional[BindingMap] = None):
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
        self.required_features: Set[str] = set()
        
        # Entry point state
        self.entry_point = None
        self.entry_point_stage = None
        
        # Type information cache
        self.type_cache: Dict[str, str] = {}
    
    def write(self, entry_point: str, shader_stage: ShaderStage) -> ReflectionInfo:
        """
        Write the complete module to HLSL.
        
        Args:
            entry_point: Name of the entry point function
            shader_stage: Shader stage type
            
        Returns:
            Reflection information about the generated shader
            
        Raises:
            ShaderError: If writing fails
        """
        try:
            self.entry_point = entry_point
            self.entry_point_stage = shader_stage
            
            self._initialize_names()
            self._find_entry_point()
            self._collect_required_features()
            
            # Write HLSL version and includes
            self._write_header()
            
            # Write type definitions
            self._write_type_definitions()
            
            # Write sampler declarations
            if self.options.supports_samplers():
                self._write_samplers()
            
            # Write texture declarations
            self._write_textures()
            
            # Write constant buffers
            self._write_constant_buffers()
            
            # Write structured buffers
            self._write_structured_buffers()
            
            # Write functions
            self._write_functions()
            
            # Write entry point
            self._write_entry_point()
            
            return self.reflection_info
            
        except Exception as e:
            raise ShaderError(f"HLSL writing failed: {e}") from e
    
    def _initialize_names(self) -> None:
        """Initialize name mappings for module elements."""
        self.names.clear()
        self.namer.reset()
        
        # Reserve HLSL keywords and built-in names
        reserved_keywords = [
            "float", "double", "int", "uint", "bool", "half",
            "float2", "float3", "float4", "double2", "double3", "double4",
            "int2", "int3", "int4", "uint2", "uint3", "uint4",
            "bool2", "bool3", "bool4", "half2", "half3", "half4",
            "float2x2", "float3x3", "float4x4", "float2x3", "float3x2",
            "float2x4", "float4x2", "float3x4", "float4x3",
            "matrix", "vector", "sampler", "sampler1D", "sampler2D", "sampler3D",
            "samplerCUBE", "sampler_state", "Texture1D", "Texture2D", "Texture3D",
            "TextureCube", "Texture1DArray", "Texture2DArray", "TextureCubeArray",
            "Texture2DMS", "Texture2DMSArray",
            "RWTexture1D", "RWTexture2D", "RWTexture3D", "RWTexture1DArray", "RWTexture2DArray", "RWTexture3D",
            "RWTextureCubeArray", "RWTexture2DMS", "RWTexture2DMSArray",
            "Buffer", "StructuredBuffer", "ByteAddressBuffer", "AppendStructuredBuffer",
            "ConsumeStructuredBuffer", "RWBuffer", "RWStructuredBuffer", "RWByteAddressBuffer",
            "RWAppendStructuredBuffer", "RWConsumeStructuredBuffer",
            "SamplerState", "SamplerComparisonState",
            "BlendState", "DepthStencilState", "RasterizerState",
            "VertexShader", "PixelShader", "HullShader", "DomainShader",
            "GeometryShader", "ComputeShader", "RaytracingShader",
            "BlendState", "DepthStencilState", "RasterizerState",
            "SV_Target", "SV_Position", "SV_VertexID", "SV_InstanceID",
            "SV_PrimitiveID", "SV_Depth", "SV_DepthLessEqual", "SV_DepthGreaterEqual",
            "SV_ClipDistance", "SV_CullDistance", "SV_Coverage",
            "SV_SampleIndex", "SV_IsFrontFace", "SV_DispatchThreadID",
            "SV_GroupID", "SV_GroupIndex", "SV_GroupThreadID",
            "SV_InnerCoverage", "SV_RenderTargetArrayIndex", "SV_ViewportArrayIndex",
            "SV_ViewID", "SV_BarycentricCoordinates", "SV_ShadingRate",
            "SV_ClipDistance0", "SV_ClipDistance1", "SV_ClipDistance2", "SV_ClipDistance3",
            "SV_ClipDistance4", "SV_ClipDistance5", "SV_ClipDistance6", "SV_ClipDistance7",
            "SV_CullDistance0", "SV_CullDistance1", "SV_CullDistance2", "SV_CullDistance3",
            "SV_CullDistance4", "SV_CullDistance5", "SV_CullDistance6", "SV_CullDistance7"
        ]
        
        for keyword in reserved_keywords:
            self.namer.reserve_name(keyword)
    
    def _find_entry_point(self) -> None:
        """Find the entry point in the module."""
        self.entry_point = None
        for ep in self.module.entry_points:
            if ep.name == self.entry_point and ep.stage == self.entry_point_stage:
                self.entry_point = ep
                break
    
    def _collect_required_features(self) -> None:
        """Analyze module and collect required HLSL features."""
        # Placeholder implementation
        # Would analyze module types, expressions, etc.
        pass
    
    def _write_header(self) -> None:
        """Write HLSL file header."""
        # Write shader model target
        if self.entry_point_stage == ShaderStage.Compute:
            self.out.write(f"// Compute Shader - Target {self.options.shader_model}\n")
        elif self.entry_point_stage == ShaderStage.Vertex:
            self.out.write(f"// Vertex Shader - Target {self.options.shader_model}\n")
        elif self.entry_point_stage == ShaderStage.Pixel:
            self.out.write(f"// Pixel Shader - Target {self.options.shader_model}\n")
        elif self.entry_point_stage == ShaderStage.Mesh:
            self.out.write(f"// Mesh Shader - Target {self.options.shader_model}\n")
        elif self.entry_point_stage == ShaderStage.Task:
            self.out.write(f"// Task Shader - Target {self.options.shader_model}\n")
        else:
            self.out.write(f"// Shader - Target {self.options.shader_model}\n")
        
        self.out.write("\n")
        
        # Write cbuffer includes if needed
        self.out.write("cbuffer RootConstant : register(b0)\n{\n    uint _start;\n}\n\n")
    
    def _write_type_definitions(self) -> None:
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
            type_name = self._type_to_hlsl(member.ty)
            self.out.write(f"    {type_name} {member_name}")
            # Add matrix packing annotations if needed
            if hasattr(member, 'binding') and member.binding and hasattr(member.binding, 'location'):
                self.out.write(f" : TEXCOORD{member.binding.location}")
            self.out.write(";\n")
        
        self.out.write("};\n")
    
    def _write_samplers(self) -> None:
        """Write sampler state declarations."""
        # Placeholder for sampler declarations
        # Would analyze module for sampler usage
        pass
    
    def _write_textures(self) -> None:
        """Write texture declarations."""
        # Placeholder for texture declarations
        # Would analyze module for texture usage
        pass
    
    def _write_constant_buffers(self) -> None:
        """Write constant buffer declarations."""
        # Placeholder for constant buffer declarations
        # Would analyze module for uniform buffer usage
        pass
    
    def _write_structured_buffers(self) -> None:
        """Write structured buffer declarations."""
        # Placeholder for structured buffer declarations
        # Would analyze module for storage buffer usage
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
        if self.entry_point_stage == ShaderStage.Vertex:
            self._write_vertex_entry_point()
        elif self.entry_point_stage == ShaderStage.Pixel:
            self._write_pixel_entry_point()
        elif self.entry_point_stage == ShaderStage.Compute:
            self._write_compute_entry_point()
        elif self.entry_point_stage == ShaderStage.Mesh:
            self._write_mesh_entry_point()
        elif self.entry_point_stage == ShaderStage.Task:
            self._write_task_entry_point()
        else:
            self._write_generic_entry_point()
    
    def _write_vertex_entry_point(self) -> None:
        """Write vertex shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"struct VSInput {{\n")
        
        # Write input parameters
        if hasattr(self.entry_point, 'function'):
            for i, arg in enumerate(self.entry_point.function.arguments):
                param_name = f"param{i}"
                param_type = self._type_to_hlsl(arg.ty)
                self.out.write(f"    {param_type} {param_name}")
                if hasattr(arg, 'binding') and arg.binding:
                    if hasattr(arg.binding, 'location'):
                        self.out.write(f" : TEXCOORD{arg.binding.location}")
                    elif hasattr(arg.binding, 'builtin'):
                        builtin_name = self._builtin_to_semantic(arg.binding.builtin)
                        self.out.write(f" : {builtin_name}")
                self.out.write(";\n")
        
        self.out.write("};\n\n")
        
        self.out.write(f"struct VSOutput {{\n")
        self.out.write(f"    float4 position : SV_Position;\n")
        self.out.write(f"    // Add other output parameters as needed\n")
        self.out.write(f"}};\n\n")
        
        self.out.write(f"VSOutput {func_name}(VSInput input) {{\n")
        self.out.write(f"    VSOutput output;\n")
        self.out.write(f"    // TODO: Implement vertex shader logic\n")
        self.out.write(f"    output.position = float4(0.0, 0.0, 0.0, 1.0);\n")
        self.out.write(f"    return output;\n")
        self.out.write(f"}}\n")
    
    def _write_pixel_entry_point(self) -> None:
        """Write pixel shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"struct PSInput {{\n")
        self.out.write(f"    float4 position : SV_Position;\n")
        self.out.write(f"    // Add other input parameters as needed\n")
        self.out.write(f"}};\n\n")
        
        self.out.write(f"struct PSOutput {{\n")
        self.out.write(f"    float4 color : SV_Target;\n")
        self.out.write(f"}};\n\n")
        
        self.out.write(f"PSOutput {func_name}(PSInput input) {{\n")
        self.out.write(f"    PSOutput output;\n")
        self.out.write(f"    // TODO: Implement pixel shader logic\n")
        self.out.write(f"    output.color = float4(1.0, 0.0, 0.0, 1.0);\n")
        self.out.write(f"    return output;\n")
        self.out.write(f"}}\n")
    
    def _write_compute_entry_point(self) -> None:
        """Write compute shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"[numthreads(1, 1, 1)]\n")
        self.out.write(f"void {func_name}(uint3 dispatchThreadID : SV_DispatchThreadID) {{\n")
        self.out.write(f"    // TODO: Implement compute shader logic\n")
        self.out.write(f"}}\n")
    
    def _write_mesh_entry_point(self) -> None:
        """Write mesh shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        if self.options.supports_mesh_shaders():
            self.out.write(f"[outputtopology(\"triangle\")]\n")
            self.out.write(f"[numthreads(1, 1, 1)]\n")
            self.out.write(f"void {func_name}(")
            # Add mesh shader specific parameters
            self.out.write("uint3 dispatchThreadID : SV_DispatchThreadID) {\n")
            self.out.write(f"    // TODO: Implement mesh shader logic\n")
            self.out.write(f"}}\n")
        else:
            self._write_generic_entry_point()
    
    def _write_task_entry_point(self) -> None:
        """Write task shader entry point."""
        func_name = self._get_function_name(self.entry_point)
        if self.options.supports_mesh_shaders():
            self.out.write(f"[numthreads(1, 1, 1)]\n")
            self.out.write(f"void {func_name}(uint3 dispatchThreadID : SV_DispatchThreadID) {{\n")
            self.out.write(f"    // TODO: Implement task shader logic\n")
            self.out.write(f"}}\n")
        else:
            self._write_generic_entry_point()
    
    def _write_generic_entry_point(self) -> None:
        """Write a generic entry point."""
        func_name = self._get_function_name(self.entry_point)
        self.out.write(f"void {func_name}() {{\n")
        self.out.write(f"    // TODO: Implement shader logic\n")
        self.out.write(f"}}\n")
    
    def _write_function(self, function: Any) -> None:
        """Write a single function."""
        func_name = self._get_function_name(function)
        
        # Write function signature
        return_type = self._type_to_hlsl(function.result.ty if function.result else None)
        self.out.write(f"{return_type} {func_name}(")
        
        # Write arguments
        for i, arg in enumerate(function.arguments):
            arg_name = f"arg_{i}"
            type_name = self._type_to_hlsl(arg.ty)
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
                    self.out.write(f" = {self._type_to_hlsl(stmt.ty)}")
                if hasattr(stmt, 'init') and stmt.init:
                    self.out.write(f" = {self._expression_to_hlsl(stmt.init)}")
                self.out.write(";\n")
            elif stmt_type == "Store":
                if hasattr(stmt, 'pointer') and hasattr(stmt, 'value'):
                    self.out.write(f"{indent}{self._expression_to_hlsl(stmt.pointer)} = {self._expression_to_hlsl(stmt.value)};\n")
            elif stmt_type == "Return":
                if hasattr(stmt, 'value') and stmt.value:
                    self.out.write(f"{indent}return {self._expression_to_hlsl(stmt.value)};\n")
                else:
                    self.out.write(f"{indent}return;\n")
            else:
                self.out.write(f"{indent}// TODO: Implement {stmt_type}\n")
    
    def _expression_to_hlsl(self, expr: Any) -> str:
        """Convert an expression to HLSL string representation."""
        expr_type = type(expr).__name__
        
        if expr_type == "Literal":
            return str(expr.value)
        elif expr_type == "Variable":
            return self._get_variable_name(expr)
        elif expr_type == "BinaryOperation":
            left = self._expression_to_hlsl(expr.left)
            right = self._expression_to_hlsl(expr.right)
            op = self._binary_op_to_hlsl(expr.op)
            return f"({left} {op} {right})"
        elif expr_type == "Call":
            func_name = self._get_function_name(expr.function)
            args = [self._expression_to_hlsl(arg) for arg in expr.arguments]
            return f"{func_name}({', '.join(args)})"
        else:
            return f"/* TODO: {expr_type} */"
    
    def _binary_op_to_hlsl(self, op: Any) -> str:
        """Convert binary operation to HLSL operator."""
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
    
    def _type_to_hlsl(self, ty: Any) -> str:
        """Convert a type to HLSL string representation."""
        if ty is None:
            return "void"
        
        cache_key = str(ty)
        if cache_key in self.type_cache:
            return self.type_cache[cache_key]
        
        if hasattr(ty, 'inner'):
            inner = ty.inner
        else:
            inner = ty
        
        if hasattr(inner, 'ty'):
            inner = inner.ty
        
        if str(inner).startswith("Scalar."):
            scalar_type = str(inner).split('.')[1]
            type_map = {
                "F16": "float",
                "F32": "float",
                "F64": "double",
                "I32": "int",
                "U32": "uint",
                "Bool": "bool"
            }
            result = type_map.get(scalar_type, str(inner).lower())
        elif str(inner).startswith("Vector"):
            scalar = self._type_to_hlsl(inner.scalar) if hasattr(inner, 'scalar') else "float"
            size = getattr(inner, 'size', 2)
            if scalar == "float":
                result = f"float{size}"
            elif scalar == "int":
                result = f"int{size}"
            elif scalar == "uint":
                result = f"uint{size}"
            elif scalar == "bool":
                result = f"bool{size}"
            else:
                result = f"{scalar}{size}"
        elif str(inner).startswith("Matrix"):
            scalar = self._type_to_hlsl(inner.scalar) if hasattr(inner, 'scalar') else "float"
            columns = getattr(inner, 'columns', 2)
            rows = getattr(inner, 'rows', 2)
            if scalar == "float":
                result = f"float{columns}x{rows}"
            else:
                result = f"{scalar}{columns}x{rows}"
        elif str(inner).startswith("Array"):
            element = self._type_to_hlsl(inner.element) if hasattr(inner, 'element') else "float"
            count = getattr(inner, 'size', None)
            if count is None:
                result = f"{element}[]"
            else:
                result = f"{element}[{count}]"
        else:
            result = str(inner).lower()
        
        self.type_cache[cache_key] = result
        return result
    
    def _builtin_to_semantic(self, builtin: Any) -> str:
        """Convert builtin to HLSL semantic."""
        builtin_map = {
            "Position": "SV_Position",
            "VertexIndex": "SV_VertexID",
            "InstanceIndex": "SV_InstanceID",
            "FrontFacing": "SV_IsFrontFace",
            "FragCoord": "SV_Position",
            "PointSize": "PSIZE",
            "ClipDistance": "SV_ClipDistance",
            "CullDistance": "SV_CullDistance",
            "PrimitiveIndex": "SV_PrimitiveID",
            "SampleIndex": "SV_SampleIndex",
            "Layer": "SV_RenderTargetArrayIndex",
            "ViewportIndex": "SV_ViewportArrayIndex"
        }
        return builtin_map.get(str(builtin), "TEXCOORD0")
    
    def _get_variable_name(self, var: Any) -> str:
        """Get the HLSL variable name."""
        if hasattr(var, 'name') and var.name:
            return self.namer.get_name(var.name)
        return f"var_{id(var)}"
    
    def _get_function_name(self, func: Any) -> str:
        """Get the HLSL function name."""
        if hasattr(func, 'name') and func.name:
            return self.namer.get_name(func.name)
        return f"func_{id(func)}"
    
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


def write_string(module: Any, info: Any, options: Options, 
                entry_point: str, shader_stage: ShaderStage,
                binding_map: Optional[BindingMap] = None) -> str:
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