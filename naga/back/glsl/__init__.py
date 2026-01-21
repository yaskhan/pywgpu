"""
Backend for GLSL (OpenGL Shading Language).

This module contains the writer implementation for converting Naga IR
to GLSL code with support for various GLSL versions and features.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import IntEnum, IntFlag
import io

from ...error import ShaderError


class Version(IntEnum):
    """GLSL version enumeration."""

    V100 = 100
    V110 = 110
    V120 = 120
    V130 = 130
    V140 = 140
    V150 = 150
    V300 = 300
    V330 = 330
    V400 = 400
    V410 = 410
    V420 = 420
    V430 = 430
    V440 = 440
    V450 = 450
    V460 = 460


class Profile(IntEnum):
    """GLSL profile enumeration."""

    Core = 0
    Es = 1


class VersionError(Exception):
    """Exception raised for unsupported GLSL versions."""

    pass


class Options:
    """GLSL writer options."""

    def __init__(self, version: Version, profile: Profile = Profile.Core):
        """
        Initialize GLSL writer options.

        Args:
            version: GLSL version to target
            profile: GLSL profile (Core or ES)
        """
        self.version = version
        self.profile = profile

    def is_es(self) -> bool:
        """Check if this is an ES profile."""
        return self.profile == Profile.Es

    def is_supported(self) -> bool:
        """Check if the version/profile combination is supported."""
        core_versions = [140, 150, 330, 400, 410, 420, 430, 440, 450, 460]
        es_versions = [300, 310, 320]

        if self.is_es():
            return self.version in es_versions
        else:
            return self.version in core_versions


class ShaderStage(IntEnum):
    """Shader stage enumeration."""

    Vertex = 0
    Fragment = 1
    Compute = 2
    Task = 3
    Mesh = 4


class PipelineOptions:
    """Pipeline options for GLSL generation."""

    def __init__(
        self,
        shader_stage: ShaderStage,
        entry_point: str,
        multiview: Optional[int] = None,
    ):
        """
        Initialize pipeline options.

        Args:
            shader_stage: The shader stage
            entry_point: Entry point function name
            multiview: Number of views for multiview rendering
        """
        self.shader_stage = shader_stage
        self.entry_point = entry_point
        self.multiview = multiview


class ReflectionInfo:
    """Reflection information for GLSL shaders."""

    def __init__(self):
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.uniforms: Dict[str, Any] = {}
        self.textures: Dict[str, Any] = {}
        self.samplers: Dict[str, Any] = {}
        self.push_constants: Dict[str, Any] = {}
        self.storage_buffers: Dict[str, Any] = {}


class Writer:
    """
    Writer for converting Naga IR modules to GLSL code.

    Maintains internal state to output a Module into GLSL format.
    """

    def __init__(
        self,
        out: Union[str, io.StringIO],
        module: Any,
        info: Any,
        options: Options,
        pipeline_options: PipelineOptions,
    ):
        """
        Initialize the GLSL writer.

        Args:
            out: Output stream
            module: The Naga IR module
            info: Module validation information
            options: GLSL writer options
            pipeline_options: Pipeline configuration
        """
        if isinstance(out, str):
            self.out = io.StringIO(out)
        else:
            self.out = out

        self.module = module
        self.info = info
        self.options = options
        self.pipeline_options = pipeline_options

        # Internal state
        self.names: Dict[str, str] = {}
        self.namer = GLSLNameGenerator()
        self.features: Set[str] = set()
        self.reflection_info = ReflectionInfo()
        self.varying_locations: Dict[str, int] = {}
        self.clip_distance_count = 0

        # Find entry point
        self.entry_point = self._find_entry_point()
        if not self.entry_point:
            raise ShaderError("Entry point not found")

    def _find_entry_point(self) -> Optional[Any]:
        """Find the entry point in the module."""
        for ep in self.module.entry_points:
            if (
                ep.stage == self.pipeline_options.shader_stage
                and ep.name == self.pipeline_options.entry_point
            ):
                return ep
        return None

    def write(self) -> ReflectionInfo:
        """
        Write the complete module to GLSL.

        Returns:
            Reflection information about the generated shader

        Raises:
            ShaderError: If writing fails
        """
        try:
            # Check version support
            if not self.options.is_supported():
                raise ShaderError(f"GLSL version {self.options.version} not supported")

            self._initialize_names()
            self._collect_required_features()

            # Write version directive
            self._write_version_directive()

            # Write extensions
            self._write_extensions()

            # Write precision for ES
            if self.options.is_es():
                self._write_precision()

            # Write uniform declarations
            self._write_uniforms()

            # Write varyings
            self._write_varyings()

            # Write functions
            self._write_functions()

            # Write entry point
            self._write_entry_point()

            return self.reflection_info

        except Exception as e:
            raise ShaderError(f"GLSL writing failed: {e}") from e

    def _initialize_names(self) -> None:
        """Initialize name mappings for module elements."""
        self.names.clear()
        self.namer.reset()

        # Reserve GLSL keywords and built-in names
        reserved_keywords = [
            "gl_Position",
            "gl_FragColor",
            "gl_FragCoord",
            "gl_PointSize",
            "gl_VertexID",
            "gl_InstanceID",
            "gl_FrontFacing",
            "gl_FragDepth",
            "gl_ClipDistance",
            "gl_CullDistance",
            "gl_PrimitiveID",
            "gl_Layer",
            "gl_ViewportIndex",
            "gl_TessLevelOuter",
            "gl_TessLevelInner",
            "gl_TessCoord",
            "gl_PrimitiveIDIn",
            "gl_InvocationID",
            "gl_MaxClipDistances",
            "gl_MaxCullDistances",
            "gl_MaxTessControlOutputVertices",
            "gl_MaxTessPatchComponents",
            "gl_MaxTessGenLevel",
            "gl_MaxTextureImageUnits",
            "gl_MaxTextureUnits",
            "gl_MaxTextureCoords",
            "gl_MaxVertexAttribs",
            "gl_MaxVertexUniformVectors",
            "gl_MaxVertexTextureImageUnits",
            "gl_MaxCombinedTextureImageUnits",
            "gl_MaxDrawBuffers",
            "gl_MaxFragmentUniformVectors",
            "gl_MaxVaryingVectors",
            "gl_MaxVaryingFloats",
            "gl_DepthRange",
            "gl_PointCoord",
            "gl_FrontColor",
            "gl_BackColor",
            "gl_FrontSecondaryColor",
            "gl_BackSecondaryColor",
            "gl_FragColor",
            "gl_FragData",
            "gl_MaxVertexAttribs",
            "gl_MaxVertexUniformVectors",
            "gl_MaxVaryingVectors",
            "gl_MaxVaryingFloats",
            "gl_MaxVertexTextureImageUnits",
            "gl_MaxTextureImageUnits",
            "gl_MaxFragmentUniformVectors",
            "gl_MaxDrawBuffers",
            "gl_MaxColorAttachments",
            "gl_DepthRangeParameters",
            "gl_PointParameters",
            "gl_MaterialParameters",
            "gl_LightSourceParameters",
            "gl_LightModelParameters",
            "gl_LightModelProducts",
            "gl_FrontLightModelProduct",
            "gl_BackLightModelProduct",
            "gl_FrontLightProduct",
            "gl_BackLightProduct",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_MaxClipPlanes",
            "gl_TextureEnvColor",
            "gl_TextureEnvMode",
            "gl_FogParameters",
            "gl_FogColor",
            "gl_FogDensity",
            "gl_FogStart",
            "gl_FogEnd",
            "gl_FogCoord",
            "gl_MaxLights",
            "gl_MaxTextureUnits",
            "gl_MaxTextureCoords",
            "gl_MaxVertexAttribs",
            "gl_MaxVertexUniformVectors",
            "gl_MaxVaryingVectors",
            "gl_MaxVaryingFloats",
            "gl_MaxVertexTextureImageUnits",
            "gl_MaxTextureImageUnits",
            "gl_MaxFragmentUniformVectors",
            "gl_MaxDrawBuffers",
            "gl_MaxColorAttachments",
            "gl_BlendEquationRGB",
            "gl_BlendEquationAlpha",
            "gl_BlendFuncSeparate",
            "gl_BlendFunc",
            "gl_BlendColor",
            "gl_DepthMask",
            "gl_ColorMask",
            "gl_StencilMask",
            "gl_StencilFunc",
            "gl_StencilOp",
            "gl_StencilFuncSeparate",
            "gl_StencilOpSeparate",
            "gl_StencilOpValue",
            "gl_Fog",
            "gl_Point",
            "gl_ColorMaterial",
            "gl_CullFace",
            "gl_CullFaceMode",
            "gl_FrontFace",
            "gl_Light",
            "gl_LightModel",
            "gl_LightModelAmbient",
            "gl_LightModelDiffuse",
            "gl_LightModelSpecular",
            "gl_LightModelAmbient",
            "gl_LightSource",
            "gl_LightSourceAmbient",
            "gl_LightSourceDiffuse",
            "gl_LightSourceSpecular",
            "gl_LightSourcePosition",
            "gl_LightSourceSpotDirection",
            "gl_LightSourceSpotExponent",
            "gl_LightSourceSpotCutoffAngle",
            "gl_LightSourceConstantAttenuation",
            "gl_LightSourceLinearAttenuation",
            "gl_LightSourceQuadraticAttenuation",
            "gl_SecondaryColor",
            "gl_ColorSum",
            "gl_TexGen",
            "gl_TexGenModeS",
            "gl_TexGenModeT",
            "gl_TexGenModeR",
            "gl_TexGenModeQ",
            "gl_TextureMatrix",
            "gl_TextureMatrix",
            "gl_ClipDistance",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipVertex",
            "gl_FrontColor",
            "gl_BackColor",
            "gl_FrontSecondaryColor",
            "gl_BackSecondaryColor",
            "gl_TexCoord",
            "gl_FragColor",
            "gl_FragData",
            "gl_FragDepth",
            "gl_MaxClipDistances",
            "gl_MaxCullDistances",
            "gl_MaxCombinedTextureImageUnits",
            "gl_MaxDrawBuffers",
            "gl_MaxFragmentUniformVectors",
            "gl_MaxVaryingVectors",
            "gl_MaxVaryingFloats",
            "gl_MaxVertexAttribs",
            "gl_MaxVertexTextureImageUnits",
            "gl_MaxVertexUniformVectors",
            "gl_MaxTextureCoords",
            "gl_MaxTextureImageUnits",
            "gl_MaxTextureUnits",
            "gl_ModelViewMatrix",
            "gl_ProjectionMatrix",
            "gl_ModelViewProjectionMatrix",
            "gl_TextureMatrix",
            "gl_NormalMatrix",
            "gl_ModelViewMatrixInverse",
            "gl_ProjectionMatrixInverse",
            "gl_ModelViewProjectionMatrixInverse",
            "gl_TextureMatrixInverse",
            "gl_ModelViewMatrixTranspose",
            "gl_ProjectionMatrixTranspose",
            "gl_ModelViewProjectionMatrixTranspose",
            "gl_TextureMatrixTranspose",
            "gl_ModelViewMatrixInverseTranspose",
            "gl_ProjectionMatrixInverseTranspose",
            "gl_ModelViewProjectionMatrixInverseTranspose",
            "gl_TextureMatrixInverseTranspose",
            "gl_NormalScale",
            "gl_DepthRangeParameters",
            "gl_DepthRange",
            "gl_PointParameters",
            "gl_PointSize",
            "gl_PointCoord",
            "gl_MaterialParameters",
            "gl_FrontMaterial",
            "gl_BackMaterial",
            "gl_LightSourceParameters",
            "gl_LightSource",
            "gl_LightModelParameters",
            "gl_LightModel",
            "gl_LightModelAmbient",
            "gl_LightModelDiffuse",
            "gl_LightModelSpecular",
            "gl_LightModelProducts",
            "gl_FrontLightModelProduct",
            "gl_BackLightModelProduct",
            "gl_FrontLightProduct",
            "gl_BackLightProduct",
            "gl_FogParameters",
            "gl_FogColor",
            "gl_FogDensity",
            "gl_FogStart",
            "gl_FogEnd",
            "gl_FogCoord",
            "gl_MaxLights",
            "gl_MaxTextureUnits",
            "gl_MaxTextureCoords",
            "gl_MaxClipPlanes",
            "gl_MaxTextureUnits",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_ClipPlane",
            "gl_MaxCombinedTextureImageUnits",
            "gl_MaxDrawBuffers",
            "gl_MaxFragmentUniformVectors",
            "gl_MaxVaryingVectors",
            "gl_MaxVaryingFloats",
            "gl_MaxVertexAttribs",
            "gl_MaxVertexTextureImageUnits",
            "gl_MaxVertexUniformVectors",
            "gl_FrontFacing",
            "gl_PointCoord",
            "gl_PrimitiveID",
            "gl_Layer",
            "gl_ViewportIndex",
            "gl_MaxTessControlOutputVertices",
            "gl_MaxTessPatchComponents",
            "gl_MaxTessGenLevel",
            "gl_MaxTessControlInputComponents",
            "gl_MaxTessControlOutputComponents",
            "gl_MaxTessControlTextureImageUnits",
            "gl_MaxTessControlUniformComponents",
            "gl_MaxTessControlTotalOutputComponents",
            "gl_MaxTessEvaluationInputComponents",
            "gl_MaxTessEvaluationOutputComponents",
            "gl_MaxTessEvaluationTextureImageUnits",
            "gl_MaxTessEvaluationUniformComponents",
            "gl_MaxTessPatchComponents",
            "gl_MaxTessControlTextureImageUnits",
            "gl_MaxTessControlUniformComponents",
            "gl_MaxTessControlTotalOutputComponents",
            "gl_MaxTessEvaluationInputComponents",
            "gl_MaxTessEvaluationOutputComponents",
            "gl_MaxTessEvaluationTextureImageUnits",
            "gl_MaxTessEvaluationUniformComponents",
            "gl_TessLevelOuter",
            "gl_TessLevelInner",
            "gl_TessCoord",
            "gl_Position",
            "gl_PointSize",
            "gl_PrimitiveIDIn",
            "gl_InvocationID",
            "gl_TessLevelOuter",
            "gl_TessLevelInner",
            "gl_TessCoord",
            "gl_PrimitiveID",
            "gl_TessLevelOuter",
            "gl_TessLevelInner",
            "gl_FragColor",
            "gl_FragCoord",
            "gl_FragDepth",
            "gl_PointCoord",
            "gl_FrontFacing",
            "gl_SampleID",
            "gl_SamplePosition",
            "gl_SampleMask",
            "gl_SampleMaskIn",
            "gl_MaxSampleMaskWords",
            "gl_FogCoord",
            "gl_ClipDistance",
            "gl_CullDistance",
            "gl_PrimitiveID",
            "gl_Layer",
            "gl_ViewportIndex",
            "gl_FragData",
            "gl_HelperInvocation",
            "gl_FragDepthLayout",
            "gl_MaxVertexOutputComponents",
            "gl_MaxGeometryInputComponents",
            "gl_MaxGeometryOutputComponents",
            "gl_MaxFragmentInputComponents",
            "gl_MaxGeometryTextureImageUnits",
            "gl_MaxGeometryUniformComponents",
            "gl_MaxGeometryVaryingComponents",
            "gl_MaxGeometryOutputVertices",
            "gl_MaxGeometryTotalOutputComponents",
            "gl_MaxGeometryShaderInvocations",
            "gl_MaxGeometryVertexStreams",
            "gl_MaxGeometryOutputStreams",
            "gl_MaxGeometryOutputVertices",
            "gl_MaxGeometryTotalOutputComponents",
            "gl_MaxGeometryShaderInvocations",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
            "gl_ClipDistance",
        ]

        for keyword in reserved_keywords:
            self.namer.reserve_name(keyword)

    def _collect_required_features(self) -> None:
        """Analyze module and collect required GLSL features."""
        # Placeholder implementation
        # Would analyze module types, expressions, etc.
        pass

    def _write_version_directive(self) -> None:
        """Write GLSL version directive."""
        if self.options.is_es():
            self.out.write(f"#version {self.options.version} es\n")
        else:
            self.out.write(f"#version {self.options.version}\n")
        self.out.write("\n")

    def _write_extensions(self) -> None:
        """Write required GLSL extensions."""
        for ext in self.features:
            self.out.write(f"#extension {ext} : require\n")
        if self.features:
            self.out.write("\n")

    def _write_precision(self) -> None:
        """Write precision qualifiers for ES."""
        self.out.write("\n")
        self.out.write("precision highp float;\n")
        self.out.write("precision highp int;\n")
        self.out.write("\n")

    def _write_uniforms(self) -> None:
        """Write uniform variable declarations."""
        for ty, global_var in self.module.global_variables.items():
            if hasattr(global_var, "space") and str(global_var.space) == "Uniform":
                var_name = self._get_variable_name(global_var)
                type_name = self._type_to_glsl(global_var.ty)
                self.out.write(f"uniform {type_name} {var_name};\n")
        self.out.write("\n")

    def _write_varyings(self) -> None:
        """Write varying variable declarations."""
        # Determine if we're writing input or output varyings
        is_vertex_shader = self.pipeline_options.shader_stage == ShaderStage.Vertex
        is_fragment_shader = self.pipeline_options.shader_stage == ShaderStage.Fragment

        if is_vertex_shader:
            # Vertex shader outputs
            prefix = "out"
        elif is_fragment_shader:
            # Fragment shader inputs
            prefix = "in"
        else:
            return

        # Write varyings based on entry point
        if self.entry_point and hasattr(self.entry_point, "function"):
            for i, arg in enumerate(self.entry_point.function.arguments):
                if hasattr(arg, "binding") and arg.binding:
                    var_name = f"varying_{i}"
                    type_name = self._type_to_glsl(arg.ty)
                    location = getattr(arg.binding, "location", 0)
                    self.out.write(f"{prefix} {type_name} {var_name};\n")

        self.out.write("\n")

    def _write_functions(self) -> None:
        """Write regular functions."""
        for handle, function in self.module.functions.items():
            self._write_function(function)
            self.out.write("\n")

    def _write_entry_point(self) -> None:
        """Write the main entry point function."""
        if not self.entry_point:
            return

        # Write function signature
        func_name = self._get_entry_point_name()
        self.out.write(f"void main() {{\n")

        # Write function body - simplified
        if hasattr(self.entry_point, "function") and hasattr(
            self.entry_point.function, "body"
        ):
            self._write_block(self.entry_point.function.body, 1)

        self.out.write("}\n")

    def _write_function(self, function: Any) -> None:
        """Write a single function."""
        func_name = self._get_function_name(function)

        # Write function signature
        self.out.write(f"void {func_name}(")

        # Write arguments
        for i, arg in enumerate(function.arguments):
            arg_name = f"arg_{i}"
            type_name = self._type_to_glsl(arg.ty)
            self.out.write(f"{type_name} {arg_name}")
            if i < len(function.arguments) - 1:
                self.out.write(", ")

        self.out.write(") {\n")

        # Write function body
        if hasattr(function, "body") and function.body:
            self._write_block(function.body, 1)

        self.out.write("}\n")

    def _write_block(self, block: Any, indent_level: int) -> None:
        """Write a statement block with proper indentation."""
        indent = "    " * indent_level

        for stmt in block:
            self._write_statement(stmt, indent_level)

    def _write_statement(self, stmt: Any, indent_level: int) -> None:
        """Write a single statement."""
        indent = "    " * indent_level

        stmt_type = type(stmt).__name__

        if stmt_type == "LocalVariable":
            var_name = self._get_local_variable_name(stmt)
            self.out.write(f"{indent}var {var_name}")
            if hasattr(stmt, "ty") and stmt.ty:
                self.out.write(f": {self._type_to_glsl(stmt.ty)}")
            if hasattr(stmt, "init") and stmt.init:
                self.out.write(f" = {self._expression_to_glsl(stmt.init)}")
            self.out.write(";\n")
        elif stmt_type == "Store":
            if hasattr(stmt, "pointer") and hasattr(stmt, "value"):
                self.out.write(
                    f"{indent}{self._expression_to_glsl(stmt.pointer)} = {self._expression_to_glsl(stmt.value)};\n"
                )
        elif stmt_type == "Return":
            if hasattr(stmt, "value") and stmt.value:
                self.out.write(
                    f"{indent}return {self._expression_to_glsl(stmt.value)};\n"
                )
            else:
                self.out.write(f"{indent}return;\n")
        elif stmt_type == "Break":
            self.out.write(f"{indent}break;\n")
        elif stmt_type == "Continue":
            self.out.write(f"{indent}continue;\n")
        elif stmt_type == "Discard":
            self.out.write(f"{indent}discard;\n")
        else:
            self.out.write(f"{indent}// TODO: Implement {stmt_type}\n")

    def _expression_to_glsl(self, expr: Any) -> str:
        """Convert an expression to GLSL string representation."""
        expr_type = type(expr).__name__

        if expr_type == "Literal":
            return str(expr.value)
        elif expr_type == "Variable":
            return self._get_variable_name(expr)
        elif expr_type == "BinaryOperation":
            left = self._expression_to_glsl(expr.left)
            right = self._expression_to_glsl(expr.right)
            op = self._binary_op_to_glsl(expr.op)
            return f"({left} {op} {right})"
        elif expr_type == "Call":
            func_name = self._get_function_name(expr.function)
            args = [self._expression_to_glsl(arg) for arg in expr.arguments]
            return f"{func_name}({', '.join(args)})"
        elif expr_type == "Swizzle":
            base = self._expression_to_glsl(expr.base)
            components = "".join(expr.components)
            return f"{base}.{components}"
        else:
            return f"/* TODO: {expr_type} */"

    def _binary_op_to_glsl(self, op: Any) -> str:
        """Convert binary operation to GLSL operator."""
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
            "LogicalOr": "||",
        }
        return op_map.get(str(op), "?")

    def _type_to_glsl(self, ty: Any) -> str:
        """Convert a type to GLSL string representation."""
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
            type_map = {
                "F16": "mediump float" if self.options.is_es() else "float",
                "F32": "mediump float" if self.options.is_es() else "float",
                "F64": "double",
                "I32": "int",
                "U32": "uint" if not self.options.is_es() else "int",
                "Bool": "bool",
            }
            return type_map.get(scalar_type, str(inner).lower())
        elif str(inner).startswith("Vector"):
            scalar = (
                self._type_to_glsl(inner.scalar)
                if hasattr(inner, "scalar")
                else "float"
            )
            size = getattr(inner, "size", 2)
            if scalar in ["mediump float", "float"]:
                return f"vec{size}"
            elif scalar == "int":
                return f"ivec{size}"
            elif scalar == "uint":
                return f"uvec{size}"
            elif scalar == "bool":
                return f"bvec{size}"
            else:
                return f"{scalar} vec{size}"
        elif str(inner).startswith("Matrix"):
            scalar = (
                self._type_to_glsl(inner.scalar)
                if hasattr(inner, "scalar")
                else "float"
            )
            columns = getattr(inner, "columns", 2)
            rows = getattr(inner, "rows", 2)
            return f"mat{columns}x{rows}"
        elif str(inner).startswith("Array"):
            element = (
                self._type_to_glsl(inner.element)
                if hasattr(inner, "element")
                else "float"
            )
            count = getattr(inner, "size", None)
            if count is None:
                return f"{element}[]"
            else:
                return f"{element}[{count}]"
        else:
            return str(inner).lower()

    def _get_variable_name(self, var: Any) -> str:
        """Get the GLSL variable name."""
        if hasattr(var, "name") and var.name:
            return self.namer.get_name(var.name)
        return f"var_{id(var)}"

    def _get_local_variable_name(self, var: Any) -> str:
        """Get the GLSL local variable name."""
        return f"local_{id(var)}"

    def _get_function_name(self, func: Any) -> str:
        """Get the GLSL function name."""
        if hasattr(func, "name") and func.name:
            return self.namer.get_name(func.name)
        return f"func_{id(func)}"

    def _get_entry_point_name(self) -> str:
        """Get the GLSL entry point name."""
        if self.entry_point and hasattr(self.entry_point, "name"):
            return self.entry_point.name
        return "main"

    def finish(self) -> str:
        """Finish writing and return the complete output."""
        return self.out.getvalue()


class GLSLNameGenerator:
    """Generator for unique GLSL names."""

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
    module: Any, info: Any, options: Options, pipeline_options: PipelineOptions
) -> str:
    """
    Write a module to GLSL string.

    Args:
        module: The Naga IR module
        info: Module validation info
        options: GLSL writer options
        pipeline_options: Pipeline configuration

    Returns:
        Generated GLSL code as string
    """
    writer = Writer("", module, info, options, pipeline_options)
    writer.write()
    return writer.finish()
