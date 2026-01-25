"""
GLSL reserved keywords.

Contains a set of all reserved GLSL keywords that should not be used
as variable or function names.
"""

from __future__ import annotations

from typing import FrozenSet


# All reserved GLSL keywords
# This list is derived from the Khronos GLSL specification
RESERVED_KEYWORD_SET: FrozenSet[str] = frozenset([
    # Keywords from GLSL 1.00 - 1.50
    "attribute",
    "const",
    "uniform",
    "varying",
    "break",
    "continue",
    "do",
    "for",
    "while",
    "if",
    "else",
    "in",
    "out",
    "inout",
    "float",
    "int",
    "void",
    "bool",
    "true",
    "false",
    "return",
    "discard",
    "struct",
    
    # Keywords from GLSL 1.10
    "invariant",
    "precision",
    
    # Keywords from GLSL 1.20
    "highp",
    "mediump",
    "lowp",
    
    # Keywords from GLSL 1.30
    "layout",
    "centroid",
    "flat",
    "smooth",
    "noperspective",
    
    # Keywords from GLSL 1.40
    "sampler1D",
    "sampler2D",
    "sampler3D",
    "samplerCube",
    "sampler1DShadow",
    "sampler2DShadow",
    "samplerCubeShadow",
    "sampler1DArray",
    "sampler2DArray",
    "sampler1DArrayShadow",
    "sampler2DArrayShadow",
    "isampler1D",
    "isampler2D",
    "isampler3D",
    "isamplerCube",
    "isampler1DArray",
    "isampler2DArray",
    "usampler1D",
    "usampler2D",
    "usampler3D",
    "usamplerCube",
    "usampler1DArray",
    "usampler2DArray",
    
    # Keywords from GLSL 1.50
    "patch",
    "subroutine",
    
    # Keywords from GLSL 3.00 ES (WebGL 2.0)
    "texture",
    "textureBuffer",
    "texture1D",
    "texture2D",
    "texture3D",
    "textureCube",
    "texture1DArray",
    "texture2DArray",
    "textureCubeArray",
    "texture2DMS",
    "texture2DMSArray",
    "itexture1D",
    "itexture2D",
    "itexture3D",
    "itextureCube",
    "itexture1DArray",
    "itexture2DArray",
    "itextureCubeArray",
    "itexture2DMS",
    "itexture2DMSArray",
    "utexture1D",
    "utexture2D",
    "utexture3D",
    "utextureCube",
    "utexture1DArray",
    "utexture2DArray",
    "utextureCubeArray",
    "utexture2DMS",
    "utexture2DMSArray",
    "samplerBuffer",
    "isamplerBuffer",
    "usamplerBuffer",
    
    # Keywords from GLSL 4.00+
    "nonuniform",
    "coherent",
    "restrict",
    "readonly",
    "writeonly",
    
    # Keywords from GLSL 4.20
    "explicit",
    
    # Keywords from GLSL 4.30
    "match",
    
    # Keywords from GLSL 4.40
    "all",
    "any",
    
    # Keywords from GLSL 4.50
    "asm",
    "class",
    "union",
    "enum",
    "typedef",
    "template",
    "this",
    "resource",
    "gshared",
    "template",
    "class",
    
    # Keywords from GLSL 4.60
    "float16_t",
    "float32_t",
    "float64_t",
    "f16vec2",
    "f16vec3",
    "f16vec4",
    "f32vec2",
    "f32vec3",
    "f32vec4",
    "f64vec2",
    "f64vec3",
    "f64vec4",
    
    # Reserved words (cannot be used as identifiers)
    "common",
    "partition",
    "active",
    "asm",
    "class",
    "union",
    "enum",
    "typedef",
    "template",
    "this",
    "resource",
    "gshared",
    "volatile",
    "restrict",
    "readonly",
    "writeonly",
    "shared",
    "coherent",
    "buffer",
    "std140",
    "std430",
    "row_major",
    "column_major",
    "packed",
    "template",
    "class",
    "in",
    "out",
    "inout",
    
    # Preprocessor directives (not keywords but reserved)
    # These are handled by the preprocessor
    
    # Built-in variables (not keywords but reserved)
    "gl_Position",
    "gl_PointSize",
    "gl_FragCoord",
    "gl_FragColor",
    "gl_FragData",
    "gl_FragDepth",
    "gl_FrontFacing",
    "gl_PointCoord",
    "gl_VertexID",
    "gl_InstanceID",
    "gl_PrimitiveID",
    "gl_Layer",
    "gl_ViewportIndex",
    "gl_TessLevelOuter",
    "gl_TessLevelInner",
    "gl_TessCoord",
    "gl_InvocationID",
    "gl_ClipDistance",
    "gl_CullDistance",
    "gl_SampleID",
    "gl_SamplePosition",
    "gl_SampleMask",
    "gl_SampleMaskIn",
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
    "gl_DepthRangeParameters",
    "gl_DepthRange",
    
    # Compute shader built-ins
    "gl_NumWorkGroups",
    "gl_WorkGroupID",
    "gl_LocalInvocationID",
    "gl_GlobalInvocationID",
    "gl_LocalInvocationIndex",
    "gl_WorkGroupSize",
    
    # Geometry shader built-ins
    "gl_InvocationID",
    "gl_PrimitiveIDIn",
    "gl_InVertexCount",
    "gl_InPrimitiveID",
    "gl_InLayer",
    "gl_OutLayer",
    "gl_OutViewportIndex",
    "gl_InvocationID",
    
    # Tessellation shader built-ins
    "gl_TessLevelOuter",
    "gl_TessLevelInner",
    "gl_TessCoord",
    "gl_PatchVerticesIn",
    "gl_PatchVertices",
    "gl_TessGenMode",
    "gl_TessGenPointMode",
    "gl_TessGenSpacing",
    "gl_TessGenVertexOrder",
    "gl_TessGenFaceType",
    
    # Subgroup built-ins
    "gl_SubgroupSize",
    "gl_SubgroupInvocationID",
    "gl_SubgroupEqMask",
    "gl_SubgroupGeMask",
    "gl_SubgroupGtMask",
    "gl_SubgroupLeMask",
    "gl_SubgroupLtMask",
    "gl_NumSubgroups",
    "gl_SubgroupID",
])


__all__ = ["RESERVED_KEYWORD_SET"]
