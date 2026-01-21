"""
Backend module for shader code generation.

This module contains writers for different shader backends:
- WGSL (WebGPU Shading Language)
- GLSL (OpenGL Shading Language)
- HLSL (High-Level Shading Language)
- MSL (Metal Shading Language)
- SPIR-V (Standard Portable Intermediate Representation)
"""

from typing import Any
from .wgsl import Writer as WgslWriter, WriterFlags, write_string as write_wgsl_string
from .glsl import (
    Writer as GlslWriter,
    Version,
    Profile,
    Options as GlslOptions,
    ShaderStage,
    write_string as write_glsl_string,
)
from .hlsl import (
    Writer as HlslWriter,
    ShaderModel,
    Options as HlslOptions,
    ShaderStage as HlslShaderStage,
    write_string as write_hlsl_string,
)
from .msl import (
    Writer as MslWriter,
    Options as MslOptions,
    ShaderStage as MslShaderStage,
    write_string as write_msl_string,
)
from .spv import (
    Writer as SpvWriter,
    Options as SpvOptions,
    write_binary as write_spirv_binary,
)

__all__ = [
    # Base Writer class
    "Writer",
    # WGSL backend
    "WgslWriter",
    "WriterFlags",
    "write_wgsl_string",
    # GLSL backend
    "GlslWriter",
    "Version",
    "Profile",
    "GlslOptions",
    "ShaderStage",
    "write_glsl_string",
    # HLSL backend
    "HlslWriter",
    "ShaderModel",
    "HlslOptions",
    "HlslShaderStage",
    "write_hlsl_string",
    # MSL backend
    "MslWriter",
    "MslOptions",
    "MslShaderStage",
    "write_msl_string",
    # SPIR-V backend
    "SpvWriter",
    "SpvOptions",
    "write_spirv_binary",
]


class Writer:
    """
    Base class for shader writers (SPV, MSL, HLSL, GLSL, WGSL).

    This class provides a common interface for all backend writers.
    Each backend should inherit from this class and implement the write method.
    """

    def write(self, module: "Any", info: "Any") -> "Any":
        """
        Write the shader module to the target format.

        Args:
            module: The Naga IR module to write
            info: Module validation information

        Returns:
            The generated shader code in the target format

        Raises:
            ShaderError: If writing fails
        """
        raise NotImplementedError("Subclasses must implement write method")

    def finish(self) -> str:
        """
        Finish writing and return the complete output.

        Returns:
            The complete generated shader code
        """
        return ""
