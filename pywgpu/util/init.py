from typing import Any


class ShaderSource:
    """Wrapper for shader source data.

    Attributes:
        source_type: Type of shader source ('spirv' or 'wgsl').
        data: The shader source data (bytes for SPIR-V, str for WGSL).
    """

    def __init__(self, source_type: str, data: Any):
        self.source_type = source_type
        self.data = data

    def is_spirv(self) -> bool:
        """Check if this is SPIR-V source."""
        return self.source_type == "spirv"

    def is_wgsl(self) -> bool:
        """Check if this is WGSL source."""
        return self.source_type == "wgsl"


def make_spirv(data: bytes) -> ShaderSource:
    """Creates a ShaderSource from SPIR-V data.

    Args:
        data: The SPIR-V binary data.

    Returns:
        A ShaderSource object wrapping the SPIR-V data.
    """
    return ShaderSource("spirv", data)


def make_wgsl(code: str) -> ShaderSource:
    """Creates a ShaderSource from WGSL code.

    Args:
        code: The WGSL shader code.

    Returns:
        A ShaderSource object wrapping the WGSL code.
    """
    return ShaderSource("wgsl", code)
