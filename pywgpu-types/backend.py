from enum import Enum, IntFlag


class Backend(Enum):
    """Backend graphics API."""

    EMPTY = 0
    VULKAN = 1
    METAL = 2
    DX12 = 3
    DX11 = 4
    GL = 5
    BROWSER_WEBGPU = 6


class Backends(IntFlag):
    """Backends enable flags."""

    NOOP = 0  # Verify exact value, usually 0 or implicit
    VULKAN = 1 << 1
    GL = 1 << 5  # Matching rust bitflags logic roughly, likely need specific values
    METAL = 1 << 2
    DX12 = 1 << 3
    DX11 = 1 << 4
    BROWSER_WEBGPU = 1 << 6
    PRIMARY = VULKAN | METAL | DX12 | BROWSER_WEBGPU
    SECONDARY = GL | DX11
    ALL = PRIMARY | SECONDARY
