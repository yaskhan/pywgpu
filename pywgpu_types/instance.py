from enum import IntFlag, Enum
from pydantic import BaseModel, Field
from typing import Optional, List
from .backend import Backends


class InstanceFlags(IntFlag):
    DEBUG = 1 << 0
    VALIDATION = 1 << 1
    DISCARD_HAL_LABELS = 1 << 2
    GPU_BASED_VALIDATION = 1 << 3


class Dx12Compiler(Enum):
    DXC = "dxc"
    FXC = "fxc"


class Gles3MinorVersion(Enum):
    AUTOMATIC = "automatic"
    V0 = "0"
    V1 = "1"
    V2 = "2"


class InstanceDescriptor(BaseModel):
    backends: Backends = Backends.ALL
    flags: InstanceFlags = InstanceFlags.from_bytes(0)  # Default empty
    dx12_shader_compiler: Dx12Compiler = Dx12Compiler.DXC
    gles_minor_version: Gles3MinorVersion = Gles3MinorVersion.AUTOMATIC
