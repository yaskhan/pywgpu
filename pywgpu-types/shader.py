from typing import Union, Optional, List
from enum import IntFlag, Enum
from pydantic import BaseModel

class ShaderStages(IntFlag):
    NONE = 0
    VERTEX = 1 << 0
    FRAGMENT = 1 << 1
    COMPUTE = 1 << 2
    VERTEX_FRAGMENT = VERTEX | FRAGMENT

class ShaderSource(BaseModel):
    """Source code for a shader module."""
    code: Union[str, bytes] 
    language: str = "wgsl" # wgsl, glsl, spirv

class ShaderModuleDescriptor(BaseModel):
    label: Optional[str] = None
    source: ShaderSource

class CompilationMessage(BaseModel):
    message: str
    type: str # 'error', 'warning', 'info'
    line_num: int
    line_pos: int
    offset: int
    length: int

class CompilationInfo(BaseModel):
    messages: List[CompilationMessage]
