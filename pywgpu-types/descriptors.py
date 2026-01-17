from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Union

class DeviceDescriptor(BaseModel):
    label: Optional[str] = None
    required_features: List[str] = Field(default_factory=list)
    required_limits: dict[str, int] = Field(default_factory=dict)

class BufferDescriptor(BaseModel):
    label: Optional[str] = None
    size: int
    usage: int
    mapped_at_creation: bool = False

class ShaderModuleDescriptor(BaseModel):
    label: Optional[str] = None
    code: str
