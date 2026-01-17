from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field
from enum import Enum, IntFlag
from .features import Features
from .limits import Limits

class DeviceLostReason(Enum):
    DESTROYED = "destroyed"

class DeviceDescriptor(BaseModel):
    label: Optional[str] = None
    required_features: List[str] = Field(default_factory=list) # Should be Features flag or list of strings
    required_limits: dict = Field(default_factory=dict) # Should be Limits object or dict
    # memory_hints: MemoryHints (todo)
    # trace_path: Optional[str] (todo)
