from typing import Optional
from pydantic import BaseModel

class CommandEncoderDescriptor(BaseModel):
    """Description for a CommandEncoder."""
    label: Optional[str] = None

class CommandBufferDescriptor(BaseModel):
    """Description for a CommandBuffer finish."""
    label: Optional[str] = None
