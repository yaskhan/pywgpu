from typing import Optional, List
from pydantic import BaseModel

class RenderBundleDescriptor(BaseModel):
    label: Optional[str] = None

class RenderBundleEncoderDescriptor(BaseModel):
    label: Optional[str] = None
    color_formats: List[Optional[str]]
    depth_stencil_format: Optional[str] = None
    sample_count: int = 1
    multiview: Optional[int] = None
