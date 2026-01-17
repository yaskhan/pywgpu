from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field

# --- Common Types ---

class Operations(BaseModel):
    load: str # 'load', 'clear', 'undefined'
    store: bool # bool or 'store', 'discard' in WebGPU, here simpler boolean often used

class RenderPassColorAttachment(BaseModel):
    view: Any # TextureView
    resolve_target: Optional[Any] = None # TextureView
    ops: Operations
    clear_value: Union[List[float], Any] = [0, 0, 0, 0]

class RenderPassDepthStencilAttachment(BaseModel):
    view: Any # TextureView
    depth_ops: Optional[Operations] = None
    stencil_ops: Optional[Operations] = None
    clear_depth: float = 1.0
    clear_stencil: int = 0

class RenderPassTimestampWrites(BaseModel):
    query_set: Any # QuerySet
    beginning_of_pass_write_index: Optional[int] = None
    end_of_pass_write_index: Optional[int] = None

class ComputePassTimestampWrites(BaseModel):
    query_set: Any # QuerySet
    beginning_of_pass_write_index: Optional[int] = None
    end_of_pass_write_index: Optional[int] = None

# --- Descriptors ---

class RenderPassDescriptor(BaseModel):
    label: Optional[str] = None
    color_attachments: List[Optional[RenderPassColorAttachment]]
    depth_stencil_attachment: Optional[RenderPassDepthStencilAttachment] = None
    timestamp_writes: Optional[RenderPassTimestampWrites] = None
    occlusion_query_set: Optional[Any] = None

class ComputePassDescriptor(BaseModel):
    label: Optional[str] = None
    timestamp_writes: Optional[ComputePassTimestampWrites] = None
