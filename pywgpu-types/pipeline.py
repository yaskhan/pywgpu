from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field

from .binding import BindGroupLayout
from .shader import ShaderModule

# --- Common Types ---


class PipelineLayoutDescriptor(BaseModel):
    label: Optional[str] = None
    bind_group_layouts: List[BindGroupLayout]
    immediate_size: int = 0


class PipelineLayout(BaseModel):
    pass


class PipelineCompilationOptions(BaseModel):
    """Options for pipeline compilation."""

    constants: Dict[str, float] = Field(default_factory=dict)
    zero_initialize_workgroup_memory: bool = True


# --- Compute Pipeline Types ---


class ComputePipelineDescriptor(BaseModel):
    label: Optional[str] = None
    layout: Optional[PipelineLayout] = None
    module: ShaderModule
    entry_point: Optional[str] = None
    compilation_options: PipelineCompilationOptions = Field(
        default_factory=PipelineCompilationOptions
    )


# --- Render Pipeline Types ---


class PrimitiveState(BaseModel):
    topology: str = "triangle-list"
    strip_index_format: Optional[str] = None
    front_face: str = "ccw"
    cull_mode: Optional[str] = None
    unclipped_depth: bool = False


class MultisampleState(BaseModel):
    count: int = 1
    mask: int = 0xFFFFFFFF
    alpha_to_coverage_enabled: bool = False


class BlendComponent(BaseModel):
    src_factor: str = "one"
    dst_factor: str = "zero"
    operation: str = "add"


class BlendState(BaseModel):
    color: BlendComponent
    alpha: BlendComponent


class ColorTargetState(BaseModel):
    format: str
    blend: Optional[BlendState] = None
    write_mask: int = 0xF


class StencilFaceState(BaseModel):
    compare: str = "always"
    fail_op: str = "keep"
    depth_fail_op: str = "keep"
    pass_op: str = "keep"


class DepthStencilState(BaseModel):
    format: str
    depth_write_enabled: bool = False
    depth_compare: str = "always"
    stencil_front: StencilFaceState = Field(default_factory=StencilFaceState)
    stencil_back: StencilFaceState = Field(default_factory=StencilFaceState)
    stencil_read_mask: int = 0xFF
    stencil_write_mask: int = 0xFF
    depth_bias: int = 0
    depth_bias_slope: float = 0.0
    depth_bias_clamp: float = 0.0


class VertexAttribute(BaseModel):
    format: str
    offset: int
    shader_location: int


class VertexBufferLayout(BaseModel):
    array_stride: int
    step_mode: str = "vertex"
    attributes: List[VertexAttribute]


class VertexState(BaseModel):
    module: ShaderModule
    entry_point: Optional[str] = None
    compilation_options: PipelineCompilationOptions = Field(
        default_factory=PipelineCompilationOptions
    )
    buffers: List[VertexBufferLayout] = Field(default_factory=list)


class FragmentState(BaseModel):
    module: ShaderModule
    entry_point: Optional[str] = None
    compilation_options: PipelineCompilationOptions = Field(
        default_factory=PipelineCompilationOptions
    )
    targets: List[Optional[ColorTargetState]]


class RenderPipelineDescriptor(BaseModel):
    label: Optional[str] = None
    layout: Optional[PipelineLayout] = None
    vertex: VertexState
    fragment: Optional[FragmentState] = None
    primitive: PrimitiveState = Field(default_factory=PrimitiveState)
    depth_stencil: Optional[DepthStencilState] = None
    multisample: MultisampleState = Field(default_factory=MultisampleState)
    multiview: Optional[int] = None


# --- Mesh Pipeline Types ---


class TaskState(BaseModel):
    module: ShaderModule
    entry_point: Optional[str] = None
    compilation_options: PipelineCompilationOptions = Field(
        default_factory=PipelineCompilationOptions
    )


class MeshState(BaseModel):
    module: ShaderModule
    entry_point: Optional[str] = None
    compilation_options: PipelineCompilationOptions = Field(
        default_factory=PipelineCompilationOptions
    )


class MeshPipelineDescriptor(BaseModel):
    label: Optional[str] = None
    layout: Optional[PipelineLayout] = None
    task: Optional[TaskState] = None
    mesh: MeshState
    fragment: Optional[FragmentState] = None
    primitive: PrimitiveState = Field(default_factory=PrimitiveState)
    depth_stencil: Optional[DepthStencilState] = None
    multisample: MultisampleState = Field(default_factory=MultisampleState)
    multiview: Optional[int] = None
