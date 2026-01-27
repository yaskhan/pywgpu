from typing import Any
import pywgpu_types as wgt

# Maximum binding size for the shaders that only support i32 indexing.
MAX_I32_BINDING_SIZE: int = (1 << 31) - 1

def map_naga_stage(stage: Any) -> Any:
    """Map naga.ShaderStage to wgt.ShaderStages."""
    # Assuming naga is available or passed as Any
    from naga import ShaderStage
    if stage == ShaderStage.Vertex:
        return wgt.ShaderStages.VERTEX
    if stage == ShaderStage.Fragment:
        return wgt.ShaderStages.FRAGMENT
    if stage == ShaderStage.Compute:
        return wgt.ShaderStages.COMPUTE
    if stage == ShaderStage.Task:
        return wgt.ShaderStages.TASK
    if stage == ShaderStage.Mesh:
        return wgt.ShaderStages.MESH
    return wgt.ShaderStages.NONE

def apply_hal_limits(limits: Any) -> Any:
    """Clamp limits to honor HAL-imposed maximums."""
    from ..lib import MAX_BIND_GROUPS, MAX_VERTEX_BUFFERS, MAX_COLOR_ATTACHMENTS
    
    limits.max_bind_groups = min(limits.max_bind_groups, MAX_BIND_GROUPS)
    limits.max_vertex_buffers = min(limits.max_vertex_buffers, MAX_VERTEX_BUFFERS)
    limits.max_color_attachments = min(limits.max_color_attachments, MAX_COLOR_ATTACHMENTS)
    
    # Round some limits down to the WebGPU alignment requirement
    limits.max_storage_buffer_binding_size &= ~(wgt.STORAGE_BINDING_SIZE_ALIGNMENT - 1)
    # limits.max_vertex_buffer_array_stride &= ~(wgt.VERTEX_ALIGNMENT - 1) # Depends on wgt
    
    return limits
