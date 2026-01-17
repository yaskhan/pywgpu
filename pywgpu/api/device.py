from typing import Optional, TYPE_CHECKING, Any
from pywgpu_types.descriptors import (
    DeviceDescriptor, 
    BufferDescriptor, 
    ShaderModuleDescriptor
)

if TYPE_CHECKING:
    from .buffer import Buffer
    from .texture import Texture
    from .sampler import Sampler
    from .bind_group import BindGroup, BindGroupLayout
    from .pipeline_layout import PipelineLayout
    from .shader_module import ShaderModule
    from .command_encoder import CommandEncoder
    from .render_bundle_encoder import RenderBundleEncoder
    from .query_set import QuerySet
    from .compute_pipeline import ComputePipeline
    from .render_pipeline import RenderPipeline
    from .pipeline_cache import PipelineCache

class Device:
    """
    Open connection to a graphics and/or compute device.
    
    Responsible for creating most resources (buffers, textures, pipelines, etc.).
    """
    
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def features(self) -> Any:
        """Returns the features supported by this device."""
        pass

    def limits(self) -> Any:
        """Returns the limits of this device."""
        pass

    def create_buffer(self, descriptor: BufferDescriptor) -> 'Buffer':
        """Creates a buffer given its descriptor."""
        pass

    def create_texture(self, descriptor: Any) -> 'Texture':
        """Creates a texture given its descriptor."""
        pass

    def create_sampler(self, descriptor: Any = None) -> 'Sampler':
        """Creates a sampler given its descriptor."""
        pass

    def create_bind_group(self, descriptor: Any) -> 'BindGroup':
        """Creates a bind group given its descriptor."""
        pass

    def create_bind_group_layout(self, descriptor: Any) -> 'BindGroupLayout':
        """Creates a bind group layout given its descriptor."""
        pass

    def create_pipeline_layout(self, descriptor: Any) -> 'PipelineLayout':
        """Creates a pipeline layout."""
        pass

    def create_shader_module(self, descriptor: ShaderModuleDescriptor) -> 'ShaderModule':
        """Creates a shader module from source code (WGSL/SPIR-V/GLSL)."""
        pass

    def create_command_encoder(self, descriptor: Any = None) -> 'CommandEncoder':
        """Creates a command encoder."""
        pass

    def create_render_bundle_encoder(self, descriptor: Any) -> 'RenderBundleEncoder':
        """Creates a render bundle encoder."""
        pass

    def create_query_set(self, descriptor: Any) -> 'QuerySet':
        """Creates a query set."""
        pass

    def create_compute_pipeline(self, descriptor: Any) -> 'ComputePipeline':
        """Creates a compute pipeline."""
        pass

    def create_render_pipeline(self, descriptor: Any) -> 'RenderPipeline':
        """Creates a render pipeline."""
        pass

    def create_pipeline_cache(self, descriptor: Any = None) -> 'PipelineCache':
        """Creates a pipeline cache."""
        pass

    def destroy(self) -> None:
        """Destroys the device."""
        pass

    def set_device_lost_callback(self, callback: Any) -> None:
        """Sets a callback to be invoked when the device is lost."""
        pass

    def on_uncaptured_error(self, callback: Any) -> None:
        """Sets a callback for uncaptured errors."""
        pass
