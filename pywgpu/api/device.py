from typing import Optional, TYPE_CHECKING, Any, Tuple
from pywgpu_types.descriptors import (
    DeviceDescriptor, 
    BufferDescriptor, 
    ShaderModuleDescriptor
)
from pywgpu.util.device import DeviceExt

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
    from .queue import Queue

class Device(DeviceExt):
    """
    Open connection to a graphics and/or compute device.
    
    Responsible for creating most resources (buffers, textures, pipelines, etc.).
    """
    
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def features(self) -> Any:
        """Returns the features supported by this device."""
        if hasattr(self._inner, 'features'):
            return self._inner.features()
        else:
            raise NotImplementedError("Backend does not support features")

    def limits(self) -> Any:
        """Returns the limits of this device."""
        if hasattr(self._inner, 'limits'):
            return self._inner.limits()
        else:
            raise NotImplementedError("Backend does not support limits")

    def create_buffer(self, descriptor: BufferDescriptor) -> 'Buffer':
        """Creates a buffer given its descriptor."""
        from .buffer import Buffer
        if hasattr(self._inner, 'create_buffer'):
            buffer_inner = self._inner.create_buffer(descriptor)
            return Buffer(buffer_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_buffer")

    def create_texture(self, descriptor: Any) -> 'Texture':
        """Creates a texture given its descriptor."""
        from .texture import Texture
        if hasattr(self._inner, 'create_texture'):
            texture_inner = self._inner.create_texture(descriptor)
            return Texture(texture_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_texture")

    def create_sampler(self, descriptor: Any = None) -> 'Sampler':
        """Creates a sampler given its descriptor."""
        from .sampler import Sampler
        if hasattr(self._inner, 'create_sampler'):
            sampler_inner = self._inner.create_sampler(descriptor)
            return Sampler(sampler_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_sampler")

    def create_bind_group(self, descriptor: Any) -> 'BindGroup':
        """Creates a bind group given its descriptor."""
        from .bind_group import BindGroup
        if hasattr(self._inner, 'create_bind_group'):
            bind_group_inner = self._inner.create_bind_group(descriptor)
            return BindGroup(bind_group_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_bind_group")

    def create_bind_group_layout(self, descriptor: Any) -> 'BindGroupLayout':
        """Creates a bind group layout given its descriptor."""
        from .bind_group_layout import BindGroupLayout
        if hasattr(self._inner, 'create_bind_group_layout'):
            layout_inner = self._inner.create_bind_group_layout(descriptor)
            return BindGroupLayout(layout_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_bind_group_layout")

    def create_pipeline_layout(self, descriptor: Any) -> 'PipelineLayout':
        """Creates a pipeline layout."""
        from .pipeline_layout import PipelineLayout
        if hasattr(self._inner, 'create_pipeline_layout'):
            layout_inner = self._inner.create_pipeline_layout(descriptor)
            return PipelineLayout(layout_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_pipeline_layout")

    def create_shader_module(self, descriptor: ShaderModuleDescriptor) -> 'ShaderModule':
        """Creates a shader module from source code (WGSL/SPIR-V/GLSL)."""
        from .shader_module import ShaderModule
        if hasattr(self._inner, 'create_shader_module'):
            shader_module_inner = self._inner.create_shader_module(descriptor)
            return ShaderModule(shader_module_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_shader_module")

    def create_command_encoder(self, descriptor: Any = None) -> 'CommandEncoder':
        """Creates a command encoder."""
        from .command_encoder import CommandEncoder
        if hasattr(self._inner, 'create_command_encoder'):
            encoder_inner = self._inner.create_command_encoder(descriptor)
            return CommandEncoder(encoder_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_command_encoder")

    def create_render_bundle_encoder(self, descriptor: Any) -> 'RenderBundleEncoder':
        """Creates a render bundle encoder."""
        from .render_bundle_encoder import RenderBundleEncoder
        if hasattr(self._inner, 'create_render_bundle_encoder'):
            encoder_inner = self._inner.create_render_bundle_encoder(descriptor)
            return RenderBundleEncoder(encoder_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_render_bundle_encoder")

    def create_query_set(self, descriptor: Any) -> 'QuerySet':
        """Creates a query set."""
        from .query_set import QuerySet
        if hasattr(self._inner, 'create_query_set'):
            query_set_inner = self._inner.create_query_set(descriptor)
            return QuerySet(query_set_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_query_set")

    def create_blas(self, descriptor: Any, geometry_descriptors: Any) -> 'Blas':
        """Creates a Bottom-Level Acceleration Structure."""
        from .rt import Blas
        if hasattr(self._inner, 'create_blas'):
            blas_inner = self._inner.create_blas(descriptor, geometry_descriptors)
            return Blas(blas_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_blas")

    def create_tlas(self, descriptor: Any) -> 'Tlas':
        """Creates a Top-Level Acceleration Structure."""
        from .rt import Tlas
        if hasattr(self._inner, 'create_tlas'):
            tlas_inner = self._inner.create_tlas(descriptor)
            return Tlas(tlas_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_tlas")

    def create_compute_pipeline(self, descriptor: Any) -> 'ComputePipeline':
        """Creates a compute pipeline."""
        from .compute_pipeline import ComputePipeline
        if hasattr(self._inner, 'create_compute_pipeline'):
            pipeline_inner = self._inner.create_compute_pipeline(descriptor)
            return ComputePipeline(pipeline_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_compute_pipeline")

    def create_render_pipeline(self, descriptor: RenderPipelineDescriptor) -> 'RenderPipeline':
        """Creates a render pipeline."""
        # from .render_pipeline import RenderPipeline # Removed runtime import
        if hasattr(self._inner, 'create_render_pipeline'):
            inner = self._inner.create_render_pipeline(descriptor.model_dump(exclude_none=True))
            return RenderPipeline(inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_render_pipeline")

    def create_mesh_pipeline(self, descriptor: MeshPipelineDescriptor) -> 'RenderPipeline':
        """Creates a mesh pipeline."""
        from .render_pipeline import RenderPipeline # Added runtime import for RenderPipeline
        if hasattr(self._inner, 'create_mesh_pipeline'):
            inner = self._inner.create_mesh_pipeline(descriptor.model_dump(exclude_none=True))
            return RenderPipeline(inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_mesh_pipeline")

    def create_pipeline_cache(self, descriptor: Any = None) -> 'PipelineCache':
        """Creates a pipeline cache."""
        from .pipeline_cache import PipelineCache
        if hasattr(self._inner, 'create_pipeline_cache'):
            cache_inner = self._inner.create_pipeline_cache(descriptor)
            return PipelineCache(cache_inner, descriptor)
        else:
            raise NotImplementedError("Backend does not support create_pipeline_cache")

    def destroy(self) -> None:
        """Destroys the device."""
        if hasattr(self._inner, 'destroy'):
            self._inner.destroy()
        # Clean up our references
        self._inner = None

    def set_device_lost_callback(self, callback: Any) -> None:
        """Sets a callback to be invoked when the device is lost."""
        if hasattr(self._inner, 'set_device_lost_callback'):
            self._inner.set_device_lost_callback(callback)
        # Store callback for future reference
        self._device_lost_callback = callback

    def on_uncaptured_error(self, callback: Any) -> None:
        """Sets a callback for uncaptured errors."""
        if hasattr(self._inner, 'on_uncaptured_error'):
            self._inner.on_uncaptured_error(callback)
        # Store callback for future reference
        self._uncaptured_error_callback = callback
