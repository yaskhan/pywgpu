from typing import Optional, Any, Tuple, List
from pywgpu_types.instance import InstanceDescriptor
from pywgpu_types.descriptors import RequestAdapterOptions, DeviceDescriptor
from pywgpu_core.global_ import Global
import asyncio


class WgpuCoreBackend:
    """
    Backend implementation using pywgpu-core.
    """

    def __init__(self, descriptor: Optional[InstanceDescriptor] = None) -> None:
        self._descriptor = descriptor or InstanceDescriptor()
        # Initialize the global state of wgpu-core
        self._global = Global("pywgpu", self._descriptor)

    async def request_adapter(
        self, options: Optional[RequestAdapterOptions] = None
    ) -> Optional["WgpuCoreAdapter"]:
        """
        Requests an adapter from the core logic.
        """
        options = options or RequestAdapterOptions()

        # Simulate async behavior for consistency with WebGPU API
        # The actual adapter request is synchronous in pywgpu-core
        await asyncio.sleep(0)

        try:
            adapter_id = self._global.instance_request_adapter(options)
            return WgpuCoreAdapter(self._global, adapter_id)
        except Exception as e:
            print(f"Failed to request adapter: {e}")
            return None

    def create_surface(self, target: Any) -> "WgpuCoreSurface":
        """
        Creates a surface using core logic.
        """
        surface_id = self._global.instance_create_surface(target)
        return WgpuCoreSurface(self._global, surface_id)

    def poll_all_devices(self, force_wait: bool = False) -> bool:
        """
        Polls all devices.
        """
        return self._global.poll_all_devices(force_wait)

    def generate_report(self) -> dict:
        """
        Generates a report from core logic.
        """
        return self._global.generate_report()


class WgpuCoreAdapter:
    def __init__(self, global_state: Global, adapter_id: Any) -> None:
        self._global = global_state
        self._adapter_id = adapter_id

    async def request_device(
        self, descriptor: Optional[DeviceDescriptor] = None
    ) -> Tuple["WgpuCoreDevice", "WgpuCoreQueue"]:
        await asyncio.sleep(0)
        # Global.adapter_request_device returns (device_id, queue_id)
        device_id, queue_id = self._global.adapter_request_device(
            self._adapter_id, descriptor
        )
        return WgpuCoreDevice(self._global, device_id), WgpuCoreQueue(
            self._global, queue_id
        )

    def features(self) -> Any:
        # Get from hub
        return self._global.hub.adapters.get(self._adapter_id).features

    def limits(self) -> Any:
        return self._global.hub.adapters.get(self._adapter_id).limits

    def get_info(self) -> Any:
        return self._global.hub.adapters.get(self._adapter_id).get_info()


class WgpuCoreDevice:
    def __init__(self, global_state: Global, device_id: Any) -> None:
        self._global = global_state
        self._device_id = device_id

    def create_buffer(self, descriptor: Any) -> "WgpuCoreBuffer":
        buffer_id, error = self._global.ops.device_create_buffer(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create buffer: {error}")
        return WgpuCoreBuffer(self._global, buffer_id)

    def create_texture(self, descriptor: Any) -> "WgpuCoreTexture":
        texture_id, error = self._global.ops.device_create_texture(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create texture: {error}")
        return WgpuCoreTexture(self._global, texture_id)

    def create_command_encoder(self, descriptor: Any) -> "WgpuCoreCommandEncoder":
        encoder_id, error = self._global.ops.device_create_command_encoder(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create command encoder: {error}")
        return WgpuCoreCommandEncoder(self._global, encoder_id)

    def create_shader_module(self, descriptor: Any) -> "WgpuCoreShaderModule":
        # Split descriptor to wgsl_code/spirv_data
        source = getattr(descriptor, "wgsl_code", None)
        shader_id, error = self._global.ops.device_create_shader_module(
            self._device_id, descriptor, source
        )
        if error:
            raise RuntimeError(f"Failed to create shader module: {error}")
        return WgpuCoreShaderModule(self._global, shader_id)

    def create_render_pipeline(self, descriptor: Any) -> "WgpuCoreRenderPipeline":
        pipeline_id, error = self._global.ops.device_create_render_pipeline(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create render pipeline: {error}")
        return WgpuCoreRenderPipeline(self._global, pipeline_id)

    def create_compute_pipeline(self, descriptor: Any) -> "WgpuCoreComputePipeline":
        pipeline_id, error = self._global.ops.device_create_compute_pipeline(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create compute pipeline: {error}")
        return WgpuCoreComputePipeline(self._global, pipeline_id)

    def create_pipeline_layout(self, descriptor: Any) -> "WgpuCorePipelineLayout":
        layout_id, error = self._global.ops.device_create_pipeline_layout(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create pipeline layout: {error}")
        return WgpuCorePipelineLayout(self._global, layout_id)

    def create_bind_group_layout(self, descriptor: Any) -> "WgpuCoreBindGroupLayout":
        layout_id, error = self._global.ops.device_create_bind_group_layout(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create bind group layout: {error}")
        return WgpuCoreBindGroupLayout(self._global, layout_id)

    def create_bind_group(self, descriptor: Any) -> "WgpuCoreBindGroup":
        group_id, error = self._global.ops.device_create_bind_group(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create bind group: {error}")
        return WgpuCoreBindGroup(self._global, group_id)

    def create_sampler(self, descriptor: Any) -> "WgpuCoreSampler":
        sampler_id, error = self._global.ops.device_create_sampler(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create sampler: {error}")
        return WgpuCoreSampler(self._global, sampler_id)

    def create_pipeline_cache(self, descriptor: Any) -> "WgpuCorePipelineCache":
        cache_id, error = self._global.ops.device_create_pipeline_cache(
            self._device_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to create pipeline cache: {error}")
        return WgpuCorePipelineCache(self._global, cache_id)


class WgpuCoreBuffer:
    def __init__(self, global_state: Global, buffer_id: Any) -> None:
        self._global = global_state
        self._buffer_id = buffer_id

    def destroy(self) -> None:
        self._global.ops.buffer_destroy(self._buffer_id)


class WgpuCoreTexture:
    def __init__(self, global_state: Global, texture_id: Any) -> None:
        self._global = global_state
        self._texture_id = texture_id


class WgpuCoreShaderModule:
    def __init__(self, global_state: Global, shader_id: Any) -> None:
        self._global = global_state
        self._shader_id = shader_id


class WgpuCoreRenderPipeline:
    def __init__(self, global_state: Global, pipeline_id: Any) -> None:
        self._global = global_state
        self._pipeline_id = pipeline_id


class WgpuCoreComputePipeline:
    def __init__(self, global_state: Global, pipeline_id: Any) -> None:
        self._global = global_state
        self._pipeline_id = pipeline_id


class WgpuCorePipelineLayout:
    def __init__(self, global_state: Global, layout_id: Any) -> None:
        self._global = global_state
        self._layout_id = layout_id


class WgpuCoreBindGroupLayout:
    def __init__(self, global_state: Global, layout_id: Any) -> None:
        self._global = global_state
        self._layout_id = layout_id


class WgpuCoreBindGroup:
    def __init__(self, global_state: Global, group_id: Any) -> None:
        self._global = global_state
        self._group_id = group_id


class WgpuCoreSampler:
    def __init__(self, global_state: Global, sampler_id: Any) -> None:
        self._global = global_state
        self._sampler_id = sampler_id


class WgpuCorePipelineCache:
    def __init__(self, global_state: Global, cache_id: Any) -> None:
        self._global = global_state
        self._cache_id = cache_id

    def get_data(self) -> Optional[bytes]:
        return self._global.ops.pipeline_cache_get_data(self._cache_id)


class WgpuCoreCommandEncoder:
    def __init__(self, global_state: Global, encoder_id: Any) -> None:
        self._global = global_state
        self._encoder_id = encoder_id

    def finish(self, descriptor: Any = None) -> Any:
        command_buffer_id, error = self._global.ops.command_encoder_finish(
            self._encoder_id, descriptor
        )
        if error:
            raise RuntimeError(f"Failed to finish command encoder: {error}")
        return WgpuCoreCommandBuffer(self._global, command_buffer_id)


class WgpuCoreQueue:
    def __init__(self, global_state: Global, queue_id: Any) -> None:
        self._global = global_state
        self._queue_id = queue_id

    def submit(self, command_buffers: List["WgpuCoreCommandBuffer"]) -> None:
        command_buffer_ids = [cb._id for cb in command_buffers]
        error = self._global.ops.queue_submit(self._queue_id, command_buffer_ids)
        if error:
            raise RuntimeError(f"Failed to submit queue: {error}")


class WgpuCoreSurface:
    def __init__(self, global_state: Global, surface_id: Any) -> None:
        self._global = global_state
        self._surface_id = surface_id

    def configure(self, device: "WgpuCoreDevice", config: Any) -> None:
        error = self._global.ops.surface_configure(self._surface_id, device._device_id, config)
        if error:
            raise RuntimeError(f"Failed to configure surface: {error}")

    def get_current_texture(self) -> Tuple[Any, bool, Any]:
        result, error = self._global.ops.surface_get_current_texture(self._surface_id)
        if error:
            raise RuntimeError(f"Failed to get current texture: {error}")
        # Returns (texture_id, suboptimal, status)
        return result

class WgpuCoreCommandBuffer:
    def __init__(self, global_state: Global, command_buffer_id: Any) -> None:
        self._global = global_state
        self._id = command_buffer_id
