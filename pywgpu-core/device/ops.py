from typing import Any, Optional, Tuple
from .resource import Device, BufferDescriptor, TextureDescriptor


class DeviceOps:
    """
    Device operations for Global.
    
    This class provides methods for device resource creation and management.
    It acts as a bridge between the Global state and the HAL device operations.
    Methods return (resource_id, error) tuples following wgpu-core conventions.
    
    Attributes:
        hub: Reference to the hub for resource registry (set by Global).
    """
    
    def __init__(self):
        """Initialize DeviceOps."""
        self.hub = None  # Will be set by Global
    
    def _get_device(self, device_id: Any) -> Device:
        """Get device from hub registry."""
        if self.hub is None:
            raise RuntimeError("DeviceOps not connected to hub")
        if not hasattr(self.hub, 'devices'):
            raise RuntimeError("Hub does not have devices registry")
        if device_id not in self.hub.devices:
            raise RuntimeError(f"Device {device_id} not found in registry")
        return self.hub.devices[device_id]
    
    def device_features(self, device_id: Any) -> Any:
        """Get device features."""
        try:
            device = self._get_device(device_id)
            return getattr(device, 'features', 0)
        except Exception:
            return 0

    def device_limits(self, device_id: Any) -> Any:
        """Get device limits."""
        try:
            device = self._get_device(device_id)
            return getattr(device, 'limits', {})
        except Exception:
            return {}

    def device_create_buffer(
        self,
        device_id: Any,
        desc: BufferDescriptor,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a buffer on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create buffer using device method
            buffer = device.create_buffer(desc)
            
            # Register in hub and get ID
            if self.hub and hasattr(self.hub, 'buffers'):
                buffer_id = id_in if id_in is not None else self.hub._next_id('buffer')
                self.hub.buffers[buffer_id] = buffer
                return (buffer_id, None)
            
            return (None, "Hub not available for buffer registration")
        except Exception as e:
            return (None, str(e))

    def device_create_texture(
        self,
        device_id: Any,
        desc: TextureDescriptor,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a texture on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create texture using device method
            texture = device.create_texture(desc)
            
            # Register in hub and get ID
            if self.hub and hasattr(self.hub, 'textures'):
                texture_id = id_in if id_in is not None else self.hub._next_id('texture')
                self.hub.textures[texture_id] = texture
                return (texture_id, None)
            
            return (None, "Hub not available for texture registration")
        except Exception as e:
            return (None, str(e))

    def device_adapter_info(self, device_id: Any) -> Any:
        """Get adapter info for device."""
        try:
            device = self._get_device(device_id)
            adapter = getattr(device, 'adapter', None)
            if adapter and hasattr(adapter, 'get_info'):
                return adapter.get_info()
            return {}
        except Exception:
            return {}

    def device_downlevel_properties(self, device_id: Any) -> Any:
        """Get downlevel properties for device."""
        try:
            device = self._get_device(device_id)
            return getattr(device, 'downlevel', {})
        except Exception:
            return {}

    def device_create_command_encoder(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a command encoder on the device."""
        try:
            device = self._get_device(device_id)
            
            # Import command encoder
            from ..command.encoder import CommandEncoder
            encoder = CommandEncoder(device)
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'command_encoders'):
                encoder_id = id_in if id_in is not None else self.hub._next_id('command_encoder')
                self.hub.command_encoders[encoder_id] = encoder
                return (encoder_id, None)
            
            return (None, "Hub not available for encoder registration")
        except Exception as e:
            return (None, str(e))

    def device_create_render_pipeline(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a render pipeline on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create render pipeline (would use HAL device method)
            # For now, create a placeholder object
            pipeline = {"type": "render_pipeline", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'render_pipelines'):
                pipeline_id = id_in if id_in is not None else self.hub._next_id('render_pipeline')
                self.hub.render_pipelines[pipeline_id] = pipeline
                return (pipeline_id, None)
            
            return (None, "Hub not available for pipeline registration")
        except Exception as e:
            return (None, str(e))

    def device_create_compute_pipeline(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a compute pipeline on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create compute pipeline (would use HAL device method)
            pipeline = {"type": "compute_pipeline", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'compute_pipelines'):
                pipeline_id = id_in if id_in is not None else self.hub._next_id('compute_pipeline')
                self.hub.compute_pipelines[pipeline_id] = pipeline
                return (pipeline_id, None)
            
            return (None, "Hub not available for pipeline registration")
        except Exception as e:
            return (None, str(e))

    def device_create_bind_group_layout(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a bind group layout on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create bind group layout
            layout = {"type": "bind_group_layout", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'bind_group_layouts'):
                layout_id = id_in if id_in is not None else self.hub._next_id('bind_group_layout')
                self.hub.bind_group_layouts[layout_id] = layout
                return (layout_id, None)
            
            return (None, "Hub not available for layout registration")
        except Exception as e:
            return (None, str(e))

    def device_create_pipeline_layout(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a pipeline layout on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create pipeline layout
            layout = {"type": "pipeline_layout", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'pipeline_layouts'):
                layout_id = id_in if id_in is not None else self.hub._next_id('pipeline_layout')
                self.hub.pipeline_layouts[layout_id] = layout
                return (layout_id, None)
            
            return (None, "Hub not available for layout registration")
        except Exception as e:
            return (None, str(e))

    def device_create_bind_group(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a bind group on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create bind group
            bind_group = {"type": "bind_group", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'bind_groups'):
                group_id = id_in if id_in is not None else self.hub._next_id('bind_group')
                self.hub.bind_groups[group_id] = bind_group
                return (group_id, None)
            
            return (None, "Hub not available for bind group registration")
        except Exception as e:
            return (None, str(e))

    def device_create_shader_module(
        self,
        device_id: Any,
        desc: Any,
        source: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a shader module on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create shader module
            shader = {"type": "shader_module", "desc": desc, "source": source, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'shader_modules'):
                shader_id = id_in if id_in is not None else self.hub._next_id('shader_module')
                self.hub.shader_modules[shader_id] = shader
                return (shader_id, None)
            
            return (None, "Hub not available for shader registration")
        except Exception as e:
            return (None, str(e))

    def device_create_sampler(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a sampler on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create sampler
            sampler = {"type": "sampler", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'samplers'):
                sampler_id = id_in if id_in is not None else self.hub._next_id('sampler')
                self.hub.samplers[sampler_id] = sampler
                return (sampler_id, None)
            
            return (None, "Hub not available for sampler registration")
        except Exception as e:
            return (None, str(e))

    def device_create_query_set(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a query set on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create query set
            query_set = {"type": "query_set", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'query_sets'):
                query_id = id_in if id_in is not None else self.hub._next_id('query_set')
                self.hub.query_sets[query_id] = query_set
                return (query_id, None)
            
            return (None, "Hub not available for query set registration")
        except Exception as e:
            return (None, str(e))

    def device_create_texture_view(
        self,
        device_id: Any,
        texture_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a texture view on the device."""
        try:
            device = self._get_device(device_id)
            
            # Get texture from hub
            if not self.hub or not hasattr(self.hub, 'textures'):
                return (None, "Hub not available")
            
            if texture_id not in self.hub.textures:
                return (None, f"Texture {texture_id} not found")
            
            texture = self.hub.textures[texture_id]
            
            # Create texture view
            view = {"type": "texture_view", "texture": texture, "desc": desc, "device": device}
            
            # Register in hub
            if hasattr(self.hub, 'texture_views'):
                view_id = id_in if id_in is not None else self.hub._next_id('texture_view')
                self.hub.texture_views[view_id] = view
                return (view_id, None)
            
            return (None, "Hub not available for view registration")
        except Exception as e:
            return (None, str(e))

    def device_poll(
        self,
        device_id: Any,
        maintain: Any = None
    ) -> Any:
        """Poll device for events."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'poll'):
                return device.poll(maintain)
            return None
        except Exception:
            return None

    def device_tick(self, device_id: Any) -> None:
        """Tick the device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'tick'):
                device.tick()
        except Exception:
            pass

    def device_destroy(self, device_id: Any) -> None:
        """Destroy the device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'destroy'):
                device.destroy()
            
            # Remove from hub
            if self.hub and hasattr(self.hub, 'devices') and device_id in self.hub.devices:
                del self.hub.devices[device_id]
        except Exception:
            pass

    def device_set_lost_closure(
        self,
        device_id: Any,
        closure: Any
    ) -> None:
        """Set lost closure for device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'set_lost_closure'):
                device.set_lost_closure(closure)
        except Exception:
            pass

    def device_get_timestamp_period(self, device_id: Any) -> float:
        """Get timestamp period for device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'get_timestamp_period'):
                return device.get_timestamp_period()
            return 1.0
        except Exception:
            return 1.0

    def device_get_hal_counters(self, device_id: Any) -> Any:
        """Get HAL counters for device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'get_hal_counters'):
                return device.get_hal_counters()
            return {}
        except Exception:
            return {}

    def device_generate_allocator_report(self, device_id: Any) -> Any:
        """Generate allocator report for device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'generate_allocator_report'):
                return device.generate_allocator_report()
            return None
        except Exception:
            return None

    def device_configure_surface(
        self,
        device_id: Any,
        surface_id: Any,
        config: Any
    ) -> Optional[Any]:
        """Configure surface for device."""
        try:
            device = self._get_device(device_id)
            if hasattr(device, 'configure_surface'):
                return device.configure_surface(surface_id, config)
            return None
        except Exception as e:
            return str(e)

    def device_create_render_bundle_encoder(
        self,
        device_id: Any,
        desc: Any
    ) -> Tuple[Any, Optional[Any]]:
        """Create a render bundle encoder on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create render bundle encoder
            encoder = {"type": "render_bundle_encoder", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'render_bundle_encoders'):
                encoder_id = self.hub._next_id('render_bundle_encoder')
                self.hub.render_bundle_encoders[encoder_id] = encoder
                return (encoder_id, None)
            
            return (None, "Hub not available for encoder registration")
        except Exception as e:
            return (None, str(e))

    def render_bundle_encoder_finish(
        self,
        bundle_encoder: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Finish a render bundle encoder."""
        try:
            # Create render bundle
            bundle = {"type": "render_bundle", "encoder": bundle_encoder, "desc": desc}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'render_bundles'):
                bundle_id = id_in if id_in is not None else self.hub._next_id('render_bundle')
                self.hub.render_bundles[bundle_id] = bundle
                return (bundle_id, None)
            
            return (None, "Hub not available for bundle registration")
        except Exception as e:
            return (None, str(e))

    def device_create_pipeline_cache(
        self,
        device_id: Any,
        desc: Any,
        id_in: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Create a pipeline cache on the device."""
        try:
            device = self._get_device(device_id)
            
            # Create pipeline cache
            if hasattr(device, 'create_pipeline_cache'):
                cache = device.create_pipeline_cache(desc)
            else:
                cache = {"type": "pipeline_cache", "desc": desc, "device": device}
            
            # Register in hub
            if self.hub and hasattr(self.hub, 'pipeline_caches'):
                cache_id = id_in if id_in is not None else self.hub._next_id('pipeline_cache')
                self.hub.pipeline_caches[cache_id] = cache
                return (cache_id, None)
            
            return (None, "Hub not available for cache registration")
        except Exception as e:
            return (None, str(e))
