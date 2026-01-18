from typing import Any, Dict
from .registry import Registry


class Hub:
    """
    Hub for all resource registries.
    """

    def __init__(self) -> None:
        self.adapters: Registry[Any] = Registry()
        self.devices: Registry[Any] = Registry()
        self.queues: Registry[Any] = Registry()
        self.pipeline_layouts: Registry[Any] = Registry()
        self.shader_modules: Registry[Any] = Registry()
        self.bind_group_layouts: Registry[Any] = Registry()
        self.bind_groups: Registry[Any] = Registry()
        self.command_encoders: Registry[Any] = Registry()
        self.command_buffers: Registry[Any] = Registry()
        self.render_bundles: Registry[Any] = Registry()
        self.render_pipelines: Registry[Any] = Registry()
        self.compute_pipelines: Registry[Any] = Registry()
        self.pipeline_caches: Registry[Any] = Registry()
        self.query_sets: Registry[Any] = Registry()
        self.buffers: Registry[Any] = Registry()
        self.textures: Registry[Any] = Registry()
        self.texture_views: Registry[Any] = Registry()
        self.samplers: Registry[Any] = Registry()

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a report of all resources in the hub.

        Returns:
            A dictionary containing reports from each registry.
        """
        return {
            "adapters": self.adapters.generate_report(),
            "devices": self.devices.generate_report(),
            "queues": self.queues.generate_report(),
            "pipeline_layouts": self.pipeline_layouts.generate_report(),
            "shader_modules": self.shader_modules.generate_report(),
            "bind_group_layouts": self.bind_group_layouts.generate_report(),
            "bind_groups": self.bind_groups.generate_report(),
            "command_encoders": self.command_encoders.generate_report(),
            "command_buffers": self.command_buffers.generate_report(),
            "render_bundles": self.render_bundles.generate_report(),
            "render_pipelines": self.render_pipelines.generate_report(),
            "compute_pipelines": self.compute_pipelines.generate_report(),
            "pipeline_caches": self.pipeline_caches.generate_report(),
            "query_sets": self.query_sets.generate_report(),
            "buffers": self.buffers.generate_report(),
            "textures": self.textures.generate_report(),
            "texture_views": self.texture_views.generate_report(),
            "samplers": self.samplers.generate_report(),
        }
