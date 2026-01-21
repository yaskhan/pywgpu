from pydantic import BaseModel


class Limits(BaseModel):
    """
    Resource limits for a device.
    """

    max_texture_dimension_1d: int = 8192
    max_texture_dimension_2d: int = 8192
    max_texture_dimension_3d: int = 2048
    max_texture_array_layers: int = 256
    max_bind_groups: int = 4
    max_bindings_per_bind_group: int = 1000
    max_dynamic_uniform_buffers_per_pipeline_layout: int = 8
    max_dynamic_storage_buffers_per_pipeline_layout: int = 4
    max_sampled_textures_per_shader_stage: int = 16
    max_samplers_per_shader_stage: int = 16
    max_storage_buffers_per_shader_stage: int = 8
    max_storage_textures_per_shader_stage: int = 4
    max_uniform_buffers_per_shader_stage: int = 12
    max_uniform_buffer_binding_size: int = 16384
    max_storage_buffer_binding_size: int = 128 << 20
    min_uniform_buffer_offset_alignment: int = 256
    min_storage_buffer_offset_alignment: int = 256
    max_vertex_buffers: int = 8
    max_buffer_size: int = 256 << 20
    max_vertex_attributes: int = 16
    max_vertex_buffer_array_stride: int = 2048
    max_inter_stage_shader_components: int = 60
    max_inter_stage_shader_variables: int = 16
    max_color_attachments: int = 8
    max_color_attachment_bytes_per_sample: int = 32
    max_compute_workgroup_storage_size: int = 16352
    max_compute_invocations_per_workgroup: int = 256
    max_compute_workgroup_size_x: int = 256
    max_compute_workgroup_size_y: int = 256
    max_compute_workgroup_size_z: int = 64
    max_compute_workgroups_per_dimension: int = 65535

    @classmethod
    def default(cls) -> "Limits":
        return cls()

    @classmethod
    def downlevel_defaults(cls) -> "Limits":
        """Limits guaranteed on all downlevel capabilities."""
        return cls(
            max_texture_dimension_1d=2048,
            max_texture_dimension_2d=2048,
            max_texture_dimension_3d=256,
            max_texture_array_layers=256,
            max_bind_groups=4,
            max_dynamic_uniform_buffers_per_pipeline_layout=8,
            max_dynamic_storage_buffers_per_pipeline_layout=4,
            max_sampled_textures_per_shader_stage=16,
            max_samplers_per_shader_stage=16,
            max_storage_buffers_per_shader_stage=4,
            max_storage_textures_per_shader_stage=4,
            max_uniform_buffers_per_shader_stage=12,
            max_uniform_buffer_binding_size=16384,
            max_storage_buffer_binding_size=128 << 20,
            max_vertex_buffers=8,
            max_vertex_attributes=16,
            max_vertex_buffer_array_stride=2048,
            max_compute_workgroup_storage_size=16352,
            max_compute_invocations_per_workgroup=256,
            max_compute_workgroup_size_x=256,
            max_compute_workgroup_size_y=256,
            max_compute_workgroup_size_z=64,
            max_compute_workgroups_per_dimension=65535,
        )
