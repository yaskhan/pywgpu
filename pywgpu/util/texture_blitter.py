"""
Texture blitting utilities.

This module provides utilities for copying/blitting textures with format conversion
and scaling support. Use this when CommandEncoder.copy_texture_to_texture won't work
because:
- Textures are in incompatible formats
- Textures are of different sizes
- Your copy destination is the surface texture without COPY_DST usage
"""

from typing import Optional


# WGSL shader for texture blitting
BLIT_SHADER = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;

    out.tex_coords = vec2<f32>(
        f32((vi << 1u) & 2u),
        f32(vi & 2u),
    );

    out.position = vec4<f32>(out.tex_coords * 2.0 - 1.0, 0.0, 1.0);

    // Invert y so the texture is not upside down
    out.tex_coords.y = 1.0 - out.tex_coords.y;
    return out;
}

@group(0) @binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

@fragment
fn fs_main(vs: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture, texture_sampler, vs.tex_coords);
}
"""


class TextureBlitterBuilder:
    """
    Builder for TextureBlitter utility.

    If you want the default TextureBlitter, use TextureBlitter.new() instead.

    Attributes:
        device: The device to create resources on.
        format: The texture format of the destination texture.
        sample_type: The sampler filtering mode.
        blend_state: Optional blend state for blitting.
    """

    def __init__(self, device, format):
        """
        Create a new TextureBlitterBuilder.

        Args:
            device: A Device instance.
            format: The TextureFormat of the texture that will be copied to.
                   This must have the RENDER_TARGET usage.
        """
        self.device = device
        self.format = format
        self.sample_type = "nearest"  # FilterMode.Nearest
        self.blend_state = None

    def with_sample_type(self, sample_type):
        """
        Set the sampler filtering mode.

        Args:
            sample_type: FilterMode ('nearest' or 'linear').

        Returns:
            Self for method chaining.
        """
        self.sample_type = sample_type
        return self

    def with_blend_state(self, blend_state):
        """
        Set the blend state.

        Args:
            blend_state: BlendState configuration.

        Returns:
            Self for method chaining.
        """
        self.blend_state = blend_state
        return self

    def build(self):
        """
        Build the TextureBlitter with the configured settings.

        Returns:
            A new TextureBlitter instance.
        """
        # Create sampler
        sampler = self.device.create_sampler(
            label="wgpu::util::TextureBlitter::sampler",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge",
            mag_filter=self.sample_type,
            min_filter=self.sample_type,
            mipmap_filter="nearest",
        )

        # Determine if filterable
        filterable = self.sample_type == "linear"

        # Create bind group layout
        bind_group_layout = self.device.create_bind_group_layout(
            label="wgpu::util::TextureBlitter::bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": "fragment",  # ShaderStages.FRAGMENT
                    "texture": {
                        "sample_type": "float" if filterable else "unfilterable-float",
                        "view_dimension": "2d",
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": "fragment",  # ShaderStages.FRAGMENT
                    "sampler": {"type": "filtering" if filterable else "non-filtering"},
                },
            ],
        )

        # Create pipeline layout
        pipeline_layout = self.device.create_pipeline_layout(
            label="wgpu::util::TextureBlitter::pipeline_layout",
            bind_group_layouts=[bind_group_layout],
        )

        # Create shader module
        shader = self.device.create_shader_module(
            label="wgpu::util::TextureBlitter::shader", code=BLIT_SHADER
        )

        # Create render pipeline
        pipeline = self.device.create_render_pipeline(
            label="wgpu::util::TextureBlitter::pipeline",
            layout=pipeline_layout,
            vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
            primitive={
                "topology": "triangle-list",
                "strip_index_format": None,
                "front_face": "ccw",
                "cull_mode": None,
            },
            depth_stencil=None,
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.format,
                        "blend": self.blend_state,
                        "write_mask": 0xF,  # ColorWrites.ALL
                    }
                ],
            },
        )

        return TextureBlitter(pipeline, bind_group_layout, sampler)


class TextureBlitter:
    """
    Texture Blitting (Copying) Utility.

    Use this if you want to render/copy texture A to texture B where
    CommandEncoder.copy_texture_to_texture would not work because:
    - Textures are in incompatible formats
    - Textures are of different sizes
    - Your copy destination is the surface texture without COPY_DST usage

    Attributes:
        pipeline: The render pipeline for blitting.
        bind_group_layout: The bind group layout.
        sampler: The sampler for texture sampling.
    """

    def __init__(self, pipeline, bind_group_layout, sampler):
        """
        Create a TextureBlitter.

        Note: Use TextureBlitter.new() or TextureBlitterBuilder instead.

        Args:
            pipeline: RenderPipeline for blitting.
            bind_group_layout: BindGroupLayout for texture binding.
            sampler: Sampler for texture sampling.
        """
        self.pipeline = pipeline
        self.bind_group_layout = bind_group_layout
        self.sampler = sampler

    @staticmethod
    def new(device, format):
        """
        Create a TextureBlitter with default settings.

        Args:
            device: A Device instance.
            format: The TextureFormat of the texture that will be copied to.
                   This must have the RENDER_TARGET usage.

        Returns:
            A new TextureBlitter instance.

        Example:
            blitter = TextureBlitter.new(device, 'bgra8unorm-srgb')
            blitter.copy(device, encoder, source_view, target_view)
        """
        return TextureBlitterBuilder(device, format).build()

    def copy(self, device, encoder, source, target):
        """
        Copy data from source TextureView to target TextureView.

        Args:
            device: A Device instance.
            encoder: A CommandEncoder instance.
            source: A TextureView that gets copied. Format doesn't matter.
            target: A TextureView that receives the data. Must match the
                   format specified in TextureBlitter.new().

        Example:
            blitter = TextureBlitter.new(device, 'bgra8unorm-srgb')
            encoder = device.create_command_encoder()
            blitter.copy(device, encoder, source_view, target_view)
            device.queue.submit([encoder.finish()])
        """
        # Create bind group for this blit operation
        bind_group = device.create_bind_group(
            label="wgpu::util::TextureBlitter::bind_group",
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": source},
                {"binding": 1, "resource": self.sampler},
            ],
        )

        # Begin render pass
        render_pass = encoder.begin_render_pass(
            label="wgpu::util::TextureBlitter::pass",
            color_attachments=[
                {
                    "view": target,
                    "resolve_target": None,
                    "load_op": "load",
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment=None,
        )

        # Set pipeline and bind group
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, bind_group)

        # Draw fullscreen triangle
        render_pass.draw(3, 1)

        # End render pass
        render_pass.end()
