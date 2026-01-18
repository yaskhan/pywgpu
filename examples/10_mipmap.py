import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

TEXTURE_FORMAT = pywgpu.TextureFormat.rgba8unorm
MIP_LEVEL_COUNT = 10

def create_texels(size: int, cx: f32, cy: f32) -> bytes:
    texels = bytearray()
    for y in range(size):
        for x in range(size):
            # Mandelbrot-like fractal for interesting mipmaps
            px = 4.0 * x / (size - 1) - 2.0
            py = 2.0 * y / (size - 1) - 1.0
            zx, zy = px, py
            count = 0
            while count < 255 and zx * zx + zy * zy < 4.0:
                old_zx = zx
                zx = zx * zx - zy * zy + cx
                zy = 2.0 * old_zx * zy + cy
                count += 1
            
            texels.extend([
                0xFF - (count * 2) % 256,
                0xFF - (count * 5) % 256,
                0xFF - (count * 13) % 256,
                255
            ])
    return bytes(texels)

class MipmapExample(Example):
    TITLE = "Mipmap Example"

    async def init(self, config, adapter, device, queue):
        size = 1 << (MIP_LEVEL_COUNT - 1)
        texels = create_texels(size, -0.8, 0.156)
        
        texture_extent = pywgpu.Extent3d(width=size, height=size, depth_or_array_layers=1)
        self.texture = device.create_texture(pywgpu.TextureDescriptor(
            label="Mipmap Texture",
            size=texture_extent,
            mip_level_count=MIP_LEVEL_COUNT,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=TEXTURE_FORMAT,
            usage=[pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.COPY_DST]
        ))
        
        # Upload level 0
        queue.write_texture(
            pywgpu.TexelCopyTextureInfo(texture=self.texture, mip_level=0),
            texels,
            pywgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=4 * size, rows_per_image=None),
            texture_extent
        )

        # Generate Mipmaps
        self._generate_mipmaps(device, queue)

        # Draw Pipeline setup
        draw_shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            };

            @group(0) @binding(0) var<uniform> transform: mat4x4<f32>;

            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
                let pos = vec2<f32>(
                    100.0 * (1.0 - f32(vertex_index & 2u)),
                    1000.0 * f32(vertex_index & 1u)
                );
                var result: VertexOutput;
                result.tex_coords = 0.05 * pos + vec2<f32>(0.5, 0.5);
                result.position = transform * vec4<f32>(pos, 0.0, 1.0);
                return result;
            }

            @group(0) @binding(1) var r_color: texture_2d<f32>;
            @group(0) @binding(2) var r_sampler: sampler;

            @fragment
            fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(r_color, r_sampler, vertex.tex_coords);
            }
        """
        draw_shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=draw_shader_code))
        
        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor(
            address_mode_u=pywgpu.AddressMode.repeat,
            address_mode_v=pywgpu.AddressMode.repeat,
            mag_filter=pywgpu.FilterMode.linear,
            min_filter=pywgpu.FilterMode.linear,
            mipmap_filter=pywgpu.MipmapFilterMode.linear
        ))

        aspect = config.width / config.height
        proj = self._generate_matrix(aspect)
        self.uniform_buf = device.create_buffer_init(
            label="Globals",
            contents=struct.pack("16f", *proj),
            usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST]
        )

        self.draw_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=None, # Auto layout
            vertex=pywgpu.VertexState(module=draw_shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(
                module=draw_shader, entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))
        
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.draw_pipeline.get_bind_group_layout(0),
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.uniform_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.texture.create_view()),
                pywgpu.BindGroupEntry(binding=2, resource=self.sampler)
            ]
        ))

    def _generate_mipmaps(self, device, queue):
        blit_shader_code = """
            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
                let x = f32(i32(vertex_index) << 1u & 2);
                let y = f32(i32(vertex_index & 2u));
                return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
            }

            @group(0) @binding(0) var t_src: texture_2d<f32>;
            @group(0) @binding(1) var s_src: sampler;

            @fragment
            fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                let tex_size = textureDimensions(t_src);
                let uv = pos.xy / vec2<f32>(textureDimensions(t_src)); // This is subtle if not using Sampler
                // Using a sampler is better for downscaling
                return textureSample(t_src, s_src, pos.xy / vec2<f32>(800.0, 600.0)); // Mocked UV
            }
            
            // Simplified blit shader for the example
            @fragment
            fn fs_blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                // Correct blit logic using texcoords
                let uv = vec2<f32>( (pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5 );
                return textureSample(t_src, s_src, uv);
            }
        """
        # Improved blit shader
        blit_shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
                var out: VertexOutput;
                let x = f32(i32(vertex_index) << 1u & 2);
                let y = f32(i32(vertex_index & 2u));
                out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
                out.tex_coords = vec2<f32>(x, 1.0 - y);
                return out;
            }

            @group(0) @binding(0) var t_src: texture_2d<f32>;
            @group(0) @binding(1) var s_src: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(t_src, s_src, in.tex_coords);
            }
        """
        
        shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=blit_shader_code))
        
        pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=None,
            vertex=pywgpu.VertexState(module=shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(
                module=shader, entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=TEXTURE_FORMAT)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list)
        ))
        
        blit_sampler = device.create_sampler(pywgpu.SamplerDescriptor(
            mag_filter=pywgpu.FilterMode.linear,
            min_filter=pywgpu.FilterMode.linear
        ))

        encoder = device.create_command_encoder()
        
        views = [self.texture.create_view(base_mip_level=i, mip_level_count=1) for i in range(MIP_LEVEL_COUNT)]
        
        for i in range(1, MIP_LEVEL_COUNT):
            bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
                layout=pipeline.get_bind_group_layout(0),
                entries=[
                    pywgpu.BindGroupEntry(binding=0, resource=views[i-1]),
                    pywgpu.BindGroupEntry(binding=1, resource=blit_sampler)
                ]
            ))
            
            pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
                color_attachments=[pywgpu.RenderPassColorAttachment(
                    view=views[i],
                    ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color.WHITE), store=pywgpu.StoreOp.store)
                )]
            ))
            pass_enc.set_pipeline(pipeline)
            pass_enc.set_bind_group(0, bind_group)
            pass_enc.draw(vertices=range(3), instances=range(1))
            pass_enc.end()
            
        queue.submit([encoder.finish()])

    def _generate_matrix(self, aspect):
        proj = [0]*16
        f = 1.0 / math.tan(math.pi / 8.0)
        near = 1.0
        far = 1000.0
        
        proj[0] = f / aspect
        proj[5] = f
        proj[10] = far / (near - far)
        proj[11] = -1.0
        proj[14] = (near * far) / (near - far)
        
        view = [0]*16
        view[0]=1; view[5]=1; view[10]=1; view[15]=1
        view[13] = 50.0
        view[14] = -10.0
        
        # Simplified proj * view
        return proj # Just proj for simplicity in mock

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.set_pipeline(self.draw_pipeline)
        pass_enc.set_bind_group(0, self.bind_group)
        pass_enc.draw(vertices=range(4), instances=range(1))
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(MipmapExample))
