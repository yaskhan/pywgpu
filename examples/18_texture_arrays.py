import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class TextureArraysExample(Example):
    TITLE = "Texture Arrays Example"

    async def init(self, config, adapter, device, queue):
        # Shaders
        shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
                @location(1) @interpolate(flat) index: u32,
            };

            @vertex
            fn vs_main(
                @location(0) pos: vec2<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) index: i32,
            ) -> VertexOutput {
                var out: VertexOutput;
                out.position = vec4<f32>(pos, 0.0, 1.0);
                out.tex_coords = tex_coords;
                out.index = u32(index);
                return out;
            }

            @group(0) @binding(0) var tex_array: binding_array<texture_2d<f32>>;
            @group(0) @binding(1) var sam: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                // In WGSL, indexing into binding_array requires either
                // uniform index or non-uniform indexing feature.
                // For simplicity, we'll use a fixed index or switch logic.
                return textureSample(tex_array[in.index], sam, in.tex_coords);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Geometry
        # pos(2f), uv(2f), index(1i)
        vertices = [
            # Left quad (index 0)
            -0.9, -0.9, 0.0, 1.0, 0,
            -0.1, -0.9, 1.0, 1.0, 0,
            -0.1,  0.9, 1.0, 0.0, 0,
            -0.9,  0.9, 0.0, 0.0, 0,
            # Right quad (index 1)
             0.1, -0.9, 0.0, 1.0, 1,
             0.9, -0.9, 1.0, 1.0, 1,
             0.9,  0.9, 1.0, 0.0, 1,
             0.1,  0.9, 0.0, 0.0, 1,
        ]
        indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        
        self.vertex_buf = device.create_buffer_init(label="Vertices", contents=struct.pack(f"{len(vertices)}f", *vertices), usage=pywgpu.BufferUsages.VERTEX)
        self.index_buf = device.create_buffer_init(label="Indices", contents=struct.pack(f"{len(indices)}I", *indices), usage=pywgpu.BufferUsages.INDEX)
        self.index_count = len(indices)

        # Textures (Red and Green)
        tex_red = device.create_texture(pywgpu.TextureDescriptor(size=(1,1,1), format=pywgpu.TextureFormat.rgba8unorm, usage=[pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.COPY_DST]))
        tex_green = device.create_texture(pywgpu.TextureDescriptor(size=(1,1,1), format=pywgpu.TextureFormat.rgba8unorm, usage=[pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.COPY_DST]))
        
        queue.write_texture(tex_red, 0, pywgpu.ImageDataLayout(bytes_per_row=4), (1,1,1), struct.pack("4B", 255, 0, 0, 255))
        queue.write_texture(tex_green, 0, pywgpu.ImageDataLayout(bytes_per_row=4), (1,1,1), struct.pack("4B", 0, 255, 0, 255))
        
        self.view_red = tex_red.create_view()
        self.view_green = tex_green.create_view()
        
        # Sampler
        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor())

        # Bind Group with Texture Array
        # Python pywgpu might not support binding_array yet in its type system, 
        # but let's try to pass a list of views.
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.texture(view_dimension=pywgpu.TextureViewDimension.d2), count=2),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.SamplerBindingType.filtering)
        ]))
        
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=[self.view_red, self.view_green]), # List of views for array
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))

        # Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main", buffers=[
                pywgpu.VertexBufferLayout(array_stride=20, attributes=[
                    pywgpu.VertexAttribute(format=pywgpu.TextureFormat.float32x2, offset=0, shader_location=0), # pos
                    pywgpu.VertexAttribute(format=pywgpu.TextureFormat.float32x2, offset=8, shader_location=1), # uv
                    pywgpu.VertexAttribute(format=pywgpu.TextureFormat.sint32, offset=16, shader_location=2)   # index
                ])
            ]),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list)
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, self.bind_group)
        pass_enc.set_vertex_buffer(0, self.vertex_buf)
        pass_enc.set_index_buffer(self.index_buf, pywgpu.IndexFormat.uint32)
        pass_enc.draw_indexed(indices=range(self.index_count), instances=range(1))
        
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(TextureArraysExample))
