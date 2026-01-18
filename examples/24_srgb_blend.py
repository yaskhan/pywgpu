import asyncio
import struct
from typing import List

import pywgpu
from framework import Example, run_example

class SrgbBlendExample(Example):
    TITLE = "SRGB Blend Example"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            struct VertexOutput {
                @location(0) color: vec4<f32>,
                @builtin(position) position: vec4<f32>,
            };

            @vertex
            fn vs_main(
                @location(0) position: vec4<f32>,
                @location(1) color: vec4<f32>,
            ) -> VertexOutput {
                var result: VertexOutput;
                result.color = color;
                result.position = position;
                return result;
            }

            @fragment
            fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
                return vertex.color;
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Vertex Data
        # We'll create two overlapping quads: red and blue, both with 0.5 alpha.
        vertices = []
        indices = []
        
        def add_quad(color, offset):
            base = len(vertices) // 8
            scale = 0.5
            # pos(4f), color(4f)
            q_verts = [
                (-1.0 + offset) * scale, (-1.0 + offset) * scale, 0.0, 1.0, *color,
                ( 1.0 + offset) * scale, (-1.0 + offset) * scale, 0.0, 1.0, *color,
                ( 1.0 + offset) * scale, ( 1.0 + offset) * scale, 0.0, 1.0, *color,
                (-1.0 + offset) * scale, ( 1.0 + offset) * scale, 0.0, 1.0, *color,
            ]
            vertices.extend(q_verts)
            indices.extend([base + 0, base + 1, base + 2, base + 2, base + 3, base + 0])

        add_quad([1.0, 0.0, 0.0, 0.5], 0.5)
        add_quad([0.0, 0.0, 1.0, 0.5], -0.5)

        self.vertex_buf = device.create_buffer_init(
            label="Vertex Buffer",
            contents=struct.pack(f"{len(vertices)}f", *vertices),
            usage=pywgpu.BufferUsages.VERTEX
        )
        self.index_buf = device.create_buffer_init(
            label="Index Buffer",
            contents=struct.pack(f"{len(indices)}H", *indices),
            usage=pywgpu.BufferUsages.INDEX
        )
        self.index_count = len(indices)

        # 3. Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(
                module=self.shader,
                entry_point="vs_main",
                buffers=[pywgpu.VertexBufferLayout(
                    array_stride=32,
                    attributes=[
                        pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x4, offset=0, shader_location=0),
                        pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x4, offset=16, shader_location=1),
                    ]
                )]
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(
                    format=config.format,
                    blend=pywgpu.BlendState.alpha_blending()
                )]
            ),
            primitive=pywgpu.PrimitiveState(cull_mode=pywgpu.Face.back)
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_index_buffer(self.index_buf, pywgpu.IndexFormat.uint16)
        pass_enc.set_vertex_buffer(0, self.vertex_buf)
        pass_enc.draw_indexed(range(self.index_count), 0, range(1))
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    import os
    # Note: To see the difference between linear and sRGB blending, 
    # one would normally toggle surface format or view format.
    # Here we use the default provided by the framework.
    asyncio.run(run_example(SrgbBlendExample))
