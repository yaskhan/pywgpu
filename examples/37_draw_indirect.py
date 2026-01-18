import struct
import asyncio
import math
from typing import List

import pywgpu
from framework import Example, run_example

class DrawIndirectExample(Example):
    TITLE = "Draw Indirect Example"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        # We use a shader that takes vertex positions and an instance offset.
        shader_code = """
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) instance_pos: vec2<f32>,
            };

            @vertex
            fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
                return vec4<f32>(input.position * 0.1 + input.instance_pos, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(0.2, 0.6, 1.0, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(
                module=self.shader,
                entry_point="vs_main",
                buffers=[
                    # Vertex positions (slot 0)
                    pywgpu.VertexBufferLayout(
                        array_stride=8,
                        step_mode=pywgpu.VertexStepMode.vertex,
                        attributes=[pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=0, shader_location=0)]
                    ),
                    # Instance positions (slot 1)
                    pywgpu.VertexBufferLayout(
                        array_stride=8,
                        step_mode=pywgpu.VertexStepMode.instance,
                        attributes=[pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=0, shader_location=1)]
                    )
                ]
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list)
        ))

        # 3. Vertex Data
        # A simple triangle
        vertices = struct.pack("6f", 
            0.0, 0.5,
            -0.5, -0.5,
            0.5, -0.5
        )
        self.vbuf = device.create_buffer_init(
            label="Vertex Buffer",
            contents=vertices,
            usage=pywgpu.BufferUsages.VERTEX
        )

        # 4. Instance Data
        # A 10x10 grid of positions
        instance_data = []
        for y in range(10):
            for x in range(10):
                px = (x / 9.0) * 1.6 - 0.8
                py = (y / 9.0) * 1.6 - 0.8
                instance_data.extend([px, py])
        
        instance_bytes = struct.pack(f"{len(instance_data)}f", *instance_data)
        self.ibuf = device.create_buffer_init(
            label="Instance Buffer",
            contents=instance_bytes,
            usage=pywgpu.BufferUsages.VERTEX
        )

        # 5. Indirect Buffer
        # DrawIndirectArgs: vertex_count, instance_count, first_vertex, first_instance
        # We want to draw 3 vertices and 100 instances.
        indirect_args = struct.pack("IIII", 3, 100, 0, 0)
        self.indirect_buf = device.create_buffer_init(
            label="Indirect Buffer",
            contents=indirect_args,
            usage=pywgpu.BufferUsages.INDIRECT
        )

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.05, g=0.05, b=0.05, a=1.0)),
                    store=pywgpu.StoreOp.store
                )
            )]
        ))
        
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_vertex_buffer(0, self.vbuf)
        pass_enc.set_vertex_buffer(1, self.ibuf)
        
        # Execute draw from indirect buffer
        pass_enc.draw_indirect(self.indirect_buf, 0)
        
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(DrawIndirectExample))
