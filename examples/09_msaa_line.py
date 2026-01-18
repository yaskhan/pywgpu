import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class MsaaLineExample(Example):
    TITLE = "MSAA Line Example"

    async def init(self, config, adapter, device, queue):
        self.sample_count = 4 # Default to 4x MSAA
        self.vertex_data = []

        # Generate vertex data for a star-like pattern
        max_lines = 50
        for i in range(max_lines):
            percent = i / max_lines
            angle = percent * 2.0 * math.pi
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            
            # Center point
            self.vertex_data.extend([0.0, 0.0]) # pos
            self.vertex_data.extend([1.0, -sin_a, cos_a, 1.0]) # color
            
            # Outer point
            self.vertex_data.extend([cos_a, sin_a]) # pos
            self.vertex_data.extend([sin_a, -cos_a, 1.0, 1.0]) # color

        # Pack vertex data: f32x2 (pos) + f32x4 (color) = 24 bytes per vertex
        packed_vertex_data = struct.pack(f"{len(self.vertex_data)}f", *self.vertex_data)
        self.vertex_buffer = device.create_buffer_init(
            label="Vertex Buffer",
            contents=packed_vertex_data,
            usage=pywgpu.BufferUsages.VERTEX
        )
        self.vertex_count = len(self.vertex_data) // 6

        # Shader
        shader_code = """
            struct VertexOutput {
                @location(0) color: vec4<f32>,
                @builtin(position) position: vec4<f32>,
            };

            @vertex
            fn vs_main(
                @location(0) position: vec2<f32>,
                @location(1) color: vec4<f32>,
            ) -> VertexOutput {
                var result: VertexOutput;
                result.position = vec4<f32>(position, 0.0, 1.0);
                result.color = color;
                return result;
            }

            @fragment
            fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
                return vertex.color;
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))
        
        self.pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(bind_group_layouts=[]))
        
        # Create pipeline and MSAA buffer
        self._recreate_resources(device, config)

    def _recreate_resources(self, device, config):
        # Pipeline with MSAA
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=self.pipeline_layout,
            vertex=pywgpu.VertexState(
                module=self.shader,
                entry_point="vs_main",
                buffers=[pywgpu.VertexBufferLayout(
                    array_stride=24,
                    step_mode=pywgpu.VertexStepMode.vertex,
                    attributes=[
                        pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=0, shader_location=0),
                        pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x4, offset=8, shader_location=1)
                    ]
                )]
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.line_list),
            multisample=pywgpu.MultisampleState(count=self.sample_count)
        ))

        # MSAA Framebuffer
        if self.sample_count > 1:
            self.msaa_texture = device.create_texture(pywgpu.TextureDescriptor(
                label="MSAA Texture",
                size=(config.width, config.height, 1),
                mip_level_count=1,
                sample_count=self.sample_count,
                dimension=pywgpu.TextureDimension.d2,
                format=config.format,
                usage=pywgpu.TextureUsages.RENDER_ATTACHMENT
            ))
            self.msaa_view = self.msaa_texture.create_view()
        else:
            self.msaa_view = None

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        color_attachment = None
        if self.sample_count > 1:
            color_attachment = pywgpu.RenderPassColorAttachment(
                view=self.msaa_view,
                resolve_target=view,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)),
                    store=pywgpu.StoreOp.discard # Don't need to store the MSAA data after resolve
                )
            )
        else:
            color_attachment = pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)),
                    store=pywgpu.StoreOp.store
                )
            )

        render_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[color_attachment]
        ))
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(vertices=range(self.vertex_count), instances=range(1))
        render_pass.end()
        
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(MsaaLineExample))
