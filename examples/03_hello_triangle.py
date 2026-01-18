"""
Hello Triangle Example

A simple example that renders a red triangle on a green background using pywgpu.
This demonstrates basic vertex buffer setup, shader creation, and render pass execution.
"""

import asyncio
import struct
import pywgpu
from framework import Example, run_example

class HelloTriangle(Example):
    TITLE = "Hello Triangle"

    async def init(self, config, adapter, device, queue):
        # Vertex data for a simple triangle
        vertices = [
            # Position (x, y), Color (r, g, b)
            -0.5, -0.5,  1.0, 0.0, 0.0,  # Red vertex
             0.5, -0.5,  0.0, 1.0, 0.0,  # Green vertex
             0.0,  0.5,  0.0, 0.0, 1.0,  # Blue vertex
        ]
        
        # Create vertex buffer and upload data
        vertex_data = struct.pack(f'{len(vertices)}f', *vertices)
        self.vertex_buffer = device.create_buffer_init(
            label="Triangle Vertex Buffer",
            contents=vertex_data,
            usage=pywgpu.BufferUsages.VERTEX
        )
        
        # Create shader module
        shader_code = """
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) color: vec3<f32>,
            };
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec3<f32>,
            };
            
            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4<f32>(input.position, 0.0, 1.0);
                output.color = input.color;
                return output;
            }
            
            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(input.color, 1.0);
            }
        """
        
        shader_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(
            label=None,
            wgsl_code=shader_code,
        ))
        
        # Create render pipeline
        pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(
            label=None,
            bind_group_layouts=[],
            immediate_size=0,
        ))
        
        self.render_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            label=None,
            layout=pipeline_layout,
            vertex=pywgpu.VertexState(
                module=shader_module,
                entry_point="vs_main",
                buffers=[
                    pywgpu.VertexBufferLayout(
                        array_stride=20, # 5 * 4
                        step_mode=pywgpu.VertexStepMode.vertex,
                        attributes=[
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x2, offset=0, shader_location=0),
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x3, offset=8, shader_location=1),
                        ],
                    )
                ],
            ),
            fragment=pywgpu.FragmentState(
                module=shader_module,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(
                    format=config.format,
                    blend=None,
                    write_mask=pywgpu.ColorWriteMask.ALL,
                )],
            ),
            primitive=pywgpu.PrimitiveState(
                topology=pywgpu.PrimitiveTopology.triangle_list,
                front_face=pywgpu.FrontFace.ccw,
                cull_mode=None,
            ),
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[
                pywgpu.RenderPassColorAttachment(
                    view=view,
                    ops=pywgpu.Operations(
                        load=pywgpu.LoadOp.clear(pywgpu.Color.GREEN),
                        store=pywgpu.StoreOp.store,
                    ),
                )
            ],
        ))
        
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(vertices=range(3), instances=range(1))
        render_pass.end()
        
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(HelloTriangle))