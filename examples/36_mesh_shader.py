import asyncio
import time
import math
import numpy as np
from typing import List, Optional, Any

import pywgpu
from framework import Example, run_example

class MeshShaderExample(Example):
    TITLE = "Mesh Shader (GPU-driven Geometry)"

    def required_features(self) -> List[pywgpu.Features]:
        # Note: Some backends might need SPIRV_SHADER_PASSTHROUGH if using certain compilers
        return [pywgpu.Features.EXPERIMENTAL_MESH_SHADER]

    async def init(self, config, adapter, device, queue):
        # Mesh shader code (WGSL)
        # Note: This is an experimental feature, syntax might vary by backend
        shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec3<f32>,
            };

            // Mesh shader
            @mesh
            @workgroup_size(1)
            fn ms_main(
                @builtin(workgroup_id) workgroup_id: vec3<u32>,
            ) -> mesh_output<VertexOutput, 3, 1, triangle> {
                var out: mesh_output<VertexOutput, 3, 1, triangle>;

                set_primitive_count(&out, 1u);

                set_vertex(&out, 0u, VertexOutput(vec4<f32>(-0.5, -0.5, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0)));
                set_vertex(&out, 1u, VertexOutput(vec4<f32>( 0.5, -0.5, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0)));
                set_vertex(&out, 2u, VertexOutput(vec4<f32>( 0.0,  0.5, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0)));

                set_index(&out, 0u, vec3<u32>(0u, 1u, 2u));

                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(in.color, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Pipeline layout
        layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(bind_group_layouts=[]))

        # Mesh pipeline
        self.pipeline = device.create_mesh_pipeline(pywgpu.MeshPipelineDescriptor(
            layout=layout,
            mesh=pywgpu.MeshState(
                module=self.shader,
                entry_point="ms_main"
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(topology="triangle-list"),
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        rpass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color(0.1, 0.2, 0.3, 1.0)),
                    store=pywgpu.StoreOp.store
                )
            )]
        ))
        rpass.set_pipeline(self.pipeline)
        rpass.draw_mesh_tasks(1, 1, 1)
        rpass.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(MeshShaderExample))
