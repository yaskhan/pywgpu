import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

TEXTURE_DIMS = (512, 512)

class StorageTextureExample(Example):
    TITLE = "Storage Texture Example (Mandelbrot)"

    async def init(self, config, adapter, device, queue):
        # Compute Shader
        shader_code = """
            @group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(1, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let dims = textureDimensions(output_tex);
                let x = f32(global_id.x);
                let y = f32(global_id.y);
                
                if (x >= f32(dims.x) || y >= f32(dims.y)) {
                    return;
                }

                let c = vec2<f32>((x / f32(dims.x)) * 3.5 - 2.5, (y / f32(dims.y)) * 2.0 - 1.0);
                var z = vec2<f32>(0.0, 0.0);
                var iteration = 0u;
                let max_iteration = 100u;

                while (z.x * z.x + z.y * z.y <= 4.0 && iteration < max_iteration) {
                    let next_z = vec2<f32>(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y);
                    z = next_z;
                    iteration = iteration + 1u;
                }

                let color = f32(iteration) / f32(max_iteration);
                textureStore(output_tex, vec2<i32>(global_id.xy), vec4<f32>(color, color, color, 1.0));
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Storage texture
        self.storage_texture = device.create_texture(pywgpu.TextureDescriptor(
            label="Storage Texture",
            size=(TEXTURE_DIMS[0], TEXTURE_DIMS[1], 1),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=pywgpu.TextureFormat.rgba8unorm,
            usage=[pywgpu.TextureUsages.STORAGE_BINDING, pywgpu.TextureUsages.TEXTURE_BINDING, pywgpu.TextureUsages.COPY_DST]
        ))
        self.storage_view = self.storage_texture.create_view()

        # Compute bind group and pipeline
        self.compute_bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.StorageTextureBindingType(access=pywgpu.StorageTextureAccess.write_only, format=pywgpu.TextureFormat.rgba8unorm, view_dimension=pywgpu.TextureViewDimension.d2))
        ]))
        self.compute_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.compute_bg_layout,
            entries=[pywgpu.BindGroupEntry(binding=0, resource=self.storage_view)]
        ))

        self.compute_pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.compute_bg_layout]),
            module=self.shader,
            entry_point="main"
        ))

        # Display Pipeline (to show on screen)
        display_shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                let tc = vec2<f32>(f32(vi & 1u), 0.5 * f32(vi & 2u));
                let pos = vec4<f32>(tc * 2.0 - 1.0, 0.0, 1.0);
                return VertexOutput(pos, vec2<f32>(tc.x, 1.0 - tc.y));
            }

            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var sam: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(tex, sam, in.tex_coords);
            }
        """
        self.display_shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=display_shader_code))
        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor())
        
        self.display_bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.TextureBindingType.float()),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.SamplerBindingType.filtering)
        ]))
        self.display_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.display_bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.storage_view),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))

        self.display_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.display_bg_layout]),
            vertex=pywgpu.VertexState(module=self.display_shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.display_shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        # 1. Compute
        pass_comp = encoder.begin_compute_pass(pywgpu.ComputePassDescriptor())
        pass_comp.set_pipeline(self.compute_pipeline)
        pass_comp.set_bind_group(0, self.compute_bg)
        pass_comp.dispatch_workgroups(TEXTURE_DIMS[0], TEXTURE_DIMS[1], 1)
        pass_comp.end()

        # 2. Display
        pass_disp = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_disp.set_pipeline(self.display_pipeline)
        pass_disp.set_bind_group(0, self.display_bg)
        pass_disp.draw(vertices=range(4), instances=range(1))
        pass_disp.end()

        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(StorageTextureExample))
