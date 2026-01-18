import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

TEXTURE_DIMS = (512, 512)

class RenderToTextureExample(Example):
    TITLE = "Render to Texture Example"

    async def init(self, config, adapter, device, queue):
        # Shader
        shader_code = """
            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
                let x = f32(i32(vertex_index) / 2) * 2.0 - 1.0;
                let y = f32(i32(vertex_index) & 1) * 2.0 - 1.0;
                return vec4<f32>(x, y, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Render target texture
        self.render_target = device.create_texture(pywgpu.TextureDescriptor(
            label="Render Target",
            size=(TEXTURE_DIMS[0], TEXTURE_DIMS[1], 1),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=pywgpu.TextureFormat.rgba8unorm,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.COPY_SRC, pywgpu.TextureUsages.TEXTURE_BINDING]
        ))
        self.target_view = self.render_target.create_view()

        # Pipeline for rendering to texture
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=pywgpu.TextureFormat.rgba8unorm)]),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))

        # Pipeline for displaying the texture on screen
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
                pywgpu.BindGroupEntry(binding=0, resource=self.target_view),
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
        
        # 1. Render to offscreen texture
        pass_off = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=self.target_view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=1, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_off.set_pipeline(self.pipeline)
        pass_off.draw(vertices=range(4), instances=range(1))
        pass_off.end()

        # 2. Display texture on screen
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
    asyncio.run(run_example(RenderToTextureExample))
