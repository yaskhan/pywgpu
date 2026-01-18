import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class MultipleRenderTargetsExample(Example):
    TITLE = "Multiple Render Targets Example"

    async def init(self, config, adapter, device, queue):
        self.width, self.height = config.width, config.height
        
        # Shader
        shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                let x = f32(i32(vi) / 2) * 2.0 - 1.0;
                let y = f32(i32(vi) & 1) * 2.0 - 1.0;
                return VertexOutput(vec4<f32>(x, y, 0.0, 1.0), vec2<f32>(x * 0.5 + 0.5, y * 0.5 + 0.5));
            }

            struct FragmentOutput {
                @location(0) target0: vec4<f32>,
                @location(1) target1: vec4<f32>,
            };

            @fragment
            fn fs_main(in: VertexOutput) -> FragmentOutput {
                var out: FragmentOutput;
                out.target0 = vec4<f32>(in.tex_coords.x, 0.0, 0.0, 1.0); // Red channel
                out.target1 = vec4<f32>(0.0, in.tex_coords.y, 0.0, 1.0); // Green channel
                return out;
            }

            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var sam: sampler;

            @fragment
            fn fs_display(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(tex, sam, in.tex_coords);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Render targets
        self.target_texture0 = device.create_texture(pywgpu.TextureDescriptor(
            label="Target 0",
            size=(self.width, self.height, 1),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=config.format,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.TEXTURE_BINDING]
        ))
        self.target_texture1 = device.create_texture(pywgpu.TextureDescriptor(
            label="Target 1",
            size=(self.width, self.height, 1),
            mip_level_count=1,
            sample_count=1,
            dimension=pywgpu.TextureDimension.d2,
            format=config.format,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.TEXTURE_BINDING]
        ))
        self.target_view0 = self.target_texture0.create_view()
        self.target_view1 = self.target_texture1.create_view()

        # Render pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_main",
                targets=[
                    pywgpu.ColorTargetState(format=config.format),
                    pywgpu.ColorTargetState(format=config.format)
                ]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))

        # Display pipeline
        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor())
        self.display_bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.TextureBindingType.float()),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.SamplerBindingType.filtering)
        ]))
        
        self.display_bg0 = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.display_bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.target_view0),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))
        self.display_bg1 = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.display_bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.target_view1),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))

        self.display_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.display_bg_layout]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_display",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))

    def resize(self, config, device, queue):
        self.width, self.height = config.width, config.height
        # Recreate targets
        self.target_texture0 = device.create_texture(pywgpu.TextureDescriptor(
            size=(self.width, self.height, 1),
            dimension=pywgpu.TextureDimension.d2,
            format=config.format,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.TEXTURE_BINDING]
        ))
        self.target_texture1 = device.create_texture(pywgpu.TextureDescriptor(
            size=(self.width, self.height, 1),
            dimension=pywgpu.TextureDimension.d2,
            format=config.format,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.TEXTURE_BINDING]
        ))
        self.target_view0 = self.target_texture0.create_view()
        self.target_view1 = self.target_texture1.create_view()
        
        self.display_bg0 = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.display_bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.target_view0),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))
        self.display_bg1 = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.display_bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.target_view1),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        # 1. Render to multiple targets
        pass_mt = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[
                pywgpu.RenderPassColorAttachment(
                    view=self.target_view0,
                    ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
                ),
                pywgpu.RenderPassColorAttachment(
                    view=self.target_view1,
                    ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
                )
            ]
        ))
        pass_mt.set_pipeline(self.pipeline)
        pass_mt.draw(vertices=range(4), instances=range(1))
        pass_mt.end()

        # 2. Display results (split screen)
        pass_disp = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_disp.set_pipeline(self.display_pipeline)
        
        # Left side
        pass_disp.set_viewport(0, 0, self.width / 2, self.height, 0, 1)
        pass_disp.set_bind_group(0, self.display_bg0)
        pass_disp.draw(vertices=range(4), instances=range(1))
        
        # Right side
        pass_disp.set_viewport(self.width / 2, 0, self.width / 2, self.height, 0, 1)
        pass_disp.set_bind_group(0, self.display_bg1)
        pass_disp.draw(vertices=range(4), instances=range(1))
        
        pass_disp.end()

        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(MultipleRenderTargetsExample))
