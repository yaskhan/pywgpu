import asyncio
import struct
from typing import List

import pywgpu
from framework import Example, run_example

class ConservativeRasterExample(Example):
    TITLE = "Conservative Rasterization Example"

    # We need specific features
    def required_features(self) -> List[pywgpu.Features]:
        return [pywgpu.Features.CONSERVATIVE_RASTERIZATION]

    async def init(self, config, adapter, device, queue):
        # 1. Shaders
        triangle_wgsl = """
            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
                let i = i32(vi % 3u);
                let x = f32(i - 1) * 0.75;
                let y = f32((i & 1) * 2 - 1) * 0.75 + x * 0.2 + 0.1;
                return vec4<f32>(x, y, 0.0, 1.0);
            }

            @fragment fn fs_red() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
            @fragment fn fs_blue() -> @location(0) vec4<f32> { return vec4<f32>(0.13, 0.31, 0.85, 1.0); }
        """
        upscale_wgsl = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            };

            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                let x = f32(i32(vi & 1u) << 2u) - 1.0;
                let y = f32(i32(vi & 2u) << 1u) - 1.0;
                var out: VertexOutput;
                out.position = vec4<f32>(x, -y, 0.0, 1.0);
                out.tex_coords = vec2<f32>(x + 1.0, y + 1.0) * 0.5;
                return out;
            }

            @group(0) @binding(0) var r_color: texture_2d<f32>;
            @group(0) @binding(1) var r_sampler: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(r_color, r_sampler, in.tex_coords);
            }
        """
        self.mod_tri = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=triangle_wgsl))
        self.mod_upscale = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=upscale_wgsl))

        # 2. Low-res target (to make conservative rasterization obvious)
        self.format = pywgpu.TextureFormat.rgba8unorm
        self.low_res_size = (config.width // 16, config.height // 16)
        self.create_resources(device)

        # 3. Pipelines
        self.pipeline_cons = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.mod_tri, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.mod_tri, entry_point="fs_red", targets=[pywgpu.ColorTargetState(format=self.format)]),
            primitive=pywgpu.PrimitiveState(conservative=True)
        ))
        
        self.pipeline_reg = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.mod_tri, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.mod_tri, entry_point="fs_blue", targets=[pywgpu.ColorTargetState(format=self.format)]),
            primitive=pywgpu.PrimitiveState(conservative=False)
        ))

        self.pipeline_upscale = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.upscale_bg_layout]),
            vertex=pywgpu.VertexState(module=self.mod_upscale, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.mod_upscale, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState()
        ))

    def create_resources(self, device):
        self.low_res_tex = device.create_texture(pywgpu.TextureDescriptor(
            label="Low Res Target",
            size=pywgpu.Extent3d(width=max(1, self.low_res_size[0]), height=max(1, self.low_res_size[1]), depth_or_array_layers=1),
            format=self.format,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.texture_binding]
        ))
        self.low_res_view = self.low_res_tex.create_view()
        
        self.sampler = device.create_sampler(pywgpu.SamplerDescriptor(
            mag_filter=pywgpu.FilterMode.nearest,
            min_filter=pywgpu.FilterMode.nearest
        ))
        
        self.upscale_bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.texture(sample_type=pywgpu.TextureSampleType.float(filterable=False))),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.sampler(pywgpu.SamplerBindingType.non_filtering))
        ]))
        
        self.upscale_bg = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.upscale_bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.low_res_view),
                pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
            ]
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        # Pass 1: Draw to low-res
        pass_low = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=self.low_res_view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        # Draw conservative (red)
        pass_low.set_pipeline(self.pipeline_cons)
        pass_low.draw(vertices=range(3), instances=range(1))
        # Draw regular (blue)
        pass_low.set_pipeline(self.pipeline_reg)
        pass_low.draw(vertices=range(3), instances=range(1))
        pass_low.end()
        
        # Pass 2: Upscale to screen
        pass_high = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_high.set_pipeline(self.pipeline_upscale)
        pass_high.set_bind_group(0, self.upscale_bg)
        pass_high.draw(vertices=range(3), instances=range(1))
        pass_high.end()
        
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(ConservativeRasterExample))
 village
