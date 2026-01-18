import asyncio
import struct
import time
from typing import List

import pywgpu
from framework import Example, run_example

class MultiviewExample(Example):
    TITLE = "Multiview Example (VR-style Rendering)"

    def required_features(self) -> List[pywgpu.Features]:
        return [pywgpu.Features.MULTI_VIEW]

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            const triangles = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -1.0), 
                vec2<f32>(3.0, -1.0), 
                vec2<f32>(-1.0, 3.0)
            );

            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(triangles[vi], 0.0, 1.0);
            }

            @fragment
            fn fs_main(@builtin(view_index) view_index: u32) -> @location(0) vec4<f32> {
                // Return different colors based on view_index
                let v = f32(view_index);
                return vec4<f32>(v * 0.25 + 0.125, 1.0 - v * 0.25, 1.0 - 0.5 * v, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Texture Array
        self.size = 512
        self.num_layers = 4
        self.layer_mask = 0b1111 # 4 views
        
        self.tex_array = device.create_texture(pywgpu.TextureDescriptor(
            label="Multiview Texture Array",
            size=pywgpu.Extent3d(width=self.size, height=self.size, depth_or_array_layers=self.num_layers),
            format=pywgpu.TextureFormat.rgba8unorm,
            usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.texture_binding]
        ))
        
        # View for all layers (for multiview render pass)
        self.multiview_view = self.tex_array.create_view(pywgpu.TextureViewDescriptor(
            format=pywgpu.TextureFormat.rgba8unorm,
            dimension=pywgpu.TextureViewDimension.d2_array,
            base_array_layer=0,
            array_layer_count=self.num_layers
        ))
        
        # Views for individual layers (for display)
        self.layer_views = [
            self.tex_array.create_view(pywgpu.TextureViewDescriptor(
                dimension=pywgpu.TextureViewDimension.d2,
                base_array_layer=i,
                array_layer_count=1
            )) for i in range(self.num_layers)
        ]

        # 3. Pipelines
        # Multiview Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=pywgpu.TextureFormat.rgba8unorm)]),
            primitive=pywgpu.PrimitiveState(),
            multiview=self.layer_mask
        ))

        # Blit Pipeline (Simple quad to display one layer)
        blit_shader = """
            struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
            @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> Out {
                let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
                return Out(vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0), uv);
            }
            @group(0) @binding(0) var t: texture_2d<f32>;
            @group(0) @binding(1) var s: sampler;
            @fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
                return textureSample(t, s, in.uv);
            }
        """
        self.blit_mod = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=blit_shader))
        self.sampler = device.create_sampler()
        self.blit_bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.texture()),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.fragment, ty=pywgpu.BindingType.sampler())
        ]))
        self.blit_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.blit_bg_layout]),
            vertex=pywgpu.VertexState(module=self.blit_mod, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.blit_mod, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list)
        ))
        
        # We'll create bind groups on the fly for simplicity or reuse one and update
        self.blit_bgs = [
            device.create_bind_group(pywgpu.BindGroupDescriptor(
                layout=self.blit_bg_layout,
                entries=[
                    pywgpu.BindGroupEntry(binding=0, resource=v),
                    pywgpu.BindGroupEntry(binding=1, resource=self.sampler)
                ]
            )) for v in self.layer_views
        ]

        self.start_time = time.time()

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        # Pass 1: Render to all layers via Multiview
        pass_multi = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=self.multiview_view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.02, g=0.02, b=0.02, a=1)), store=pywgpu.StoreOp.store)
            )],
            multiview=self.layer_mask
        ))
        pass_multi.set_pipeline(self.pipeline)
        pass_multi.draw(vertices=range(3), instances=range(1))
        pass_multi.end()
        
        # Pass 2: Display one layer of the array texture
        # Cycle through layers every second
        current_layer = int(time.time() - self.start_time) % self.num_layers
        
        pass_blit = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0, g=0, b=0, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_blit.set_pipeline(self.blit_pipeline)
        pass_blit.set_bind_group(0, self.blit_bgs[current_layer])
        pass_blit.draw(vertices=range(3), instances=range(1))
        pass_blit.end()
        
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(MultiviewExample))
