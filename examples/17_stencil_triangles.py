import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class StencilTrianglesExample(Example):
    TITLE = "Stencil Triangles Example"
    STENCIL_FORMAT = pywgpu.TextureFormat.stencil8

    async def init(self, config, adapter, device, queue):
        # Shader
        shader_code = """
            @vertex
            fn vs_main(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
                return pos;
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # Vertex buffers
        mask_vertices = struct.pack("12f", -0.5, 0.0, 0, 1, 0.0, -1.0, 0, 1, 0.5, 0.0, 0, 1)
        outer_vertices = struct.pack("12f", -1.0, -1.0, 0, 1, 1.0, -1.0, 0, 1, 0.0, 1.0, 0, 1)
        
        self.mask_vertex_buf = device.create_buffer_init(label="Mask Vertices", contents=mask_vertices, usage=pywgpu.BufferUsages.VERTEX)
        self.outer_vertex_buf = device.create_buffer_init(label="Outer Vertices", contents=outer_vertices, usage=pywgpu.BufferUsages.VERTEX)

        # Pipelines
        v_layout = pywgpu.VertexBufferLayout(array_stride=16, attributes=[pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x4, offset=0, shader_location=0)])
        
        # Mask pipeline: Writes to stencil, no color
        self.mask_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main", buffers=[v_layout]),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format, write_mask=0)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list),
            depth_stencil=pywgpu.DepthStencilState(
                format=self.STENCIL_FORMAT,
                depth_write_enabled=False,
                depth_compare=pywgpu.CompareFunction.always,
                stencil=pywgpu.StencilState(
                    front=pywgpu.StencilFaceState(compare=pywgpu.CompareFunction.always, pass_op=pywgpu.StencilOperation.replace)
                )
            )
        ))

        # Outer pipeline: Render where stencil matches
        self.outer_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main", buffers=[v_layout]),
            fragment=pywgpu.FragmentState(
                module=self.shader, entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list),
            depth_stencil=pywgpu.DepthStencilState(
                format=self.STENCIL_FORMAT,
                depth_write_enabled=False,
                depth_compare=pywgpu.CompareFunction.always,
                stencil=pywgpu.StencilState(
                    front=pywgpu.StencilFaceState(compare=pywgpu.CompareFunction.greater)
                )
            )
        ))

        self.resize(config, device, queue)

    def resize(self, config, device, queue):
        self.stencil_texture = device.create_texture(pywgpu.TextureDescriptor(
            size=(config.width, config.height, 1),
            dimension=pywgpu.TextureDimension.d2,
            format=self.STENCIL_FORMAT,
            usage=pywgpu.TextureUsages.RENDER_ATTACHMENT
        ))
        self.stencil_view = self.stencil_texture.create_view()

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )],
            depth_stencil_attachment=pywgpu.RenderPassDepthStencilAttachment(
                view=self.stencil_view,
                stencil_ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(0), store=pywgpu.StoreOp.store)
            )
        ))
        
        pass_enc.set_stencil_reference(1)
        
        # 1. Draw mask
        pass_enc.set_pipeline(self.mask_pipeline)
        pass_enc.set_vertex_buffer(0, self.mask_vertex_buf)
        pass_enc.draw(vertices=range(3), instances=range(1))
        
        # 2. Draw outer (will only show where stencil is > 0)
        # Actually in rust example it uses Greater, and reference is 1.
        # So it will draw where it's 1? No, Greater means it will draw where 1 > stencil?
        # Rust: compare: wgpu::CompareFunction::Greater. Ref: 1.
        # So Ref (1) > StencilValue (0).
        # Where mask was drawn, StencilValue = 1. Ref (1) > StencilValue (1) is False.
        # So it draws OUTSIDE the mask.
        pass_enc.set_pipeline(self.outer_pipeline)
        pass_enc.set_vertex_buffer(0, self.outer_vertex_buf)
        pass_enc.draw(vertices=range(3), instances=range(1))
        
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(StencilTrianglesExample))
