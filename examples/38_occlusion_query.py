import struct
import asyncio
from typing import List

import pywgpu
from framework import Example, run_example

class OcclusionQueryExample(Example):
    TITLE = "Occlusion Query Example"

    async def init(self, config, adapter, device, queue):
        # 1. Pipeline
        # We use a shader that takes position (vec3) and color (vec4)
        shader_code = """
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
            };

            @vertex
            fn vs_main(
                @location(0) pos: vec3<f32>,
                @location(1) color: vec4<f32>
            ) -> VertexOutput {
                var out: VertexOutput;
                out.position = vec4<f32>(pos, 1.0);
                out.color = color;
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return in.color;
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Depth Texture
        self.depth_texture = device.create_texture(pywgpu.TextureDescriptor(
            label="Depth Texture",
            size=[config.width, config.height, 1],
            format=pywgpu.TextureFormat.depth24plus,
            usage=pywgpu.TextureUsages.RENDER_ATTACHMENT
        ))
        self.depth_view = self.depth_texture.create_view()

        # 3. Pipeline
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(
                module=self.shader,
                entry_point="vs_main",
                buffers=[
                    pywgpu.VertexBufferLayout(
                        array_stride=28, # 3*f32 + 4*f32
                        attributes=[
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x3, offset=0, shader_location=0),
                            pywgpu.VertexAttribute(format=pywgpu.VertexFormat.float32x4, offset=12, shader_location=1)
                        ]
                    )
                ]
            ),
            fragment=pywgpu.FragmentState(
                module=self.shader,
                entry_point="fs_main",
                targets=[pywgpu.ColorTargetState(format=config.format)]
            ),
            depth_stencil=pywgpu.DepthStencilState(
                format=pywgpu.TextureFormat.depth24plus,
                depth_write_enabled=True,
                depth_compare=pywgpu.CompareFunction.less
            ),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_list)
        ))

        # 4. Data
        # Green square (front, Z=0.1)
        # Red square (back, Z=0.5, smaller, would be occluded by green)
        # We'll position them so red is mostly hidden.
        # Position (x,y,z), Color (r,g,b,a)
        front_square = [
            -0.5, -0.5, 0.1,  0.0, 1.0, 0.0, 1.0,
             0.5, -0.5, 0.1,  0.0, 1.0, 0.0, 1.0,
            -0.5,  0.5, 0.1,  0.0, 1.0, 0.0, 1.0,
            -0.5,  0.5, 0.1,  0.0, 1.0, 0.0, 1.0,
             0.5, -0.5, 0.1,  0.0, 1.0, 0.0, 1.0,
             0.5,  0.5, 0.1,  0.0, 1.0, 0.0, 1.0,
        ]
        back_square = [
            -0.3, -0.3, 0.5,  1.0, 0.0, 0.0, 1.0,
             0.3, -0.3, 0.5,  1.0, 0.0, 0.0, 1.0,
            -0.3,  0.3, 0.5,  1.0, 0.0, 0.0, 1.0,
            -0.3,  0.3, 0.5,  1.0, 0.0, 0.0, 1.0,
             0.3, -0.3, 0.5,  1.0, 0.0, 0.0, 1.0,
             0.3,  0.3, 0.5,  1.0, 0.0, 0.0, 1.0,
        ]
        
        self.vbuf_front = device.create_buffer_init(
            label="Front Buffer",
            contents=struct.pack(f"{len(front_square)}f", *front_square),
            usage=pywgpu.BufferUsages.VERTEX
        )
        self.vbuf_back = device.create_buffer_init(
            label="Back Buffer",
            contents=struct.pack(f"{len(back_square)}f", *back_square),
            usage=pywgpu.BufferUsages.VERTEX
        )

        # 5. Queries
        self.query_set = device.create_query_set(pywgpu.QuerySetDescriptor(
            type=pywgpu.QueryType.occlusion,
            count=2
        ))
        
        self.query_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            label="Query Buffer",
            size=16, # 2 * u64
            usage=[pywgpu.BufferUsages.QUERY_RESOLVE, pywgpu.BufferUsages.COPY_SRC]
        ))
        
        self.staging_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=16,
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))
        
        self.frame_count = 0

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.1, b=0.1, a=1.0)),
                    store=pywgpu.StoreOp.store
                )
            )],
            depth_stencil_attachment=pywgpu.RenderPassDepthStencilAttachment(
                view=self.depth_view,
                depth_ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(1.0),
                    store=pywgpu.StoreOp.store
                )
            ),
            occlusion_query_set=self.query_set
        ))
        
        pass_enc.set_pipeline(self.pipeline)
        
        # 1. Draw front square (green) - should pass all samples
        pass_enc.begin_occlusion_query(0)
        pass_enc.set_vertex_buffer(0, self.vbuf_front)
        pass_enc.draw(6)
        pass_enc.end_occlusion_query()
        
        # 2. Draw back square (red) - should be occluded by green
        pass_enc.begin_occlusion_query(1)
        pass_enc.set_vertex_buffer(0, self.vbuf_back)
        pass_enc.draw(6)
        pass_enc.end_occlusion_query()
        
        pass_enc.end()
        
        # Resolve query results
        encoder.resolve_query_set(self.query_set, 0, 2, self.query_buffer, 0)
        encoder.copy_buffer_to_buffer(self.query_buffer, 0, self.staging_buffer, 0, 16)
        
        queue.submit([encoder.finish()])
        
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            async def read_query():
                await self.staging_buffer.map_async(pywgpu.MapMode.read)
                data = self.staging_buffer.get_mapped_range()
                # Query results are 64-bit unsigned integers
                front_passed, back_passed = struct.unpack("QQ", data)
                print(f"Frame {self.frame_count}:")
                print(f"  Front square (green) samples passed: {front_passed}")
                print(f"  Back square (red, occluded) samples passed: {back_passed}")
                self.staging_buffer.unmap()
            
            asyncio.create_task(read_query())

if __name__ == "__main__":
    asyncio.run(run_example(OcclusionQueryExample))
