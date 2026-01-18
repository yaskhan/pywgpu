import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class TimestampQueriesExample(Example):
    TITLE = "Timestamp Queries Example"

    async def init(self, config, adapter, device, queue):
        # 1. Feature check
        if not pywgpu.Features.TIMESTAMP_QUERY in adapter.features:
            print("Adapter does not support TIMESTAMP_QUERY. This example will not work.")
            self.unsupported = True
            return
        
        self.unsupported = False
        self.num_queries = 4
        
        # 2. Query Set
        self.query_set = device.create_query_set(pywgpu.QuerySetDescriptor(
            label="Timestamp Query Set",
            ty=pywgpu.QueryType.timestamp,
            count=self.num_queries
        ))
        
        # 3. Buffers for resolving
        self.resolve_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            label="Resolve Buffer",
            size=self.num_queries * 8, # u64
            usage=[pywgpu.BufferUsages.QUERY_RESOLVE, pywgpu.BufferUsages.COPY_SRC]
        ))
        self.staging_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=self.num_queries * 8,
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))

        # 4. Shader for some work
        shader_code = """
            @vertex
            fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
                let x = f32(i32(vi) / 2) * 2.0 - 1.0;
                let y = f32(i32(vi) & 1) * 2.0 - 1.0;
                return vec4<f32>(x, y, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(0.2, 0.5, 0.3, 1.0);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))
        
        self.pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[]),
            vertex=pywgpu.VertexState(module=self.shader, entry_point="vs_main"),
            fragment=pywgpu.FragmentState(module=self.shader, entry_point="fs_main", targets=[pywgpu.ColorTargetState(format=config.format)]),
            primitive=pywgpu.PrimitiveState(topology=pywgpu.PrimitiveTopology.triangle_strip)
        ))

        self.done = False

    def render(self, view, device, queue):
        if self.unsupported:
            return

        encoder = device.create_command_encoder()
        
        # 1. Profiled Pass
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )],
            timestamp_writes=pywgpu.RenderPassTimestampWrites(
                query_set=self.query_set,
                beginning_of_pass_write_index=0,
                end_of_pass_write_index=1
            )
        ))
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.draw(vertices=range(4), instances=range(1))
        pass_enc.end()

        # 2. Resolve
        encoder.resolve_query_set(self.query_set, 0, self.num_queries, self.resolve_buffer, 0)
        encoder.copy_buffer_to_buffer(self.resolve_buffer, 0, self.staging_buffer, 0, self.staging_buffer.size)
        
        queue.submit([encoder.finish()])

        # 3. Read back (once per frame for demo, though usually not recommended)
        if not self.done:
            def callback(status):
                if status == pywgpu.BufferMapAsyncStatus.success:
                    data = self.staging_buffer.get_mapped_range()
                    timestamps = struct.unpack(f"{self.num_queries}Q", data)
                    delta = timestamps[1] - timestamps[0]
                    # Convert to nanoseconds (approximate, usually needs queue.get_timestamp_period())
                    print(f"Pass duration: {delta} ticks")
                    self.staging_buffer.unmap()
                self.done = False # Keep profiling if needed
            
            self.staging_buffer.map_async(pywgpu.MapMode.read, callback=callback)
            self.done = True

if __name__ == "__main__":
    asyncio.run(run_example(TimestampQueriesExample))
 village
