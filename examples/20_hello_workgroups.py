import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class HelloWorkgroupsExample(Example):
    TITLE = "Hello Workgroups Example"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            @group(0) @binding(0) var<storage, read_write> a: array<i32>;
            @group(0) @binding(1) var<storage, read_write> b: array<i32>;

            @compute @workgroup_size(2, 1, 1)
            fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
                if (lid.x == 0u) {
                    a[wid.x] += 1;
                } else if (lid.x == 1u) {
                    b[wid.x] += 1;
                }
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Data
        self.count = 100
        data_a = [i for i in range(self.count)]
        data_b = [i * 2 for i in range(self.count)]
        
        # 3. Buffers
        self.buffer_a = device.create_buffer_init(
            label="Buffer A",
            contents=struct.pack(f"{self.count}i", *data_a),
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC]
        )
        self.buffer_b = device.create_buffer_init(
            label="Buffer B",
            contents=struct.pack(f"{self.count}i", *data_b),
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC]
        )
        self.staging_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=self.count * 4,
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))

        # 4. Bind Group
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.storage(read_only=False)),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.storage(read_only=False))
        ]))
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.buffer_a.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.buffer_b.as_entire_binding())
            ]
        ))

        # 5. Pipeline
        self.pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            module=self.shader,
            entry_point="main"
        ))
        
        self.done = False

    def render(self, view, device, queue):
        if not self.done:
            encoder = device.create_command_encoder()
            pass_comp = encoder.begin_compute_pass(pywgpu.ComputePassDescriptor())
            pass_comp.set_pipeline(self.pipeline)
            pass_comp.set_bind_group(0, self.bind_group)
            pass_comp.dispatch_workgroups(self.count, 1, 1)
            pass_comp.end()
            
            queue.submit([encoder.finish()])
            
            # Read back both buffers
            self.read_back(device, queue, self.buffer_a, "A")
            self.read_back(device, queue, self.buffer_b, "B")
            self.done = True

        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.2, g=0.1, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.end()
        queue.submit([encoder.finish()])

    def read_back(self, device, queue, buffer, label):
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(buffer, 0, self.staging_buf, 0, self.staging_buf.size)
        queue.submit([encoder.finish()])
        
        def callback(status):
            if status == pywgpu.BufferMapAsyncStatus.success:
                data = self.staging_buf.get_mapped_range()
                result = struct.unpack(f"{self.count}i", data)
                print(f"Output {label}: {result[:10]} ...")
                self.staging_buf.unmap()
        
        self.staging_buf.map_async(pywgpu.MapMode.read, callback=callback)

if __name__ == "__main__":
    asyncio.run(run_example(HelloWorkgroupsExample))
