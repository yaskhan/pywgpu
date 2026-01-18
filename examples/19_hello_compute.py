import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class HelloComputeExample(Example):
    TITLE = "Hello Compute Example"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                let array_length = arrayLength(&input);
                if (index >= array_length) {
                    return;
                }
                output[index] = input[index] * 2.0;
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Data
        self.input_data = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 100.0]
        input_bytes = struct.pack(f"{len(self.input_data)}f", *self.input_data)
        
        # 3. Buffers
        self.input_buf = device.create_buffer_init(
            label="Input Buffer",
            contents=input_bytes,
            usage=pywgpu.BufferUsages.STORAGE
        )
        self.output_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Output Buffer",
            size=len(input_bytes),
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC]
        ))
        self.staging_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=len(input_bytes),
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))

        # 4. Bind Group
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.storage(read_only=True)),
            pywgpu.BindGroupLayoutEntry(binding=1, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.storage(read_only=False))
        ]))
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=self.input_buf.as_entire_binding()),
                pywgpu.BindGroupEntry(binding=1, resource=self.output_buf.as_entire_binding())
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
            
            # 1. Run Compute
            pass_comp = encoder.begin_compute_pass(pywgpu.ComputePassDescriptor())
            pass_comp.set_pipeline(self.pipeline)
            pass_comp.set_bind_group(0, self.bind_group)
            workgroup_count = math.ceil(len(self.input_data) / 64)
            pass_comp.dispatch_workgroups(workgroup_count, 1, 1)
            pass_comp.end()

            # 2. Copy to staging
            encoder.copy_buffer_to_buffer(self.output_buf, 0, self.staging_buf, 0, self.staging_buf.size)
            
            queue.submit([encoder.finish()])
            
            # 3. Read back
            def callback(status):
                if status == pywgpu.BufferMapAsyncStatus.success:
                    data = self.staging_buf.get_mapped_range()
                    result = struct.unpack(f"{len(self.input_data)}f", data)
                    print(f"Input:  {self.input_data}")
                    print(f"Output: {result}")
                    self.staging_buf.unmap()
                else:
                    print(f"Failed to map buffer: {status}")
            
            self.staging_buf.map_async(pywgpu.MapMode.read, callback=callback)
            self.done = True

        # Clear screen to something nice
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.2, b=0.3, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    asyncio.run(run_example(HelloComputeExample))
