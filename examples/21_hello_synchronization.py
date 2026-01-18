import asyncio
import struct
import math
from typing import List

import pywgpu
from framework import Example, run_example

class HelloSynchronizationExample(Example):
    TITLE = "Hello Synchronization Example"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;
            var<workgroup> count: atomic<u32>;

            @compute @workgroup_size(16)
            fn patient_main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
                atomicAdd(&count, 1u);
                workgroupBarrier();
                if (lid.x == 0u) {
                    output[wid.x] = atomicLoad(&count);
                }
            }

            @compute @workgroup_size(16)
            fn hasty_main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
                atomicAdd(&count, 1u);
                if (lid.x == 0u) {
                    output[wid.x] = atomicLoad(&count);
                }
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Data
        self.count = 128
        
        # 3. Buffers
        self.output_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Output Buffer",
            size=self.count * 4,
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC]
        ))
        self.staging_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=self.count * 4,
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))

        # 4. Bind Group
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.storage(read_only=False))
        ]))
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[pywgpu.BindGroupEntry(binding=0, resource=self.output_buf.as_entire_binding())]
        ))

        # 5. Pipelines
        self.patient_pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            module=self.shader,
            entry_point="patient_main"
        ))
        self.hasty_pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            module=self.shader,
            entry_point="hasty_main"
        ))
        
        self.done = False

    def render(self, view, device, queue):
        if not self.done:
            # 1. Patient execution
            self.run_pipeline(device, queue, self.patient_pipeline, "Patient")
            # 2. Hasty execution
            self.run_pipeline(device, queue, self.hasty_pipeline, "Hasty")
            self.done = True

        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.1, g=0.3, b=0.2, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.end()
        queue.submit([encoder.finish()])

    def run_pipeline(self, device, queue, pipeline, label):
        encoder = device.create_command_encoder()
        pass_comp = encoder.begin_compute_pass(pywgpu.ComputePassDescriptor())
        pass_comp.set_pipeline(pipeline)
        pass_comp.set_bind_group(0, self.bind_group)
        pass_comp.dispatch_workgroups(self.count, 1, 1)
        pass_comp.end()
        
        encoder.copy_buffer_to_buffer(self.output_buf, 0, self.staging_buf, 0, self.staging_buf.size)
        queue.submit([encoder.finish()])
        
        def callback(status):
            if status == pywgpu.BufferMapAsyncStatus.success:
                data = self.staging_buf.get_mapped_range()
                result = struct.unpack(f"{self.count}I", data)
                # Check if all were 16 (since workgroup size is 16)
                all_16 = all(v == 16 for v in result)
                print(f"{label} results: {result[:8]}... All 16: {all_16}")
                self.staging_buf.unmap()
        
        self.staging_buf.map_async(pywgpu.MapMode.read, callback=callback)

if __name__ == "__main__":
    asyncio.run(run_example(HelloSynchronizationExample))
