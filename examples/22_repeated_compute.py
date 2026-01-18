import asyncio
import struct
import random
from typing import List

import pywgpu
from framework import Example, run_example

class RepeatedComputeExample(Example):
    TITLE = "Repeated Compute Example (Collatz)"

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            @group(0) @binding(0) var<storage, read_write> v_indices: array<u32>;

            fn collatz_iterations(n_base: u32) -> u32 {
                var n: u32 = n_base;
                var i: u32 = 0u;
                loop {
                    if (n <= 1u) {
                        break;
                    }
                    if (n % 2u == 0u) {
                        n = n / 2u;
                    } else {
                        if (n >= 1431655765u) { // Overflow 0x55555555
                            return 4294967295u; // 0xffffffff
                        }
                        n = 3u * n + 1u;
                    }
                    i = i + 1u;
                }
                return i;
            }

            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Setup
        self.count = 256
        self.storage_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            label="Storage Buffer",
            size=self.count * 4,
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_DST, pywgpu.BufferUsages.COPY_SRC]
        ))
        self.staging_buffer = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=self.count * 4,
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))

        # 3. Bind Group
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(binding=0, visibility=pywgpu.ShaderStages.compute, ty=pywgpu.BufferBindingType.storage(read_only=False))
        ]))
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[pywgpu.BindGroupEntry(binding=0, resource=self.storage_buffer.as_entire_binding())]
        ))

        # 4. Pipeline
        self.pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            module=self.shader,
            entry_point="main"
        ))

        self.iterations = 0
        self.max_iterations = 5
        self.computing = False

    def render(self, view, device, queue):
        if self.iterations < self.max_iterations and not self.computing:
            self.run_iteration(device, queue)
            self.iterations += 1

        # Background color
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.3, g=0.2, b=0.1, a=1)), store=pywgpu.StoreOp.store)
            )]
        ))
        pass_enc.end()
        queue.submit([encoder.finish()])

    def run_iteration(self, device, queue):
        self.computing = True
        numbers = [random.randint(1, 10000) for _ in range(self.count)]
        print(f"Iteration {self.iterations + 1} starting with: {numbers[:8]}...")
        
        queue.write_buffer(self.storage_buffer, 0, struct.pack(f"{self.count}I", *numbers))
        
        encoder = device.create_command_encoder()
        pass_comp = encoder.begin_compute_pass(pywgpu.ComputePassDescriptor())
        pass_comp.set_pipeline(self.pipeline)
        pass_comp.set_bind_group(0, self.bind_group)
        pass_comp.dispatch_workgroups(self.count, 1, 1)
        pass_comp.end()
        
        encoder.copy_buffer_to_buffer(self.storage_buffer, 0, self.staging_buffer, 0, self.staging_buffer.size)
        queue.submit([encoder.finish()])
        
        def callback(status):
            if status == pywgpu.BufferMapAsyncStatus.success:
                data = self.staging_buffer.get_mapped_range()
                result = struct.unpack(f"{self.count}I", data)
                print(f"Iteration {self.iterations} results: {result[:8]}...")
                self.staging_buffer.unmap()
                self.computing = False
            else:
                print(f"Iter {self.iterations} failed: {status}")
                self.computing = False
        
        self.staging_buffer.map_async(pywgpu.MapMode.read, callback=callback)

if __name__ == "__main__":
    asyncio.run(run_example(RepeatedComputeExample))
