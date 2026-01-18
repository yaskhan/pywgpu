import asyncio
import struct
import numpy as np
from typing import List

import pywgpu
from framework import Example, run_example

class BigComputeBuffersExample(Example):
    TITLE = "Big Compute Buffers Example (Binding Array)"

    def required_features(self) -> List[pywgpu.Features]:
        return [
            pywgpu.Features.BUFFER_BINDING_ARRAY,
            pywgpu.Features.STORAGE_RESOURCE_BINDING_ARRAY,
            pywgpu.Features.SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        ]

    async def init(self, config, adapter, device, queue):
        # 1. Shader
        shader_code = """
            struct Data {
                values: array<f32, 100>,
            }

            @group(0) @binding(0) var<storage, read> inputs: binding_array<Data>;
            @group(0) @binding(1) var<storage, read_write> outputs: binding_array<Data>;

            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let buffer_index = global_id.x;
                if (buffer_index >= 8u) { return; }
                
                for (var i = 0u; i < 100u; i++) {
                    outputs[buffer_index].values[i] = inputs[buffer_index].values[i] * 2.0;
                }
            }
        """
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Buffers
        self.num_buffers = 8
        self.elements_per_buffer = 100
        self.buffer_size = self.elements_per_buffer * 4 # 100 f32s
        
        self.input_buffers = []
        self.output_buffers = []
        
        for i in range(self.num_buffers):
            # Create input buffer with data
            data = np.arange(self.elements_per_buffer, dtype=np.float32) + (i * 100.0)
            buf_in = device.create_buffer(pywgpu.BufferDescriptor(
                label=f"Input Buffer {i}",
                size=self.buffer_size,
                usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_DST]
            ))
            queue.write_buffer(buf_in, 0, data.tobytes())
            self.input_buffers.append(buf_in)
            
            # Create output buffer
            buf_out = device.create_buffer(pywgpu.BufferDescriptor(
                label=f"Output Buffer {i}",
                size=self.buffer_size,
                usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC]
            ))
            self.output_buffers.append(buf_out)

        # 3. Bind Group
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(
                binding=0, 
                visibility=pywgpu.ShaderStages.compute, 
                ty=pywgpu.BufferBindingType.storage(read_only=True),
                count=self.num_buffers
            ),
            pywgpu.BindGroupLayoutEntry(
                binding=1, 
                visibility=pywgpu.ShaderStages.compute, 
                ty=pywgpu.BufferBindingType.storage(read_only=False),
                count=self.num_buffers
            )
        ]))
        
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[
                pywgpu.BindGroupEntry(binding=0, resource=[b.as_entire_binding() for b in self.input_buffers]),
                pywgpu.BindGroupEntry(binding=1, resource=[b.as_entire_binding() for b in self.output_buffers])
            ]
        ))

        # 4. Pipeline
        self.pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bg_layout]),
            module=self.shader,
            entry_point="main"
        ))

    def render(self, view, device, queue):
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, self.bind_group)
        pass_enc.dispatch_workgroups(self.num_buffers, 1, 1)
        pass_enc.end()
        
        queue.submit([encoder.finish()])
        
        # Verify first and last result
        asyncio.create_task(self.verify_results(device, queue))

    async def verify_results(self, device, queue):
        # We only do this once
        if hasattr(self, 'verified'): return
        self.verified = True
        
        print("Verifying Big Compute Buffers results...")
        
        for i in [0, self.num_buffers - 1]:
            # Read back output buffer i
            staging = device.create_buffer(pywgpu.BufferDescriptor(
                size=self.buffer_size,
                usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
            ))
            
            encoder = device.create_command_encoder()
            encoder.copy_buffer_to_buffer(self.output_buffers[i], 0, staging, 0, self.buffer_size)
            queue.submit([encoder.finish()])
            
            await staging.map_async(pywgpu.MapMode.read)
            data = staging.get_mapped_range()
            results = np.frombuffer(data, dtype=np.float32)
            staging.unmap()
            
            # Check first element
            # Expected: (i * 100.0 + 0) * 2.0
            expected_first = (i * 100.0) * 2.0
            print(f"Buffer {i}: first element result={results[0]}, expected={expected_first}")
            
            if abs(results[0] - expected_first) < 0.001:
                print(f"Buffer {i} verification PASSED")
            else:
                print(f"Buffer {i} verification FAILED")

if __name__ == "__main__":
    asyncio.run(run_example(BigComputeBuffersExample))
 village
 village
