import struct
import asyncio
from typing import List

import pywgpu
from framework import Example, run_example

class SubgroupExample(Example):
    TITLE = "Subgroup Operations Example"

    async def init(self, config, adapter, device, queue):
        # Check for SUBGROUP feature
        if not adapter.features.contains(pywgpu.Features.SUBGROUP):
            print("ERROR: SUBGROUP feature not supported on this adapter.")
            print("This example requires hardware support for subgroup operations.")
            self.done = True
            return

        # 1. Shader
        # Requires 'subgroups' feature and 'enable subgroups;' in WGSL
        shader_code = """
            enable subgroups;

            @group(0) @binding(0) var<storage, read_write> output_data: array<u32>;

            @compute @workgroup_size(64)
            fn main(
                @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
                @builtin(subgroup_size) subgroup_size: u32,
                @builtin(global_invocation_id) global_id: vec3<u32>
            ) {
                // Each invocation contributes its own subgroup_invocation_id.
                // subgroupAdd will sum these across all active invocations in the subgroup.
                let val = subgroup_invocation_id;
                let sum = subgroupAdd(val);
                
                // Store result for thread 0 of each subgroup
                if (subgroup_invocation_id == 0u) {
                    // There can be multiple subgroups in a workgroup (size 64).
                    // We use global_id.x / subgroup_size to get a unique index for this subgroup.
                    let subgroup_index = global_id.x / subgroup_size;
                    output_data[subgroup_index] = sum;
                    
                    // Also store the subgroup size so we can verify the sum on CPU
                    // We store it in the high 16 bits
                    output_data[subgroup_index] |= (subgroup_size << 16u);
                }
            }
        """
        
        self.shader = device.create_shader_module(pywgpu.ShaderModuleDescriptor(wgsl_code=shader_code))

        # 2. Buffers
        # We'll allocate enough space for many subgroups (though we only dispatch 1 workgroup of 64)
        self.output_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Output Buffer",
            size=1024, # 256 * 4 bytes
            usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC]
        ))
        
        self.staging_buf = device.create_buffer(pywgpu.BufferDescriptor(
            label="Staging Buffer",
            size=1024,
            usage=[pywgpu.BufferUsages.MAP_READ, pywgpu.BufferUsages.COPY_DST]
        ))

        # 3. Bind Group
        self.bg_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(entries=[
            pywgpu.BindGroupLayoutEntry(
                binding=0, 
                visibility=pywgpu.ShaderStages.compute, 
                ty=pywgpu.BufferBindingType.storage(read_only=False)
            )
        ]))
        self.bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
            layout=self.bg_layout,
            entries=[pywgpu.BindGroupEntry(binding=0, resource=self.output_buf.as_entire_binding())]
        ))

        # 4. Pipeline
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
            pass_comp.dispatch_workgroups(1, 1, 1) # Full workgroup of 64
            pass_comp.end()
            
            encoder.copy_buffer_to_buffer(self.output_buf, 0, self.staging_buf, 0, 1024)
            queue.submit([encoder.finish()])
            
            async def read_results():
                await self.staging_buf.map_async(pywgpu.MapMode.read)
                data = self.staging_buf.get_mapped_range()
                # Read results
                results = struct.unpack("256I", data)
                
                # Filter out entries where subgroup_size (high 16 bits) is > 0
                for i, r in enumerate(results):
                    size = r >> 16
                    if size > 0:
                        sub_sum = r & 0xFFFF
                        # Expected sum of 0..size-1 is size*(size-1)/2
                        expected = size * (size - 1) // 2
                        print(f"Subgroup {i}: Size = {size}, Sum = {sub_sum} (Expected = {expected})")
                
                self.staging_buf.unmap()
            
            asyncio.create_task(read_results())
            self.done = True

        # Standard framework clear
        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
            color_attachments=[pywgpu.RenderPassColorAttachment(
                view=view,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color(r=0.2, g=0.1, b=0.3, a=1.0)), 
                    store=pywgpu.StoreOp.store
                )
            )]
        ))
        pass_enc.end()
        queue.submit([encoder.finish()])

if __name__ == "__main__":
    # We MUST request the SUBGROUP feature in the config
    config = {"required_features": [pywgpu.Features.SUBGROUP]}
    asyncio.run(run_example(SubgroupExample, config=config))
