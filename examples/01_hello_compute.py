"""
To serve as an introduction to the pywgpu API, we will implement a simple
compute shader which takes a list of numbers on the CPU and doubles them on the GPU.

While this isn't a very practical example, you will see all the major components
of using pywgpu, including getting a device, running a shader, and transferring
data between the CPU and GPU.

If you time the recording and execution of this example you will certainly see that
running on the gpu is slower than doing the same calculation on the cpu. This is because
floating point multiplication is a very simple operation so the transfer/submission overhead
is quite a lot higher than the actual computation. This is normal and shows that the GPU
needs a lot higher work/transfer ratio to come out ahead.
"""

import asyncio
import argparse
from typing import List

import pywgpu


async def main():
    parser = argparse.ArgumentParser(description="Double numbers using GPU compute shader")
    parser.add_argument("numbers", nargs="*", type=float, help="Numbers to double")
    args = parser.parse_args()
    
    if not args.numbers:
        print("No arguments provided. Please provide a list of numbers to double.")
        return
    
    print(f"Parsed {len(args.numbers)} arguments")
    
    # Initialize the wgpu instance
    instance = pywgpu.Instance()
    
    # Request an adapter (physical GPU)
    adapter = await instance.request_adapter()
    
    if not adapter:
        print("No suitable adapter found")
        return
    
    print(f"Running on Adapter: {adapter.get_info()}")
    
    # Check if the adapter supports compute shaders
    downlevel_capabilities = adapter.get_downlevel_capabilities()
    if not downlevel_capabilities.flags.compute_shaders:
        print("Adapter does not support compute shaders")
        return
    
    # Create a device and queue from the adapter
    device, queue = await adapter.request_device(pywgpu.DeviceDescriptor(
        label=None,
        required_features=[],
        required_limits=pywgpu.Limits.downlevel_defaults(),
        experimental_features=pywgpu.ExperimentalFeatures.disabled(),
        memory_hints=pywgpu.MemoryHints.memory_usage,
        trace=pywgpu.Trace.off(),
    ))

    
    # Create a shader module from our shader code
    shader_code = """
        @group(0) @binding(0)
        var<storage, read> input_data: array<f32>;
        
        @group(0) @binding(1)
        var<storage, read_write> output_data: array<f32>;
        
        @compute @workgroup_size(64)
        fn doubleMe(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index < arrayLength(&input_data)) {
                output_data[index] = input_data[index] * 2.0;
            }
        }
    """
    
    module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(
        label=None,
        wgsl_code=shader_code,
    ))
    
    # Create input data buffer and upload data
    import struct
    packed_input_data = struct.pack(f'{len(args.numbers)}f', *args.numbers)
    input_data_buffer = device.create_buffer_init(
        label="Input Buffer",
        contents=packed_input_data,
        usage=pywgpu.BufferUsages.STORAGE
    )
    
    # Create output data buffer
    output_data_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label=None,
        size=len(args.numbers) * 4,
        usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_SRC],
        mapped_at_creation=False,
    ))
    
    # Create download buffer for reading results
    download_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label=None,
        size=len(args.numbers) * 4,
        usage=[pywgpu.BufferUsages.COPY_DST, pywgpu.BufferUsages.MAP_READ],
        mapped_at_creation=False,
    ))
    
    # Create bind group layout
    bind_group_layout = device.create_bind_group_layout(pywgpu.BindGroupLayoutDescriptor(
        label=None,
        entries=[
            # Input buffer
            pywgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=[pywgpu.ShaderStages.COMPUTE],
                ty=pywgpu.BindingType.buffer(
                    ty=pywgpu.BufferBindingType.storage(read_only=True),
                    min_binding_size=4,
                    has_dynamic_offset=False,
                ),
                count=None,
            ),
            # Output buffer
            pywgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=[pywgpu.ShaderStages.COMPUTE],
                ty=pywgpu.BindingType.buffer(
                    ty=pywgpu.BufferBindingType.storage(read_only=False),
                    min_binding_size=4,
                    has_dynamic_offset=False,
                ),
                count=None,
            ),
        ],
    ))
    
    # Create bind group
    bind_group = device.create_bind_group(pywgpu.BindGroupDescriptor(
        label=None,
        layout=bind_group_layout,
        entries=[
            pywgpu.BindGroupEntry(
                binding=0,
                resource=input_data_buffer.as_entire_binding(),
            ),
            pywgpu.BindGroupEntry(
                binding=1,
                resource=output_data_buffer.as_entire_binding(),
            ),
        ],
    ))
    
    # Create pipeline layout
    pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(
        label=None,
        bind_group_layouts=[bind_group_layout],
        immediate_size=0,
    ))
    
    # Create compute pipeline
    pipeline = device.create_compute_pipeline(pywgpu.ComputePipelineDescriptor(
        label=None,
        layout=pipeline_layout,
        module=module,
        entry_point="doubleMe",
        compilation_options=pywgpu.PipelineCompilationOptions.default(),
        cache=None,
    ))
    
    # Create command encoder
    encoder = device.create_command_encoder(pywgpu.CommandEncoderDescriptor(
        label=None,
    ))
    
    # Begin compute pass
    compute_pass = encoder.begin_compute_pass(pywgpu.ComputePassDescriptor(
        label=None,
        timestamp_writes=None,
    ))
    
    # Set pipeline and bind group
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group, [])
    
    # Dispatch workgroups
    workgroup_count = (len(args.numbers) + 63) // 64  # Ceiling division by 64
    compute_pass.dispatch_workgroups(workgroup_count, 1, 1)
    
    # End compute pass
    compute_pass.end()
    
    # Copy output buffer to download buffer
    encoder.copy_buffer_to_buffer(
        output_data_buffer,
        0,
        download_buffer,
        0,
        output_data_buffer.size,
    )
    
    # Finish encoding
    command_buffer = encoder.finish()
    
    # Submit work to queue
    queue.submit([command_buffer])

    
    # Wait for completion and read results
    await download_buffer.map_async(pywgpu.MapMode.READ)
    result_data = download_buffer.get_mapped_range()
    
    # Convert bytes back to f32 array
    import struct
    result = [struct.unpack('<f', result_data[i:i+4])[0] for i in range(0, len(result_data), 4)]
    
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())