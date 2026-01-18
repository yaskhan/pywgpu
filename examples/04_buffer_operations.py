"""
Buffer Operations Example

Demonstrates buffer creation, data upload, and basic GPU buffer operations using pywgpu.
This example shows how to create different types of buffers and work with them.
"""

import asyncio
import struct

import pywgpu


async def main():
    print("Buffer Operations example using pywgpu")
    print("This example demonstrates buffer creation and data management")
    
    # Initialize the wgpu instance
    instance = pywgpu.Instance()
    
    # Request an adapter (physical GPU)
    adapter = await instance.request_adapter()
    
    if not adapter:
        print("No suitable adapter found")
        return
    
    print(f"Running on Adapter: {adapter.get_info()}")
    
    # Create a device and queue from the adapter
    device = await adapter.request_device(pywgpu.DeviceDescriptor(
        label=None,
        required_features=[],
        required_limits=pywgpu.Limits.default(),
        experimental_features=pywgpu.ExperimentalFeatures.disabled(),
        memory_hints=pywgpu.MemoryHints.memory_usage,
        trace=pywgpu.Trace.off(),
    ))
    
    # 1. Create a vertex buffer
    print("\\n1. Creating vertex buffer...")
    vertex_data = [
        -0.5, -0.5, 1.0, 0.0, 0.0,  # Position (x, y), Color (r, g, b)
         0.5, -0.5, 0.0, 1.0, 0.0,
         0.0,  0.5, 0.0, 0.0, 1.0,
    ]
    
    vertex_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label="Vertex Buffer",
        size=len(vertex_data) * 4,  # f32 = 4 bytes
        usage=[pywgpu.BufferUsages.VERTEX, pywgpu.BufferUsages.COPY_DST],
        mapped_at_creation=False,
    ))
    
    # Upload vertex data
    packed_vertex_data = struct.pack(f'{len(vertex_data)}f', *vertex_data)
    await vertex_buffer.map_async(pywgpu.MapMode.WRITE)
    mapped_data = vertex_buffer.get_mapped_range()
    mapped_data[:] = packed_vertex_data
    vertex_buffer.unmap()
    print(f"  - Vertex buffer created with {len(vertex_data)} floats")
    print(f"  - Buffer size: {vertex_buffer.size} bytes")
    print(f"  - Buffer usage: {vertex_buffer.usage}")
    
    # 2. Create an index buffer
    print("\\n2. Creating index buffer...")
    index_data = [0, 1, 2]  # Triangle indices
    
    index_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label="Index Buffer",
        size=len(index_data) * 4,  # u32 = 4 bytes
        usage=[pywgpu.BufferUsages.INDEX, pywgpu.BufferUsages.COPY_DST],
        mapped_at_creation=False,
    ))
    
    # Upload index data
    packed_index_data = struct.pack(f'{len(index_data)}I', *index_data)
    await index_buffer.map_async(pywgpu.MapMode.WRITE)
    mapped_data = index_buffer.get_mapped_range()
    mapped_data[:] = packed_index_data
    index_buffer.unmap()
    print(f"  - Index buffer created with {len(index_data)} indices")
    print(f"  - Buffer size: {index_buffer.size} bytes")
    
    # 3. Create a uniform buffer
    print("\\n3. Creating uniform buffer...")
    # Uniform buffer for transformation matrix (4x4 matrix of f32)
    uniform_data = [
        1.0, 0.0, 0.0, 0.0,  # Row 0
        0.0, 1.0, 0.0, 0.0,  # Row 1
        0.0, 0.0, 1.0, 0.0,  # Row 2
        0.0, 0.0, 0.0, 1.0,  # Row 3
    ]
    
    uniform_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label="Uniform Buffer",
        size=len(uniform_data) * 4,
        usage=[pywgpu.BufferUsages.UNIFORM, pywgpu.BufferUsages.COPY_DST],
        mapped_at_creation=False,
    ))
    
    # Upload uniform data
    packed_uniform_data = struct.pack(f'{len(uniform_data)}f', *uniform_data)
    await uniform_buffer.map_async(pywgpu.MapMode.WRITE)
    mapped_data = uniform_buffer.get_mapped_range()
    mapped_data[:] = packed_uniform_data
    uniform_buffer.unmap()
    print(f"  - Uniform buffer created for 4x4 matrix")
    print(f"  - Buffer size: {uniform_buffer.size} bytes")
    
    # 4. Create a storage buffer for compute operations
    print("\\n4. Creating storage buffer...")
    compute_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    storage_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label="Storage Buffer",
        size=len(compute_data) * 4,
        usage=[pywgpu.BufferUsages.STORAGE, pywgpu.BufferUsages.COPY_DST, pywgpu.BufferUsages.COPY_SRC],
        mapped_at_creation=False,
    ))
    
    # Upload compute data
    packed_compute_data = struct.pack(f'{len(compute_data)}f', *compute_data)
    await storage_buffer.map_async(pywgpu.MapMode.WRITE)
    mapped_data = storage_buffer.get_mapped_range()
    mapped_data[:] = packed_compute_data
    storage_buffer.unmap()
    print(f"  - Storage buffer created for compute operations")
    print(f"  - Initial data: {compute_data}")
    print(f"  - Buffer size: {storage_buffer.size} bytes")
    
    # 5. Create a read-only buffer for results
    print("\\n5. Creating read buffer...")
    read_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label="Read Buffer",
        size=storage_buffer.size,
        usage=[pywgpu.BufferUsages.COPY_DST, pywgpu.BufferUsages.MAP_READ],
        mapped_at_creation=False,
    ))
    
    # Demonstrate buffer copy operation
    print("\\n6. Demonstrating buffer copy...")
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(
        storage_buffer,
        0,
        read_buffer,
        0,
        storage_buffer.size,
    )
    
    command_buffer = command_encoder.finish()
    device.queue.submit([command_buffer])
    
    # Read back the data
    await read_buffer.map_async(pywgpu.MapMode.READ)
    read_data = read_buffer.get_mapped_range()
    
    # Convert back to float array
    result_data = [struct.unpack('<f', read_data[i:i+4])[0] for i in range(0, len(read_data), 4)]
    print(f"  - Copied data back from GPU: {result_data}")
    read_buffer.unmap()
    
    print("\\nBuffer operations completed successfully!")
    print("\\nThis example demonstrated:")
    print("- Different buffer types (vertex, index, uniform, storage)")
    print("- Buffer creation with various usage flags")
    print("- Data upload to GPU buffers")
    print("- Buffer mapping and unmapping")
    print("- Buffer-to-buffer copy operations")
    print("- Reading data back from GPU buffers")


if __name__ == "__main__":
    asyncio.run(main())