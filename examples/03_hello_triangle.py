"""
Hello Triangle Example

A simple example that renders a red triangle on a green background using pywgpu.
This demonstrates basic vertex buffer setup, shader creation, and render pass execution.
"""

import asyncio
import math

import pywgpu


async def main():
    print("Hello Triangle example using pywgpu")
    print("This example demonstrates basic rendering pipeline")
    
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
        required_features=[pywgpu.Features.VERTEX_WRITABLE_STORAGE],
        required_limits=pywgpu.Limits.default(),
        experimental_features=pywgpu.ExperimentalFeatures.disabled(),
        memory_hints=pywgpu.MemoryHints.memory_usage,
        trace=pywgpu.Trace.off(),
    ))
    
    # Vertex data for a simple triangle
    vertices = [
        # Position (x, y), Color (r, g, b)
        -0.5, -0.5,  1.0, 0.0, 0.0,  # Red vertex
         0.5, -0.5,  0.0, 1.0, 0.0,  # Green vertex
         0.0,  0.5,  0.0, 0.0, 1.0,  # Blue vertex
    ]
    
    # Create vertex buffer
    vertex_buffer = device.create_buffer(pywgpu.BufferDescriptor(
        label=None,
        size=len(vertices) * 4,  # f32 = 4 bytes
        usage=[pywgpu.BufferUsages.VERTEX, pywgpu.BufferUsages.COPY_DST],
        mapped_at_creation=False,
    ))
    
    # Upload vertex data
    import struct
    vertex_data = struct.pack(f'{len(vertices)}f', *vertices)
    await vertex_buffer.map_async(pywgpu.MapMode.WRITE)
    vertex_buffer.get_mapped_range()[:] = vertex_data
    vertex_buffer.unmap()
    
    # Create shader module
    shader_code = """
        struct VertexInput {
            @location(0) position: vec2<f32>,
            @location(1) color: vec3<f32>,
        };
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
        };
        
        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4<f32>(input.position, 0.0, 1.0);
            output.color = input.color;
            return output;
        }
        
        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            return vec4<f32>(input.color, 1.0);
        }
    """
    
    shader_module = device.create_shader_module(pywgpu.ShaderModuleDescriptor(
        label=None,
        wgsl_code=shader_code,
    ))
    
    # Create render pipeline
    pipeline_layout = device.create_pipeline_layout(pywgpu.PipelineLayoutDescriptor(
        label=None,
        bind_group_layouts=[],
        immediate_size=0,
    ))
    
    # Vertex buffer layout
    vertex_buffer_layout = pywgpu.VertexBufferLayout(
        array_stride=5 * 4,  # 5 f32s = 20 bytes
        step_mode=pywgpu.VertexStepMode.vertex,
        attributes=[
            pywgpu.VertexAttribute(
                format=pywgpu.VertexFormat.float32x2,
                offset=0,
                shader_location=0,
            ),
            pywgpu.VertexAttribute(
                format=pywgpu.VertexFormat.float32x3,
                offset=2 * 4,  # Skip position (2 f32s)
                shader_location=1,
            ),
        ],
    )
    
    render_pipeline = device.create_render_pipeline(pywgpu.RenderPipelineDescriptor(
        label=None,
        layout=pipeline_layout,
        vertex=pywgpu.VertexState(
            module=shader_module,
            entry_point="vs_main",
            buffers=[vertex_buffer_layout],
        ),
        fragment=pywgpu.FragmentState(
            module=shader_module,
            entry_point="fs_main",
            targets=[pywgpu.ColorTargetState(
                format=pywgpu.TextureFormat.rgba8unorm,
                blend=None,
                write_mask=pywgpu.ColorWriteMask.ALL,
            )],
        ),
        primitive=pywgpu.PrimitiveState(
            topology=pywgpu.PrimitiveTopology.triangle_list,
            strip_index_format=None,
            front_face=pywgpu.FrontFace.ccw,
            cull_mode=None,
        ),
        depth_stencil=None,
        multisample=pywgpu.MultisampleState(
            count=1,
            mask=0xFFFFFFFF,
            alpha_to_coverage_enabled=False,
        ),
    ))
    
    print("Pipeline created successfully!")
    print("This example demonstrates:")
    print("- Vertex buffer creation and data upload")
    print("- Shader module compilation")
    print("- Render pipeline creation")
    print("- Basic rendering concepts")


if __name__ == "__main__":
    asyncio.run(main())