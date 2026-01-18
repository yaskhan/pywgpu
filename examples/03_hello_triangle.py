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
    device, queue = await adapter.request_device(pywgpu.DeviceDescriptor(
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
    
    # Create vertex buffer and upload data
    import struct
    vertex_data = struct.pack(f'{len(vertices)}f', *vertices)
    vertex_buffer = device.create_buffer_init(
        label="Triangle Vertex Buffer",
        contents=vertex_data,
        usage=pywgpu.BufferUsages.VERTEX
    )
    
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
    
    # Create a texture to render into (since we don't have a window/surface)
    texture = device.create_texture(pywgpu.TextureDescriptor(
        label="Render Texture",
        size=(800, 600, 1),
        mip_level_count=1,
        sample_count=1,
        dimension=pywgpu.TextureDimension.d2,
        format=pywgpu.TextureFormat.rgba8unorm,
        usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.COPY_SRC],
        view_formats=[],
    ))
    
    texture_view = texture.create_view()
    
    # Create command encoder
    encoder = device.create_command_encoder(pywgpu.CommandEncoderDescriptor(
        label="Render Encoder"
    ))
    
    # Begin render pass
    render_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
        label="Triangle Render Pass",
        color_attachments=[
            pywgpu.RenderPassColorAttachment(
                view=texture_view,
                resolve_target=None,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color.GREEN),
                    store=pywgpu.StoreOp.store,
                ),
                depth_slice=None,
            )
        ],
        depth_stencil_attachment=None,
        timestamp_writes=None,
        occlusion_query_set=None,
    ))
    
    # Set pipeline and vertex buffer
    render_pass.set_pipeline(render_pipeline)
    render_pass.set_vertex_buffer(0, vertex_buffer)
    
    # Draw the triangle
    render_pass.draw(vertices=3, instances=1, first_vertex=0, first_instance=0)
    
    # End render pass
    render_pass.end()
    
    # Finish encoding and submit
    command_buffer = encoder.finish()
    queue.submit([command_buffer])
    
    print("Triangle rendering commands submitted successfully!")
    print("This example now demonstrates:")
    print("- Vertex buffer creation and data upload")
    print("- Shader module compilation")
    print("- Render pipeline creation")
    print("- Texture and View creation")
    print("- Command encoding for a render pass")
    print("- Drawing commands execution")



if __name__ == "__main__":
    asyncio.run(main())