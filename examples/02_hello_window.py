"""
Hello Window Example

Shows how to create a window and render into it using pywgpu.
This is a basic example that demonstrates the window creation and rendering pipeline.
"""

import asyncio
import time

import pywgpu


async def main():
    # Initialize the wgpu instance
    instance = pywgpu.Instance()
    
    # Request an adapter (physical GPU)
    adapter = await instance.request_adapter(pywgpu.RequestAdapterOptions(
        power_preference=pywgpu.PowerPreference.high_performance,
        force_fallback_adapter=False,
        compatible_surface=None,
    ))
    
    if not adapter:
        print("No suitable adapter found")
        return
    
    print(f"Running on Adapter: {adapter.get_info()}")
    
    # Create a device and queue from the adapter
    device, queue = await adapter.request_device(pywgpu.DeviceDescriptor(
        label="Main Device",
        required_features=[],
        required_limits=pywgpu.Limits.default(),
    ))

    
    print("Device created successfully!")

    # In a real application with a window backend (like GLFW or SDL2):
    # surface = instance.create_surface(window_handle)
    # config = surface.get_default_config(adapter, width, height)
    # surface.configure(device, config)
    
    print("\nSimulating a simple render loop (clearing to green)...")
    
    # Since we don't have a real window surface, we'll create a mock texture
    # that represents our 'swapchain' image.
    width, height = 800, 600
    texture = device.create_texture(pywgpu.TextureDescriptor(
        label="Mock Swapchain Texture",
        size=(width, height, 1),
        mip_level_count=1,
        sample_count=1,
        dimension=pywgpu.TextureDimension.d2,
        format=pywgpu.TextureFormat.rgba8unorm,
        usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT],
    ))
    
    view = texture.create_view()
    
    # Create command encoder
    encoder = device.create_command_encoder(pywgpu.CommandEncoderDescriptor(
        label="Initial Clear Encoder"
    ))
    
    # Begin render pass to clear the 'screen'
    render_pass = encoder.begin_render_pass(pywgpu.RenderPassDescriptor(
        label="Clear Pass",
        color_attachments=[
            pywgpu.RenderPassColorAttachment(
                view=view,
                resolve_target=None,
                ops=pywgpu.Operations(
                    load=pywgpu.LoadOp.clear(pywgpu.Color.GREEN),
                    store=pywgpu.StoreOp.store,
                ),
                depth_slice=None,
            )
        ]
    ))
    
    # End the render pass
    render_pass.end()
    
    # Submit the commands
    command_buffer = encoder.finish()
    queue.submit([command_buffer])
    
    print("Successfully submitted commands to clear the surface!")
    print("\nThis example demonstrates:")
    print("- Instance and Adapter initialization")
    print("- Device and Queue acquisition")
    print("- How a surface would be cleared in a render loop")
    print("- Basic command encoding and submission")



if __name__ == "__main__":
    asyncio.run(main())