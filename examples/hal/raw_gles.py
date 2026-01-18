"""
Raw GLES Example - Python Translation

This example shows interop with raw GLES contexts - the ability to hook up
pywgpu-hal to an existing context and draw into it.

This is a Python translation of wgpu-hal/examples/raw-gles.rs
"""

import sys
import os

# Add pywgpu-hal to path
hal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pywgpu-hal')
if hal_path not in sys.path:
    sys.path.insert(0, hal_path)

import lib as hal
import pywgpu_types as wgt


def fill_screen(exposed_adapter, width: int, height: int):
    """Fill the screen with a blue color.
    
    Args:
        exposed_adapter: The exposed HAL adapter.
        width: Screen width.
        height: Screen height.
    """
    print("Filling the screen")
    
    # Open the device
    open_device = exposed_adapter.adapter.open(
        features=wgt.Features.NONE,
        limits=wgt.Limits.downlevel_defaults(),
        memory_hints=wgt.MemoryHints()
    )
    
    device = open_device.device
    queue = open_device.queue
    
    # Create default framebuffer texture
    format = wgt.TextureFormat.RGBA8_UNORM_SRGB
    texture = hal.Texture.default_framebuffer(format)
    
    # Create texture view
    view = device.create_texture_view(
        texture,
        hal.TextureViewDescriptor(
            label=None,
            format=format,
            dimension=wgt.TextureViewDimension.D2,
            usage=wgt.TextureUses.COLOR_TARGET,
            range=wgt.ImageSubresourceRange()
        )
    )
    
    # Create command encoder
    encoder = device.create_command_encoder(
        hal.CommandEncoderDescriptor(
            label=None,
            queue=queue
        )
    )
    
    # Create fence for synchronization
    fence = device.create_fence()
    
    # Render pass descriptor
    rp_desc = hal.RenderPassDescriptor(
        label=None,
        extent=wgt.Extent3d(
            width=width,
            height=height,
            depth_or_array_layers=1
        ),
        sample_count=1,
        color_attachments=[
            hal.ColorAttachment(
                target=hal.Attachment(
                    view=view,
                    usage=wgt.TextureUses.COLOR_TARGET
                ),
                depth_slice=None,
                resolve_target=None,
                ops=hal.AttachmentOps.STORE | hal.AttachmentOps.LOAD_CLEAR,
                clear_value=wgt.Color.BLUE
            )
        ],
        depth_stencil_attachment=None,
        multiview_mask=None,
        timestamp_writes=None,
        occlusion_query_set=None
    )
    
    # Encode and submit
    encoder.begin_encoding(None)
    encoder.begin_render_pass(rp_desc)
    encoder.end_render_pass()
    cmd_buf = encoder.end_encoding()
    
    queue.submit([cmd_buf], [], (fence, 0))
    
    print("Screen filled with blue color")


def main():
    """Main function for desktop platforms."""
    try:
        import glfw
        from OpenGL import GL
    except ImportError:
        print("This example requires glfw and PyOpenGL")
        print("Install with: pip install glfw PyOpenGL")
        return
    
    print("Initializing external GL context")
    
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    
    # Request OpenGL ES 3.0+ context
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    
    # Create window
    window = glfw.create_window(
        800, 600,
        "pywgpu raw GLES example (press ESC to exit)",
        None, None
    )
    
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return
    
    # Make context current
    glfw.make_context_current(window)
    
    print("Hooking up to pywgpu-hal")
    
    # Create HAL adapter from external context
    # This would use the GL function loader
    try:
        exposed = hal.Adapter.new_external(
            lambda name: glfw.get_proc_address(name),
            wgt.GlBackendOptions()
        )
    except Exception as e:
        print(f"Failed to initialize GL adapter: {e}")
        glfw.terminate()
        return
    
    # Main loop
    while not glfw.window_should_close(window):
        # Get framebuffer size
        width, height = glfw.get_framebuffer_size(window)
        
        if width > 0 and height > 0:
            # Fill screen
            fill_screen(exposed, width, height)
            
            # Swap buffers
            glfw.swap_buffers(window)
        
        # Poll events
        glfw.poll_events()
        
        # Check for ESC key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
    
    # Cleanup
    glfw.terminate()
    print("Example completed")


if __name__ == "__main__":
    main()
