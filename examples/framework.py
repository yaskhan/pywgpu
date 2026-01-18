import asyncio
import time
import argparse
from typing import Optional, List, Tuple
import pywgpu

class Example:
    """Base class for pywgpu examples."""
    TITLE: str = "pywgpu Example"
    
    def __init__(self):
        pass

    async def init(
        self,
        config: pywgpu.SurfaceConfiguration,
        adapter: pywgpu.Adapter,
        device: pywgpu.Device,
        queue: pywgpu.Queue,
    ):
        """Initialize example resources."""
        pass

    def resize(
        self,
        config: pywgpu.SurfaceConfiguration,
        device: pywgpu.Device,
        queue: pywgpu.Queue,
    ):
        """Handle window resize."""
        pass

    def update(self, delta_time: float):
        """Update example state."""
        pass

    def render(
        self,
        view: pywgpu.TextureView,
        device: pywgpu.Device,
        queue: pywgpu.Queue,
    ):
        """Render a frame."""
        pass

async def run_example(example_cls: type[Example]):
    parser = argparse.ArgumentParser(description=example_cls.TITLE)
    # Add any common arguments here
    args = parser.parse_args()

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
        label=example_cls.TITLE,
        required_features=[],
        required_limits=pywgpu.Limits.default(),
    ))

    # Mock surface setup
    width, height = 800, 600
    config = pywgpu.SurfaceConfiguration(
        usage=pywgpu.TextureUsages.RENDER_ATTACHMENT,
        format=pywgpu.TextureFormat.rgba8unorm,
        width=width,
        height=height,
        present_mode=pywgpu.PresentMode.fifo,
        alpha_mode=pywgpu.CompositeAlphaMode.auto,
        view_formats=[pywgpu.TextureFormat.rgba8unorm],
    )

    example = example_cls()
    await example.init(config, adapter, device, queue)

    print(f"Initialized {example_cls.TITLE}")

    # Mock render loop
    # In a real app with a window backend, this would be tied to the window events.
    
    texture = device.create_texture(pywgpu.TextureDescriptor(
        label="Mock Swapchain Texture",
        size=(width, height, 1),
        mip_level_count=1,
        sample_count=1,
        dimension=pywgpu.TextureDimension.d2,
        format=pywgpu.TextureFormat.rgba8unorm,
        usage=[pywgpu.TextureUsages.RENDER_ATTACHMENT, pywgpu.TextureUsages.COPY_SRC],
    ))
    view = texture.create_view()

    last_time = time.time()
    
    # Run a few frames to verify
    for frame_idx in range(5):
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        
        example.update(delta_time)
        example.render(view, device, queue)
        
        print(f"Rendered frame {frame_idx}")
        
    print(f"Example {example_cls.TITLE} ran successfully (mock loop).")
