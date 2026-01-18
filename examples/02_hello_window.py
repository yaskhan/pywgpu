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
    
    # Create a surface (window) - this would require a proper window backend
    # For now, we'll just create a basic surface without window
    print("Hello Window example would require a window backend")
    print("This is a placeholder for the window example")
    
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
    
    print("Device created successfully!")
    print("This example demonstrates basic pywgpu initialization")
    print("In a full implementation, this would create a window and render loop")


if __name__ == "__main__":
    asyncio.run(main())