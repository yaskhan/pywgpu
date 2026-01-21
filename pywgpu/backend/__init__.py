import os
import sys
from typing import Optional, Any
from pywgpu_types.instance import InstanceDescriptor
from pywgpu_types.backend import Backends


def get_backend(descriptor: Optional[InstanceDescriptor] = None) -> Any:
    """
    Returns the appropriate backend implementation based on the descriptor
    and the current platform.
    """
    if descriptor is None:
        descriptor = InstanceDescriptor()

    backends = descriptor.backends

    # Logic for selecting the backend
    # 1. Check for wasm (Browser/WebGPU)
    if sys.platform == "emscripten" or (backends & Backends.BROWSER_WEBGPU):
        try:
            from .webgpu import WebGPUBackend

            return WebGPUBackend(descriptor)
        except ImportError:
            if sys.platform == "emscripten":
                raise RuntimeError("WebGPU backend not found on Emscripten platform")

    # 2. Check for wgpu-core (Vulkan, DX12, Metal, etc. via HAL)
    if backends & (Backends.VULKAN | Backends.DX12 | Backends.METAL | Backends.GL):
        from .wgpu_core import WgpuCoreBackend

        return WgpuCoreBackend(descriptor)

    # 3. Fallback to default wgpu-core if nothing specified
    from .wgpu_core import WgpuCoreBackend

    return WgpuCoreBackend(descriptor)
