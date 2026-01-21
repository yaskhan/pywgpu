from typing import Any, Optional, Tuple, List, Union


class Dispatch:
    """
    Interface for wgpu backend dispatch.

    This mirrors the Dispatch trait in wgpu-core, abstracting over
    different backends (Vulkan, Metal, DX12, etc.).
    """

    def request_adapter(self, options: Any) -> Any:
        raise NotImplementedError

    def create_surface(self, target: Any) -> Any:
        raise NotImplementedError
