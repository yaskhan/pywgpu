from typing import Any
from pywgpu_types.descriptors import BindGroupLayoutDescriptor


class BindGroupLayout:
    """
    Handle to a bind group layout.

    Defines the interface between a set of resources bound in a :class:`BindGroup`
    and the shader stages that use them.

    Created with :meth:`Device.create_bind_group_layout`.
    """

    def __init__(self, inner: Any, descriptor: BindGroupLayoutDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor
