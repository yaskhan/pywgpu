from typing import Any
from pywgpu_types.sampler import SamplerDescriptor


class Sampler:
    """
    Handle to a sampler.

    A Sampler defines how a texture is sampled (filtering, address modes, etc.).

    Created with :meth:`Device.create_sampler`.
    """

    def __init__(self, inner: Any, descriptor: SamplerDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor
