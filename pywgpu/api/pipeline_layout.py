from typing import Any
from pywgpu_types.descriptors import PipelineLayoutDescriptor


class PipelineLayout:
    """
    Handle to a pipeline layout.

    Describes the mapping between resources (bind groups) and headers.

    Created with :meth:`Device.create_pipeline_layout`.
    """

    def __init__(self, inner: Any, descriptor: PipelineLayoutDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor
