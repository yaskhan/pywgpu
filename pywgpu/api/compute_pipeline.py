from typing import Any, TYPE_CHECKING
from pywgpu_types.pipeline import ComputePipelineDescriptor

if TYPE_CHECKING:
    from .bind_group_layout import BindGroupLayout


class ComputePipeline:
    """
    Handle to a compute pipeline.

    Created with :meth:`Device.create_compute_pipeline`.
    """

    def __init__(self, inner: Any, descriptor: ComputePipelineDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def get_bind_group_layout(self, index: int) -> "BindGroupLayout":
        """
        Returns the bind group layout at the given index.

        Args:
            index: The index of the bind group layout.
        """
        from .bind_group_layout import BindGroupLayout

        if hasattr(self._inner, "get_bind_group_layout"):
            layout_inner = self._inner.get_bind_group_layout(index)
            return BindGroupLayout(layout_inner)
        else:
            raise NotImplementedError("Backend does not support get_bind_group_layout")
