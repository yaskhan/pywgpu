from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import BlasDescriptor


class Blas:
    """
    Bottom Level Acceleration Structure (BLAS).

    Contains geometry for ray tracing.

    Created with :meth:`Device.create_blas`.
    """

    def __init__(self, inner: Any, descriptor: BlasDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def destroy(self) -> None:
        """Destroys the BLAS."""
        # In the Rust implementation, BLAS doesn't have an explicit destroy method
        # as it's handled by Rust's Drop trait. In Python, we implement it explicitly.
        if hasattr(self._inner, "destroy"):
            self._inner.destroy()
        # If no inner destroy method, do nothing (resources managed by backend)
