from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import TlasDescriptor

class Tlas:
    """
    Top Level Acceleration Structure (TLAS).

    Contains instances of BLASes for ray tracing.

    Created with :meth:`Device.create_tlas`.
    """

    def __init__(self, inner: Any, descriptor: TlasDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def destroy(self) -> None:
        """Destroys the TLAS."""
        # In the Rust implementation, TLAS doesn't have an explicit destroy method
        # as it's handled by Rust's Drop trait. In Python, we implement it explicitly.
        if hasattr(self._inner, 'destroy'):
            self._inner.destroy()
        # If no inner destroy method, do nothing (resources managed by backend)
