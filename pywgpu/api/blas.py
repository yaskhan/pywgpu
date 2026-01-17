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
        pass
