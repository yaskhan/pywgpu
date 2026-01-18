from typing import Any, TYPE_CHECKING
from pywgpu_types.descriptors import QuerySetDescriptor

class QuerySet:
    """
    Handle to a query set.
    
    Used to record results of queries (e.g. occlusion, timestamp) on a pass.
    
    Created with :meth:`Device.create_query_set`.
    """
    
    def __init__(self, inner: Any, descriptor: QuerySetDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def destroy(self) -> None:
        """Destroys the query set."""
        if hasattr(self._inner, 'destroy'):
            self._inner.destroy()
        # If no inner destroy method, do nothing (resources managed by backend)
