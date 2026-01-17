from typing import Any, Optional
from pywgpu_types.descriptors import PipelineCacheDescriptor

class PipelineCache:
    """
    Cache for compiled pipelines.
    
    Created with :meth:`Device.create_pipeline_cache`.
    """
    
    def __init__(self, inner: Any, descriptor: PipelineCacheDescriptor) -> None:
        self._inner = inner
        self._descriptor = descriptor
    
    def get_data(self) -> Optional[bytes]:
        """Returns the cache data."""
        pass
