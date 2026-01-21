from typing import Optional, Any


class PipelineCache:
    """
    Handle to a pipeline cache.

    A pipeline cache can be used to accelerate the creation of pipelines.
    """

    def __init__(self, inner: Any, descriptor: Any) -> None:
        self._inner = inner
        self._descriptor = descriptor

    def get_data(self) -> Optional[bytes]:
        """
        Returns the data contained in the pipeline cache.
        """
        if hasattr(self._inner, "get_data"):
            return self._inner.get_data()
        return None
