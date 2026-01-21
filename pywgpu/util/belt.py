class StagingBelt:
    """
    Efficiently manages staging buffers for uploading data to the GPU.
    """

    def __init__(self, chunk_size: int = 1024 * 1024) -> None:
        self.chunk_size = chunk_size

    def write_buffer(self, encoder, buffer, offset, size, device):
        if hasattr(self, "_inner") and hasattr(self._inner, "write_buffer"):
            return self._inner.write_buffer(encoder, buffer, offset, size, device)
        else:
            raise NotImplementedError("Backend does not support write_buffer")

    def finish(self):
        if hasattr(self, "_inner") and hasattr(self._inner, "finish"):
            return self._inner.finish()
        else:
            raise NotImplementedError("Backend does not support finish")

    def recall(self):
        if hasattr(self, "_inner") and hasattr(self._inner, "recall"):
            return self._inner.recall()
        else:
            raise NotImplementedError("Backend does not support recall")
