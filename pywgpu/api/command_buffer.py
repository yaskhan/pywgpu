from typing import Any

class CommandBuffer:
    """
    Handle to a command buffer.
    
    A CommandBuffer contains a list of commands that have been encoded and 
    are ready for submission to a :class:`Queue`.
    
    Created with :meth:`CommandEncoder.finish`.
    """
    
    def __init__(self, inner: Any) -> None:
        self._inner = inner
