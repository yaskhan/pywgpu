"""
Validation canary for internal testing.

Stores the text of any validation errors that have occurred since
the last call to `get_and_reset`.

Each value is a validation error and a message associated with it,
or None if the error has no message from the API.

This is used for internal wgpu testing only and _must not_ be used
as a way to check for errors.

This works as a module-level singleton because tests run in separate
processes, so each test gets its own canary.

This prevents the issue of one validation error terminating the
entire process.
"""

from typing import List
import threading


class ValidationCanary:
    """Flag for internal testing.
    
    Stores validation error messages in a thread-safe manner.
    """
    
    def __init__(self):
        self._inner: List[str] = []
        self._lock = threading.Lock()
    
    def add(self, msg: str) -> None:
        """Add a validation error message.
        
        Args:
            msg: The validation error message to store.
        
        Note:
            This is for internal use only.
        """
        with self._lock:
            self._inner.append(msg)
    
    def get_and_reset(self) -> List[str]:
        """Returns any API validation errors that have occurred in this process
        since the last call to this function.
        
        Returns:
            List of validation error messages. The list is cleared after retrieval.
        """
        with self._lock:
            errors = self._inner.copy()
            self._inner.clear()
            return errors


# Global validation canary instance
VALIDATION_CANARY = ValidationCanary()

