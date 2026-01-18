from typing import Callable, TypeVar, Generic
import threading

T = TypeVar("T")

class RacyLock(Generic[T]):
    """
    A thread-safe, lazy-initialization mechanism.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._value: T | None = None

    def get_or_init(self, initializer: Callable[[], T]) -> T:
        if self._value is not None:
            return self._value

        with self._lock:
            if self._value is None:
                self._value = initializer()
            return self._value
