from typing import Callable, TypeVar, Generic
import threading

T = TypeVar("T")


class RacyLock(Generic[T]):
    """
    A thread-safe, lazy-initialization mechanism.
    """

    def __init__(self, init: Callable[[], T]):
        self._init = init
        self._lock = threading.Lock()
        self._value: T | None = None

    def get(self) -> T:
        """
        Loads the internal value, initializing it if required.
        """
        if self._value is not None:
            return self._value

        with self._lock:
            if self._value is None:
                self._value = self._init()
            return self._value

    def __call__(self) -> T:
        return self.get()
