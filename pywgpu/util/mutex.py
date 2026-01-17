# Mutex utilities
# Rust uses parking_lot or std based on features.
# Python uses threading.Lock usually.

from threading import Lock

class Mutex:
    """Simple wrapper around a lock."""
    def __init__(self):
        self._lock = Lock()

    def lock(self):
        self._lock.acquire()

    def unlock(self):
        self._lock.release()
