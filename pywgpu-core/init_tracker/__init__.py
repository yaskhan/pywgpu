from typing import List, Any

class InitTracker:
    """
    Track initialization state of resources.
    """
    def __init__(self) -> None:
        self.uninitialized_ranges: List[Any] = []

    def check(self, range_start: int, range_end: int) -> bool:
        return True
