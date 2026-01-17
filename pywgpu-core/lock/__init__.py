from typing import Any
from .rank import Rank

class Mutex:
    """
    Ranked Mutex.
    """
    def __init__(self, rank: Rank, data: Any) -> None:
        self.rank = rank
        self.data = data

    def lock(self) -> Any:
        return self.data
