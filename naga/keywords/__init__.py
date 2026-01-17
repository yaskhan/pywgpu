from typing import Set
from .wgsl import RESERVED as WGSL_RESERVED

class Keywords:
    """
    Keyword management.
    """
    def __init__(self) -> None:
        self.reserved: Set[str] = set()

    def add_wgsl(self) -> None:
        self.reserved.update(WGSL_RESERVED)

    def is_reserved(self, word: str) -> bool:
        return word in self.reserved
