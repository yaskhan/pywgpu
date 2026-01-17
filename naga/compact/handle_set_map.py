from typing import Generic, TypeVar, Dict, Optional
from ..arena import Handle

T = TypeVar('T')

class HandleMap(Generic[T]):
    """
    Map old handles to new handles.
    """
    def __init__(self) -> None:
        self.map: Dict[Handle[T], Handle[T]] = {}

    def insert(self, old: Handle[T], new: Handle[T]) -> None:
        self.map[old] = new

    def get(self, old: Handle[T]) -> Optional[Handle[T]]:
        return self.map.get(old)
