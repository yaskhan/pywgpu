from typing import Generic, TypeVar, Any

T = TypeVar('T')

class Handle(Generic[T]):
    """
    A strongly typed handle to an item in an Arena.
    """
    def __init__(self, index: int) -> None:
        self.index = index

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Handle):
            return self.index == other.index
        return False

    def __hash__(self) -> int:
        return hash(self.index)

    def __repr__(self) -> str:
        return f"Handle({self.index})"

    def __lt__(self, other: 'Handle[T]') -> bool:
        return self.index < other.index
