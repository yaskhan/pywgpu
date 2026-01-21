from typing import Generic, TypeVar, Any

T = TypeVar("T")


class BadHandle(Exception):
    """
    Error raised when a handle is invalid.
    """

    def __init__(self, kind: str, index: int):
        self.kind = kind
        self.index = index
        super().__init__(f"Handle {index} of {kind} is either not present, or inaccessible yet")


class Handle(Generic[T]):
    """
    A strongly typed reference to an arena item.

    A Handle value can be used as an index into an Arena or UniqueArena.
    """

    def __init__(self, index: int) -> None:
        self._index = index

    @property
    def index(self) -> int:
        """Returns the index of this handle."""
        return self._index

    @classmethod
    def from_usize(cls, index: int) -> "Handle[T]":
        """Convert a usize index into a Handle[T]."""
        if index < 0:
            raise ValueError("Index cannot be negative")
        return cls(index)

    @classmethod
    def from_usize_unchecked(cls, index: int) -> "Handle[T]":
        """Convert a usize index into a Handle[T], without range checks."""
        return cls(index)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Handle):
            return self._index == other._index
        return False

    def __hash__(self) -> int:
        return hash(self._index)

    def __repr__(self) -> str:
        return f"[{self._index}]"

    def __lt__(self, other: "Handle[T]") -> bool:
        return self._index < other._index

    def __str__(self) -> str:
        return str(self._index)

    def write_prefixed(self, prefix: str) -> str:
        """Write this handle's index preceded by prefix."""
        return f"{prefix}{self._index}"
