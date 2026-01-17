from typing import Generic, TypeVar, List, Any

T = TypeVar('T')

class ResourceMetadata(Generic[T]):
    """
    Metadata for tracked resources.
    """
    def __init__(self) -> None:
        self.owned: List[T] = []

    def tracker_assert_in_bounds(self, index: int) -> None:
        assert index < len(self.owned)
