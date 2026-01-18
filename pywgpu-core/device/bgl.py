from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class BindGroupLayoutEntry:
    """
    Bind group layout entry.
    """
    binding: int
    visibility: int
    buffer: Optional[Any] = None
    sampler: Optional[Any] = None
    texture: Optional[Any] = None
    storage_texture: Optional[Any] = None
    external_texture: Optional[Any] = None


class EntryMap:
    """
    Map of binding index to entry.
    """
    def __init__(self) -> None:
        self.inner: Dict[int, BindGroupLayoutEntry] = {}

    def add(self, entry: BindGroupLayoutEntry) -> None:
        """Add an entry to the map."""
        self.inner[entry.binding] = entry

    def get(self, binding: int) -> Optional[BindGroupLayoutEntry]:
        """Get an entry by binding index."""
        return self.inner.get(binding)

    def remove(self, binding: int) -> Optional[BindGroupLayoutEntry]:
        """Remove an entry by binding index."""
        return self.inner.pop(binding, None)

    def clear(self) -> None:
        """Clear all entries."""
        self.inner.clear()

    def items(self):
        """Return all entries."""
        return self.inner.items()

    def values(self):
        """Return all entry values."""
        return self.inner.values()

    def keys(self):
        """Return all binding indices."""
        return self.inner.keys()
