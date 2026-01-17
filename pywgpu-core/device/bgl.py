from typing import Any, Dict

class BindGroupLayoutEntry:
    """
    Bind group layout entry.
    """
    pass

class EntryMap:
    """
    Map of binding index to entry.
    """
    def __init__(self) -> None:
        self.inner: Dict[int, Any] = {}
