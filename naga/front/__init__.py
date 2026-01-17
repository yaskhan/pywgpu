from typing import Any, Protocol

class Parser(Protocol):
    """
    Base trait for all parsers.
    """
    def parse(self, source: Any) -> Any:
        ...
