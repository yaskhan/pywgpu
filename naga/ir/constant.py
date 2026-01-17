from typing import Any, Optional

class Constant:
    """
    IR Constant definition.
    """
    def __init__(self, name: Optional[str], special: Any, inner: Any) -> None:
        self.name = name
        self.special = special
        self.inner = inner
