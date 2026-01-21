from enum import Enum
from typing import Optional, Any


class Statement(Enum):
    """
    IR Statement enum.
    """

    BLOCK = "block"
    IF = "if"
    SWITCH = "switch"
    LOOP = "loop"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    KILL = "kill"
    BARRIER = "barrier"
    STORE = "store"
    IMAGE_STORE = "image-store"
    ATOMIC = "atomic"
    CALL = "call"
    RAY_QUERY = "ray-query"
    # ... others
