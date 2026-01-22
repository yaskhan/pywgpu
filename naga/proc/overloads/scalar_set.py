from dataclasses import dataclass
from naga import ir

@dataclass(frozen=True)
class ScalarSet:
    """
    A set of Naga IR scalar types.
    """
    # TODO: Implement ScalarSet mirroring the bit-mask based implementation in Rust
    bits: int = 0
    
    # TODO: Add methods like contains(), insert(), etc.
