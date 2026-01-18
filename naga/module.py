from __future__ import annotations
from typing import List

class Type:
    """Placeholder for a Naga IR Type."""
    pass

class Constant:
    """Placeholder for a Naga IR Constant."""
    pass

class GlobalVariable:
    """Placeholder for a Naga IR GlobalVariable."""
    pass

class Function:
    """Placeholder for a Naga IR Function."""
    pass

class Module:
    """Represents a Naga Intermediate Representation (IR) module."""
    def __init__(self) -> None:
        self.types: List[Type] = []
        self.constants: List[Constant] = []
        self.global_variables: List[GlobalVariable] = []
        self.functions: List[Function] = []
