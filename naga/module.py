from __future__ import annotations
from typing import Any

class Module:
    """Represents a Naga Intermediate Representation (IR) module."""
    def __init__(self) -> None:
        self.functions: list[Any] = []
        self.types: list[Any] = []
        self.constants: list[Any] = []
