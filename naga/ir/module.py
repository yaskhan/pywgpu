from typing import Any, List

class Module:
    """
    Naga Intermediate Representation (IR) module.
    """
    def __init__(self) -> None:
        self.types = []
        self.constants = []
        self.functions = []
        self.entry_points = []
