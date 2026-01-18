from typing import Any, Optional, Union

class Constant:
    """
    IR Constant definition.
    """
    def __init__(self, name: Optional[str], ty: int, value: Any) -> None:
        self.name = name
        self.ty = ty
        self.value = value
        self.special = None

    def literal(ty: int, value: Any) -> 'Constant':
        """Create a literal constant."""
        return Constant(None, ty, value)

    def composite(ty: int, elements: list) -> 'Constant':
        """Create a composite constant."""
        const = Constant(None, ty, None)
        const.components = elements
        return const

    def zero(ty: int) -> 'Constant':
        """Create a zero constant."""
        return Constant(None, ty, 0)
