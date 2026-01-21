U32_MAX = 0xFFFFFFFF


class NonMaxU32:
    """
    An unsigned 32-bit value known not to be u32::MAX.
    """

    def __init__(self, n: int):
        if n == U32_MAX:
            raise ValueError("NonMaxU32 cannot be u32::MAX")
        self._value = n

    def get(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"NonMaxU32({self._value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NonMaxU32):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)
