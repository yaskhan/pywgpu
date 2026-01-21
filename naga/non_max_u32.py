from typing import Optional

U32_MAX = 0xFFFFFFFF


class NonMaxU32:
    """
    An unsigned 32-bit value known not to be u32::MAX.

    A NonMaxU32 value can represent any value in the range 0 .. u32::MAX - 1,
    and an Option<NonMaxU32> is still a 32-bit value. In other words,
    NonMaxU32 is just like NonZeroU32, except that a different value is
    missing from the full u32 range.

    Since zero is a very useful value in indexing, NonMaxU32 is more useful
    for representing indices than NonZeroU32.
    """

    def __init__(self, n: int):
        if n == U32_MAX:
            raise ValueError("NonMaxU32 cannot be u32::MAX")
        self._value = n

    @classmethod
    def new(cls, n: int) -> Optional["NonMaxU32"]:
        """Construct a NonMaxU32 whose value is n, if possible."""
        if n == U32_MAX:
            return None
        return cls(n)

    @classmethod
    def new_unchecked(cls, n: int) -> "NonMaxU32":
        """
        Construct a NonMaxU32 whose value is n.

        Safety:
        The value of n must not be u32::MAX.
        """
        return cls(n)

    @classmethod
    def from_usize_unchecked(cls, index: int) -> "NonMaxU32":
        """
        Construct a NonMaxU32 whose value is index.

        Safety:
        The value of index must be strictly less than u32::MAX.
        """
        return cls(index)

    def get(self) -> int:
        """Return the value of self as an int."""
        return self._value

    def checked_add(self, n: int) -> Optional["NonMaxU32"]:
        """
        Add n to self, returning None if the result would be u32::MAX.
        """
        result = self._value + n
        if result == U32_MAX:
            return None
        return NonMaxU32(result)

    def __repr__(self) -> str:
        return f"NonMaxU32({self._value})"

    def __str__(self) -> str:
        return str(self._value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NonMaxU32):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __lt__(self, other: "NonMaxU32") -> bool:
        return self._value < other._value

    def __le__(self, other: "NonMaxU32") -> bool:
        return self._value <= other._value

    def __gt__(self, other: "NonMaxU32") -> bool:
        return self._value > other._value

    def __ge__(self, other: "NonMaxU32") -> bool:
        return self._value >= other._value
