from typing import Iterator

class OneBitsIter:
    """
    Utility for iterating over the indices of set bits in an integer.
    """
    def __init__(self, bits: int):
        self.bits = bits

    def __iter__(self) -> Iterator[int]:
        bits = self.bits
        index = 0
        while bits > 0:
            if bits & 1:
                yield index
            bits >>= 1
            index += 1
