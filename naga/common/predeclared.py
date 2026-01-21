from typing import Any, Optional
from enum import Enum


class PredeclaredType(Enum):
    """
    Types that are predeclared in the IR.
    """
    ATOMIC_COMPARE_EXCHANGE_WEAK_RESULT = "atomic_compare_exchange_weak_result"
    MODF_RESULT = "modf_result"
    FREXP_RESULT = "frexp_result"

    def struct_name(self) -> str:
        """
        Generate a struct name for this predeclared type.
        """
        if self == PredeclaredType.ATOMIC_COMPARE_EXCHANGE_WEAK_RESULT:
            return "__atomic_compare_exchange_result"
        elif self == PredeclaredType.MODF_RESULT:
            return "__modf_result"
        elif self == PredeclaredType.FREXP_RESULT:
            return "__frexp_result"
        else:
            return f"__{self.value}"
