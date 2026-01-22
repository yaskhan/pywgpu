from dataclasses import dataclass
from typing import List
from naga import ir, UniqueArena
from .overload_set import OverloadSet
from .rule import Rule

@dataclass(frozen=True)
class Regular:
    """
    A compact, efficient representation for the kinds of overload sets
    commonly seen for Naga IR mathematical functions.
    """
    # TODO: Implement Regular fields and OverloadSet methods
    
    def is_empty(self) -> bool:
        # TODO
        return False

    def min_arguments(self) -> int:
        # TODO
        return 0

    def max_arguments(self) -> int:
        # TODO
        return 0

    def arg(self, i: int, ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> 'Regular':
        # TODO
        return self

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'Regular':
        # TODO
        return self

    def most_preferred(self) -> Rule:
        # TODO
        raise NotImplementedError()
