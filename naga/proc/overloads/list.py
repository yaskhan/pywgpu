from dataclasses import dataclass
from typing import List
from naga import ir, UniqueArena
from .overload_set import OverloadSet
from .rule import Rule

@dataclass(frozen=True)
class ListOverloadSet: # Renamed from List to avoid conflict with typing.List
    """
    A straightforward list of overloads.
    """
    rules: List[Rule]
    # TODO: Implement OverloadSet methods
    
    def is_empty(self) -> bool:
        return len(self.rules) == 0

    def min_arguments(self) -> int:
        # TODO
        return 0

    def max_arguments(self) -> int:
        # TODO
        return 0

    def arg(self, i: int, ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> 'ListOverloadSet':
        # TODO
        return self

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'ListOverloadSet':
        # TODO
        return self

    def most_preferred(self) -> Rule:
        # TODO
        raise NotImplementedError()
