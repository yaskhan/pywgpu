from dataclasses import dataclass
from typing import List, Optional
from naga import ir, UniqueArena
from .overload_set import OverloadSet
from .rule import Rule

@dataclass(frozen=True)
class ConstructorSet:
    """
    Overload set for constructors.
    """
    # TODO: Implement fields and OverloadSet methods
    
    def is_empty(self) -> bool:
        # TODO
        return False

    def min_arguments(self) -> int:
        # TODO
        return 0

    def max_arguments(self) -> int:
        # TODO
        return 0

    def arg(self, i: int, ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> 'ConstructorSet':
        # TODO
        return self

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'ConstructorSet':
        # TODO
        return self

    def most_preferred(self) -> Rule:
        # TODO
        raise NotImplementedError()

@dataclass(frozen=True)
class ConstructorSize:
    # TODO: Implement ConstructorSize (Scalar, Vector, Matrix)
    pass
