from typing import Protocol, List, runtime_checkable, TypeVar
from naga import ir, UniqueArena
from naga.proc import TypeResolution
from .rule import Rule

T = TypeVar("T", bound="OverloadSet")

@runtime_checkable
class OverloadSet(Protocol):
    """
    A Protocol for types representing a set of Naga IR type rules.
    
    Given an expression like `max(x, y)`, there are multiple type rules that
    could apply, depending on the types of `x` and `y`.
    """
    
    def is_empty(self) -> bool:
        """Return true if self is the empty set of overloads."""
        ...

    def min_arguments(self) -> int:
        """Return the smallest number of arguments in any type rule in the set."""
        ...

    def max_arguments(self) -> int:
        """Return the largest number of arguments in any type rule in the set."""
        ...

    def arg(self, i: int, ty: ir.TypeInner, types: UniqueArena[ir.Type]) -> 'OverloadSet':
        """
        Find the overloads that could accept a given argument.
        
        Return a new overload set containing those members of `self` that could
        accept a value of type `ty` for their `i`'th argument, once
        feasible automatic conversions have been applied.
        """
        ...

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'OverloadSet':
        """Limit `self` to overloads whose arguments are all concrete types."""
        ...

    def most_preferred(self) -> Rule:
        """
        Return the most preferred candidate.
        
        Rank the candidates in `self` as described in WGSL's overload
        resolution algorithm, and return a singleton set containing the
        most preferred candidate.
        """
        ...
