from typing import Protocol, List, runtime_checkable, TypeVar, Any
from naga import ir, UniqueArena
from naga.proc import TypeResolution
from .rule import Rule

T = TypeVar("T", bound="OverloadSet")

@runtime_checkable
class OverloadSet(Protocol):
    """
    A Protocol for types representing a set of Naga IR type rules.
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
        """
        ...

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'OverloadSet':
        """Limit self to overloads whose arguments are all concrete types."""
        ...

    def most_preferred(self) -> Rule:
        """
        Return the most preferred candidate.
        """
        ...

    def overload_list(self, gctx: Any = None) -> List[Rule]:
        """Return a type rule for each of the overloads in self."""
        ...

    def allowed_args(self, i: int, gctx: Any = None) -> List[TypeResolution]:
        """Return a list of the types allowed for argument i."""
        ...

    def for_debug(self, types: UniqueArena[ir.Type]) -> Any:
        """Return an object that can be formatted with repr()."""
        ...
