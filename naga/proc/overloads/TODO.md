# Naga Overload Resolution Implementation Task

This directory is a Python port of `naga/src/proc/overloads` from the Rust `wgpu` repository.
It implements overload resolution for builtin functions in Naga IR.

## Progress Tracking

- [ ] `rule.py`: `Rule` and `Conclusion` types. (TODO)
- [ ] `overload_set.py`: `OverloadSet` Protocol (trait) definition. (TODO)
- [ ] `constructor_set.py`: `ConstructorSet` implementation. (TODO)
- [ ] `regular.py`: `Regular` overload set implementation. (TODO)
- [ ] `list.py`: `List` overload set implementation. (TODO)
- [ ] `any_overload_set.py`: `AnyOverloadSet` enum/union. (TODO)
- [ ] `scalar_set.py`: `ScalarSet` utility. (TODO)
- [ ] `mathfunction.py`: Math function overloads registry. (TODO)
- [ ] `utils.py`: Helpers for constructing overload sets. (TODO)
- [ ] `one_bits_iter.py`: Bit manipulation utility. (TODO)
- [ ] `__init__.py`: Package exports. (TODO)

## Python Skeleton with TODOs

### `rule.py`
```python
from dataclasses import dataclass
from typing import List, Union
from naga import ir, UniqueArena
from naga.proc import TypeResolution

@dataclass
class Rule:
    """A single type rule."""
    arguments: List[TypeResolution]
    conclusion: 'Conclusion'

class Conclusion:
    """The result type of a Rule."""
    # TODO: Implement Conclusion as a Union of Value(TypeInner) and Predeclared(PredeclaredType)
    pass

class MissingSpecialType(Exception):
    """Special type is not registered within the module."""
    pass
```

### `overload_set.py`
```python
from typing import Protocol, List, runtime_checkable
from naga import ir, UniqueArena
from naga.proc import TypeResolution
from .rule import Rule

@runtime_checkable
class OverloadSet(Protocol):
    """A Protocol for types representing a set of Naga IR type rules."""
    
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
        """Find the overloads that could accept a given argument."""
        ...

    def concrete_only(self, types: UniqueArena[ir.Type]) -> 'OverloadSet':
        """Limit self to overloads whose arguments are all concrete types."""
        ...

    def most_preferred(self) -> Rule:
        """Return the most preferred candidate."""
        ...
```

### `regular.py`
```python
from dataclasses import dataclass
from .overload_set import OverloadSet
# TODO: Implement Regular overload set
```

### `mathfunction.py`
```python
from naga import ir
from .overload_set import OverloadSet

def get_math_function_overloads(fun: ir.MathFunction) -> OverloadSet:
    """Return the overload set for a given math function."""
    # TODO: Implement mapping from MathFunction to its OverloadSet
    pass
```
