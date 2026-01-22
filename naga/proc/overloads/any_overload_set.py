from typing import Union
from .overload_set import OverloadSet
from .regular import Regular
from .list import ListOverloadSet
from .constructor_set import ConstructorSet

# TODO: AnyOverloadSet should be a Union of the different overload set types
AnyOverloadSet = Union[Regular, ListOverloadSet, ConstructorSet]
