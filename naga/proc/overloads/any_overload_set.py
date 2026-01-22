from typing import Union
from .regular import Regular
from .list import ListOverloadSet

AnyOverloadSet = Union[Regular, ListOverloadSet]
