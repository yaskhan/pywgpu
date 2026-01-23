from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
from .function import Function
from .type import Type
from .constant import Constant


@dataclass
class GlobalVariable:
    """
    Global variable in the module.
    """
    name: Optional[str]
    space: Any  # AddressSpace
    binding: Optional[Dict[str, int]]
    ty: Any  # Type handle
    init: Optional[Any] = None  # Expression handle


class EntryPoint:
    """
    IR Entry point definition.
    """

    def __init__(
        self,
        name: str,
        stage: str,
        function: Function,
        early_depth_test: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.stage = stage
        self.function = function
        self.early_depth_test = early_depth_test
        self.workgroup_size = [1, 1, 1]
        self.group_to_binding_map = {}


class Module:
    """
    Naga Intermediate Representation (IR) module.
    """

    def __init__(self) -> None:
        self.types: List[Type] = []
        self.constants: List[Constant] = []
        self.global_variables: List[Any] = []
        self.global_expressions: List[Any] = []
        self.functions: List[Function] = []
        self.entry_points: List[EntryPoint] = []
        self.named_expressions: Dict[str, Any] = {}
    def add_global_variable(self, name: Optional[str], space: Any, binding: Optional[Dict[str, int]], ty: Any, init: Optional[Any] = None) -> int:
        """Add a global variable to the module."""
        var = GlobalVariable(name, space, binding, ty, init)
        self.global_variables.append(var)
        return len(self.global_variables) - 1

    def get_global_handle_by_name(self, name: str) -> Optional[int]:
        """Find a global variable handle by its name."""
        for i, var in enumerate(self.global_variables):
            if var.name == name:
                return i
        return None
        """Add a type to the module's type arena."""
        typ = Type(name, inner)
        self.types.append(typ)
        return len(self.types) - 1

    def add_constant(self, name: Optional[str], inner: Any, value: Any) -> int:
        """Add a constant to the module's constant arena."""
        const = Constant(name, inner, value)
        self.constants.append(const)
        return len(self.constants) - 1

    def add_function(
        self, name: Optional[str], result: Optional[Any], body: Any
    ) -> Function:
        """Add a function to the module."""
        func = Function(name, result, body)
        self.functions.append(func)
        return func

    def add_entry_point(
        self,
        name: str,
        stage: str,
        function: Function,
        early_depth_test: Optional[Any] = None,
    ) -> EntryPoint:
        """Add an entry point to the module."""
        entry = EntryPoint(name, stage, function, early_depth_test)
        self.entry_points.append(entry)
        return entry

    def set_named_expression(self, name: str, expr_handle: Any) -> None:
        """Set a named expression at module level."""
        self.named_expressions[name] = expr_handle
