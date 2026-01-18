from typing import Optional, List, Any, Dict
from .block import Block

class FunctionArgument:
    """
    IR Function argument.
    """
    def __init__(self, name: Optional[str], ty: Any, binding: Optional[Any]) -> None:
        self.name = name
        self.ty = ty
        self.binding = binding

class FunctionResult:
    """
    IR Function result.
    """
    def __init__(self, ty: Any, binding: Optional[Any]) -> None:
        self.ty = ty
        self.binding = binding

class LocalVariable:
    """
    IR Local variable.
    """
    def __init__(self, name: Optional[str], ty: Any, init: Optional[Any]) -> None:
        self.name = name
        self.ty = ty
        self.init = init

class Function:
    """
    IR Function definition.
    """
    def __init__(self, name: Optional[str], result: Optional[FunctionResult], body: Block) -> None:
        self.name = name
        self.arguments: List[FunctionArgument] = []
        self.result: Optional[FunctionResult] = result
        self.local_variables: List[LocalVariable] = []
        self.expressions: List[Any] = [] # Arena[Expression]
        self.named_expressions: Dict[str, Any] = {}
        self.body: Block = body

    def new(name: Optional[str], result: Optional[FunctionResult], body: Block) -> 'Function':
        """Create a new function."""
        return Function(name, result, body)

    def add_argument(self, name: Optional[str], ty: Any, binding: Optional[Any]) -> FunctionArgument:
        """Add an argument to the function."""
        arg = FunctionArgument(name, ty, binding)
        self.arguments.append(arg)
        return arg

    def add_local_var(self, name: Optional[str], ty: Any, init: Optional[Any]) -> LocalVariable:
        """Add a local variable to the function."""
        var = LocalVariable(name, ty, init)
        self.local_variables.append(var)
        return var

    def add_expression(self, expr: Any) -> Any:
        """Add an expression to the function's expression arena."""
        self.expressions.append(expr)
        return len(self.expressions) - 1

    def set_named_expression(self, name: str, expr_handle: Any) -> None:
        """Set a named expression."""
        self.named_expressions[name] = expr_handle
