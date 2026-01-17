from typing import Optional, List, Any
from .block import Block

class FunctionResult:
    """
    IR Function result.
    """
    def __init__(self, ty: Any, binding: Optional[Any]) -> None:
        self.ty = ty
        self.binding = binding

class Function:
    """
    IR Function definition.
    """
    def __init__(self, name: Optional[str], result: Optional[FunctionResult], body: Block) -> None:
        self.name = name
        self.arguments = []
        self.result = result
        self.local_variables = []
        self.expressions = [] # Arena[Expression]
        self.named_expressions = {}
        self.body = body
