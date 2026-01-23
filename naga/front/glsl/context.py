from typing import Any, Optional, Dict, List
from ...ir import Function, Block, Expression, LocalVariable, Type
from ...arena import Arena

class Context:
    """
    Translation context for lowering GLSL to NAGA IR.
    
    This class manages local variable arenas, expression arenas,
    and statement blocks for a single function.
    """
    
    def __init__(self, frontend: Any, function: Function):
        self.frontend = frontend
        self.function = function
        self.scopes: List[Dict[str, int]] = [{}]
        self.emit_start = 0
        self.statements: List[Statement] = []
        
    def push_scope(self) -> None:
        """Push a new variable scope."""
        self.scopes.append({})
        
    def pop_scope(self) -> None:
        """Pop the current variable scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
            
    def define_variable(self, name: str, handle: Any) -> None:
        """Define a variable in the current scope. Handle is (kind, index)."""
        self.scopes[-1][name] = handle
        
    def resolve_variable(self, name: str) -> Optional[Any]:
        """Resolve a variable name to its (kind, index)."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
        
    def add_expression(self, expr: Expression) -> int:
        """Add an expression to the current function's arena and return its handle."""
        return self.function.add_expression(expr)
        
    def add_local_variable(self, name: Optional[str], ty: Any, init: Optional[Any] = None) -> int:
        """Add a local variable to the current function."""
        var_handle = self.function.add_local_var(name, ty, init)
        if name:
            self.define_variable(name, ('local', var_handle))
        return var_handle

    def emit_expressions(self, block: List[Statement]) -> None:
        """Emit all pending expressions into the given block."""
        end = len(self.function.expressions)
        if self.emit_start < end:
            from ...ir import Statement
            block.append(Statement.new_emit(range(self.emit_start, end)))
            self.emit_start = end

    def add_statement(self, stmt: Statement) -> None:
        """Add a statement to the current function's body."""
        self.emit_expressions(self.statements)
        self.statements.append(stmt)

    def get_expression_type(self, handle: int) -> Optional[Any]:
        """Get the type (TypeInner) of an expression handle."""
        from ...ir import ExpressionType, TypeInner, Scalar, ScalarKind
        
        expr = self.function.expressions[handle]
        
        if expr.type == ExpressionType.LITERAL:
            if isinstance(expr.literal, bool):
                return TypeInner.new_scalar(Scalar(ScalarKind.BOOL, 1))
            elif isinstance(expr.literal, int):
                return TypeInner.new_scalar(Scalar(ScalarKind.SINT, 4))
            elif isinstance(expr.literal, float):
                return TypeInner.new_scalar(Scalar(ScalarKind.FLOAT, 4))
                
        elif expr.type == ExpressionType.LOCAL_VARIABLE:
            var = self.function.local_variables[expr.local_variable]
            return self.frontend.module.types[var.ty].inner
            
        elif expr.type == ExpressionType.GLOBAL_VARIABLE:
            var = self.frontend.module.global_variables[expr.global_variable]
            return self.frontend.module.types[var.ty].inner
            
        elif expr.type == ExpressionType.FUNCTION_ARGUMENT:
            arg = self.function.arguments[expr.function_argument]
            return self.frontend.module.types[arg.ty].inner
            
        elif expr.type == ExpressionType.BINARY:
            # Simplified: binary ops usually return same type as operands (except relational)
            return self.get_expression_type(expr.binary_left)
            
        elif expr.type == ExpressionType.SWIZZLE:
            base_type = self.get_expression_type(expr.swizzle_vector)
            if base_type and base_type.vector:
                from ...ir import Vector
                return TypeInner.new_vector(expr.swizzle_size, base_type.vector.scalar)
            
        # Fallback
        return None
