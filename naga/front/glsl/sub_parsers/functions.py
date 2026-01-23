"""
Parser for GLSL function declarations and calls.

This module handles parsing of function declarations, definitions, and calls.
"""

from typing import Any, Optional, List, Dict
from enum import Enum
from ..token import TokenValue


class FunctionKind(Enum):
    """Types of function declarations."""
    DECLARATION = "declaration"
    DEFINITION = "definition"


class ParameterDirection(Enum):
    """Parameter direction qualifiers."""
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    CONST = "const"


class FunctionParser:
    """Parser for GLSL function declarations and calls."""
    
    def __init__(self, lexer: Any):
        self.lexer = lexer
        self.errors: List[str] = []
        self.function_signatures: Dict[str, Any] = {}
    
    def parse_function_declaration(self, ctx: Any, frontend: Any, return_type: Any, name: str, qualifiers: List[Any]) -> Optional[Any]:
        """Parse a function declaration or definition."""
        # Current token is '('
        from ..token import TokenValue
        from ....ir import Function as IRFunction, FunctionResult, Block
        from ..context import Context
        
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        parameters = self.parse_parameter_list(ctx, frontend)
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        
        token = ctx.peek(frontend)
        if token and token.value == TokenValue.SEMICOLON:
            # Function prototype
            ctx.bump(frontend)
            # Store prototype info
            sig = self._create_signature(name, parameters)
            self.add_function_overload(sig, {"name": name, "return_type": return_type, "parameters": parameters})
            return {"kind": FunctionKind.DECLARATION, "name": name}
        
        elif token and token.value == TokenValue.LEFT_BRACE:
            # Function definition
            # Create a new IR Function
            res = FunctionResult(ty=return_type, binding=None) if return_type != "void" else None
            ir_func = IRFunction(name=name, result=res, body=Block.from_vec([]))
            
            # Create a translation context for this function
            translation_ctx = Context(frontend, ir_func)
            
            # Add parameters to context and IR
            for p in parameters:
                arg_handle = ir_func.add_argument(name=p['name'], ty=p['type'], binding=None)
                if p['name']:
                    translation_ctx.define_variable(p['name'], ('arg', arg_handle))
            
            # Parse body using the context
            body_block = frontend.statement_parser.parse_compound_statement(ctx, frontend, translation_ctx)
            ir_func.body = body_block
            
            # If name is 'main', add as entry point
            if name == "main":
                # Entry point handling will be done by GlslParser
                pass
            else:
                # Add to module's functions
                handle_index = len(frontend.module.functions)
                frontend.module.functions.append(ir_func)
                # Update signature with definition handle
                sig = self._create_signature(name, parameters)
                self.add_function_overload(sig, {"name": name, "return_type": return_type, "parameters": parameters, "handle": handle_index})
                
            return {"kind": FunctionKind.DEFINITION, "name": name, "handle": ir_func}
            
        return None

    def _create_signature(self, name: str, parameters: List[Any]) -> str:
        """Create a unique signature for a function for overload resolution."""
        params_str = ",".join([str(p['type']) for p in parameters])
        return f"{name}({params_str})"

    def _skip_block(self, ctx: Any, frontend: Any) -> None:
        """Skip a balanced block of braces."""
        from ..token import TokenValue
        ctx.expect(frontend, TokenValue.LEFT_BRACE)
        depth = 1
        while depth > 0:
            token = ctx.next(frontend)
            if token is None:
                break
            if token.value == TokenValue.LEFT_BRACE:
                depth += 1
            elif token.value == TokenValue.RIGHT_BRACE:
                depth -= 1
    
    def parse_parameter_list(self, ctx: Any, frontend: Any) -> List[Any]:
        """Parse function parameter list."""
        from ..token import TokenValue
        parameters = []
        
        while ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.RIGHT_PAREN:
            # Parse parameter qualifiers (const, in, out, inout)
            param_qualifiers = []
            while ctx.peek(frontend) and ctx.peek(frontend).value in [
                TokenValue.CONST, TokenValue.IN, TokenValue.OUT, TokenValue.INOUT
            ]:
                param_qualifiers.append(ctx.bump(frontend))
            
            # Parse type
            param_type = frontend.type_parser_impl.parse_type_specifier(ctx, frontend)
            
            # Parse identifier (optional for prototypes)
            param_name = None
            if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.IDENTIFIER:
                param_name, _ = ctx.expect_ident(frontend)
            
            parameters.append({
                "name": param_name,
                "type": param_type,
                "qualifiers": param_qualifiers
            })
            
            if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.COMMA:
                ctx.bump(frontend)
            else:
                break
                
        return parameters
    
    def parse_function_call(self, ctx: Any, frontend: Any, translation_ctx: Any, name: str) -> Any:
        """Parse and resolve a function call."""
        from ..token import TokenValue
        ctx.expect(frontend, TokenValue.LEFT_PAREN)
        args = []
        if ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.RIGHT_PAREN:
            while True:
                args.append(frontend.expression_parser.parse_assignment(ctx, frontend, translation_ctx))
                if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.COMMA:
                    ctx.bump(frontend)
                else:
                    break
        ctx.expect(frontend, TokenValue.RIGHT_PAREN)
        
        # Resolve function call
        resolved = self.resolve_function_call(name, args, frontend, translation_ctx)
        
        from ....ir import Expression, ExpressionType
        if resolved:
            # Check if it's a builtin mathematical function
            if hasattr(resolved, 'kind') and resolved.kind.value == "math":
                from ....ir import MathFunction
                try:
                    math_fun = MathFunction[name.upper()]
                    expr = Expression(type=ExpressionType.MATH, math_fun=math_fun, math_arg=args[0] if args else None)
                    if len(args) > 1: expr = Expression(type=ExpressionType.MATH, math_fun=math_fun, math_arg=args[0], math_arg1=args[1])
                    if len(args) > 2: expr = Expression(type=ExpressionType.MATH, math_fun=math_fun, math_arg=args[0], math_arg1=args[1], math_arg2=args[2])
                    return translation_ctx.add_expression(expr)
                except (KeyError, AttributeError):
                    pass
            
            # Handle user-defined function calls (ExpressionType.CALL_RESULT)
            if isinstance(resolved, dict) and 'handle' in resolved:
                from ....ir import Statement
                func_handle = resolved['handle']
                expr = Expression(type=ExpressionType.CALL_RESULT, call_result=func_handle)
                expr_handle = translation_ctx.add_expression(expr)
                translation_ctx.add_statement(Statement.new_call(function=func_handle, arguments=args, result=expr_handle))
                return expr_handle
            
        # Fallback for unresolved or complex builtins
        return translation_ctx.add_expression(Expression(type=ExpressionType.BINARY, binary_op=None, binary_left=None, binary_right=None))

    def resolve_function_call(self, function_name: str, args: List[Any], frontend: Any, translation_ctx: Any) -> Optional[Any]:
        """Resolve a function call to a specific function overload."""
        arg_types = [translation_ctx.get_expression_type(arg) for arg in args]
        
        # 1. Try to find user functions
        for sig, overloads in self.function_signatures.items():
            if sig.startswith(f"{function_name}("):
                for overload in overloads:
                    params = overload.get('parameters', [])
                    if len(params) == len(args):
                        # Match argument types
                        match = True
                        for i, param in enumerate(params):
                            expected_type = param.get('type') # This should be a TypeInner
                            if isinstance(expected_type, int):
                                expected_type = frontend.module.types[expected_type].inner
                            
                            if not self.check_implicit_conversions(expected_type, arg_types[i]):
                                match = False
                                break
                        if match:
                            return overload
                            
        # 2. Try to find builtins
        builtin = frontend.builtins.get_builtin_function(function_name, args)
        if builtin:
            return builtin
            
        return None

    def add_function_overload(self, signature: str, func_info: Any) -> None:
        """
        Add a function overload to the lookup table.
        
        Args:
            signature: Function signature string
            func_info: Function information
        """
        if signature not in self.function_signatures:
            self.function_signatures[signature] = []
        self.function_signatures[signature].append(func_info)
    
    def check_implicit_conversions(self, expected_type: Any, actual_type: Any) -> bool:
        """Check if an implicit conversion is possible between types."""
        if expected_type is None or actual_type is None:
            return False
        if expected_type == actual_type:
            return True
            
        from ....ir import TypeInnerType, ScalarKind
        
        # Scalar promotions
        if expected_type.type == TypeInnerType.SCALAR and actual_type.type == TypeInnerType.SCALAR:
            e_scalar = expected_type.scalar
            a_scalar = actual_type.scalar
            if a_scalar.kind == ScalarKind.SINT and e_scalar.kind == ScalarKind.UINT: return True
            if a_scalar.kind == ScalarKind.SINT and e_scalar.kind == ScalarKind.FLOAT: return True
            if a_scalar.kind == ScalarKind.UINT and e_scalar.kind == ScalarKind.FLOAT: return True

        # Vector promotions
        if expected_type.type == TypeInnerType.VECTOR and actual_type.type == TypeInnerType.VECTOR:
            if expected_type.vector.size == actual_type.vector.size:
                e_scalar = expected_type.vector.scalar
                a_scalar = actual_type.vector.scalar
                if a_scalar.kind == ScalarKind.SINT and e_scalar.kind == ScalarKind.UINT: return True
                if a_scalar.kind == ScalarKind.SINT and e_scalar.kind == ScalarKind.FLOAT: return True
                if a_scalar.kind == ScalarKind.UINT and e_scalar.kind == ScalarKind.FLOAT: return True
                    
        return False
