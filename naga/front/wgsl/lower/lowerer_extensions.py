"""
Additional lowering methods for WGSL lowerer.

This module contains helper methods for lowering expressions and statements.
"""


def add_lowering_methods(lowerer_class):
    """Add lowering helper methods to Lowerer class."""
    
    # Expression lowering methods
    def _lower_literal(self, ast_expr, ctx):
        """Lower literal expression."""
        from ...ir import Expression as IRExpression, Literal, ScalarValue
        
        # ast_expr.data is a LiteralExpression
        value = ast_expr.data.value
        
        # Simple mapping of Python types to IR ScalarValue
        if isinstance(value, bool):
            ir_value = ScalarValue.BOOL(value)
        elif isinstance(value, int):
            ir_value = ScalarValue.I32(value)
        elif isinstance(value, float):
            ir_value = ScalarValue.F32(value)
        else:
            # Assume it's a parsed number object or something similar
            ir_value = ScalarValue.F32(float(str(value)))
        
        # Create IR literal expression
        literal = Literal(ir_value)
        expr = IRExpression.LITERAL(literal)
        
        return ctx.add_expression(expr)
    
    def _lower_identifier(self, ast_expr, ctx):
        """Lower identifier expression."""
        from ...ir import Expression as IRExpression
        
        # ast_expr.data is an Ident
        ident_name = ast_expr.data.name
        
        # Resolve identifier to variable/constant/function
        # Check local variables first
        if ident_name in ctx.local_table:
            local_handle = ctx.local_table[ident_name]
            # If it's a handle, we assume it's a local variable handle
            expr = IRExpression.LOCAL_VARIABLE(local_handle)
            return ctx.add_expression(expr)
        
        # Check global variables
        if ident_name in self.var_map:
            var_handle = self.var_map[ident_name]
            expr = IRExpression.GLOBAL_VARIABLE(var_handle)
            return ctx.add_expression(expr)
        
        # Check constants
        if ident_name in self.const_map:
            const_handle = self.const_map[ident_name]
            expr = IRExpression.CONSTANT(const_handle)
            return ctx.add_expression(expr)
        
        # Check functions
        if ident_name in self.function_map:
            func_handle = self.function_map[ident_name]
            # This would usually be a call result or similar, but naked 
            # function name is often used in calls.
            return func_handle
        
        # Unknown identifier - error
        return None
    
    def _lower_binary(self, ast_expr, ctx):
        """Lower binary operation."""
        from ...ir import Expression as IRExpression, BinaryOperator
        
        # ast_expr.data is a BinaryExpression
        binary_data = ast_expr.data
        left_handle = self._lower_expression(binary_data.left, ctx)
        right_handle = self._lower_expression(binary_data.right, ctx)
        
        # Map AST operator to IR BinaryOperator
        op_map = {
            "+": BinaryOperator.ADD,
            "-": BinaryOperator.SUBTRACT,
            "*": BinaryOperator.MULTIPLY,
            "/": BinaryOperator.DIVIDE,
            "%": BinaryOperator.MODULO,
            "==": BinaryOperator.EQUAL,
            "!=": BinaryOperator.NOT_EQUAL,
            "<": BinaryOperator.LESS,
            "<=": BinaryOperator.LESS_EQUAL,
            ">": BinaryOperator.GREATER,
            ">=": BinaryOperator.GREATER_EQUAL,
            "&&": BinaryOperator.LOGICAL_AND,
            "||": BinaryOperator.LOGICAL_OR,
            "&": BinaryOperator.AND,
            "|": BinaryOperator.OR,
            "^": BinaryOperator.XOR,
            "<<": BinaryOperator.SHIFT_LEFT,
            ">>": BinaryOperator.SHIFT_RIGHT,
        }
        op = op_map.get(binary_data.op, BinaryOperator.ADD)
        
        # Create binary operation expression
        expr = IRExpression.BINARY(left=left_handle, op=op, right=right_handle)
        
        return ctx.add_expression(expr)
    
    def _lower_unary(self, ast_expr, ctx):
        """Lower unary operation."""
        from ...ir import Expression as IRExpression, UnaryOperator
        
        # ast_expr.data is a UnaryExpression
        unary_data = ast_expr.data
        operand_handle = self._lower_expression(unary_data.expr, ctx)
        
        # Map AST operator to IR UnaryOperator
        op_map = {
            "-": UnaryOperator.NEGATE,
            "!": UnaryOperator.NOT,
            "~": UnaryOperator.NOT, # Bitwise NOT often shared or different in IR
        }
        op = op_map.get(unary_data.op, UnaryOperator.NEGATE)
        
        # Create unary operation expression
        expr = IRExpression.UNARY(op=op, expr=operand_handle)
        
        return ctx.add_expression(expr)

    def _lower_call(self, ast_expr, ctx):
        """Lower function call."""
        from ...ir import Expression as IRExpression
        
        # ast_expr.data is a CallExpression
        call_data = ast_expr.data
        
        # Resolve function
        # For now, we assume it's just an identifier, but it could be complex
        function_handle = self._lower_expression(call_data.function, ctx)
        
        # Lower arguments
        arguments = [self._lower_expression(arg, ctx) for arg in call_data.arguments]
        
        # Create call expression
        # NAGA IR usually marks this as a result of a call statement or a specific call expression
        expr = IRExpression.CALL_RESULT(function=function_handle)
        
        # TODO: Handle parameter passing if NAGA IR requires it here
        return ctx.add_expression(expr)
    
    def _lower_member(self, ast_expr, ctx):
        """Lower member access."""
        from ...ir import Expression as IRExpression
        
        # ast_expr.data is a MemberExpression
        member_data = ast_expr.data
        base_handle = self._lower_expression(member_data.base, ctx)
        member_name = member_data.member.name
        
        # Determine if it's a vector swizzle or struct member
        # This is a bit simplified without full type info
        swizzle_map = {
            'x': 0, 'y': 1, 'z': 2, 'w': 3,
            'r': 0, 'g': 1, 'b': 2, 'a': 3,
        }
        
        if len(member_name) == 1 and member_name in swizzle_map:
            # Single component swizzle
            member_index = swizzle_map[member_name]
            expr = IRExpression.ACCESS_INDEX(base=base_handle, index=member_index)
        else:
            # Struct member or multi-component swizzle
            # TODO: Proper struct member lookup using type of base_handle
            member_index = 0
            expr = IRExpression.ACCESS_INDEX(base=base_handle, index=member_index)
        
        return ctx.add_expression(expr)

    
    def _lower_index(self, ast_expr, ctx):
        """Lower index access."""
        from ...ir import Expression as IRExpression
        
        # ast_expr.data is an IndexExpression
        index_data = ast_expr.data
        base_handle = self._lower_expression(index_data.base, ctx)
        index_handle = self._lower_expression(index_data.index, ctx)
        
        # Create access expression
        expr = IRExpression.ACCESS_INDEX(base=base_handle, index=index_handle)
        
        return ctx.add_expression(expr)
    
    def _lower_constructor(self, ast_expr, ctx):
        """Lower constructor expression."""
        from .construction import ConstructorHandler
        
        # ast_expr.data is a ConstructExpression
        construct_data = ast_expr.data
        
        handler = ConstructorHandler(self.module)
        constructor_type = self._lower_type(construct_data.ty)
        arguments = [self._lower_expression(arg, ctx) for arg in construct_data.arguments]
        
        return handler.handle_constructor(constructor_type, arguments, ctx)
    
    # Statement lowering methods
    def _lower_var_stmt(self, ast_stmt, ctx):
        """Lower variable declaration statement."""
        from ...ir import Statement as IRStatement, LocalVariable
        
        # ast_stmt.data is a VarDecl
        var_decl = ast_stmt.data
        var_name = var_decl.name.name
        var_type = self._lower_type(var_decl.type_) if var_decl.type_ else None
        
        initializer = None
        if var_decl.initializer:
            initializer = self._lower_expression(var_decl.initializer, ctx)
        
        # Create local variable
        local_var = LocalVariable(name=var_name, ty=var_type, init=initializer)
        
        # Add to function context and get handle
        handle = ctx.function.add_local_variable(local_var)
        ctx.local_table[var_name] = handle
        
        # Create statement
        stmt = IRStatement.LOCAL_VARIABLE(handle)
        ctx.add_statement(stmt)
    
    def _lower_assignment(self, ast_stmt, ctx):
        """Lower assignment statement."""
        from ...ir import Statement as IRStatement
        
        # ast_stmt.data is an AssignmentStatement
        assign_data = ast_stmt.data
        pointer = self._lower_expression(assign_data.lhs, ctx)
        value = self._lower_expression(assign_data.rhs, ctx)
        
        # Handle compound assignments (e.g., +=)
        if assign_data.op != "=":
            # Transform: a += b  =>  *a = *a + b
            # In IR, this usually involves a Load, a BinaryOp, and a Store
            pass
        
        # Create store statement
        stmt = IRStatement.STORE(pointer=pointer, value=value)
        ctx.add_statement(stmt)
    
    def _lower_if(self, ast_stmt, ctx):
        """Lower if statement."""
        from ...ir import Statement as IRStatement
        
        # ast_stmt.data is an IfStatement
        if_data = ast_stmt.data
        condition = self._lower_expression(if_data.condition, ctx)
        
        # Lower then block
        ctx.push_block()
        for stmt in if_data.accept:
            self._lower_statement(stmt, ctx)
        accept = ctx.pop_block()
        
        # Lower else block
        ctx.push_block()
        for stmt in if_data.reject:
            self._lower_statement(stmt, ctx)
        reject = ctx.pop_block()
        
        # Create if statement
        stmt = IRStatement.IF(condition=condition, accept=accept, reject=reject)
        ctx.add_statement(stmt)
    
    def _lower_switch(self, ast_stmt, ctx):
        """Lower switch statement."""
        from ...ir import Statement as IRStatement, SwitchCase
        
        # ast_stmt.data is a SwitchStatement
        switch_data = ast_stmt.data
        selector = self._lower_expression(switch_data.selector, ctx)
        
        ir_cases = []
        for case in switch_data.cases:
            # Lower case body
            ctx.push_block()
            for stmt in case["body"]:
                self._lower_statement(stmt, ctx)
            body = ctx.pop_block()
            
            # Map values
            if case["kind"] == "default":
                value = None # NAGA uses None for default
            else:
                # Resolve case value (must be constant)
                value = case["value"] # TODO: Evaluate to integer
                
            ir_cases.append(SwitchCase(value=value, body=body, fall_through=False))
        
        # Create switch statement
        stmt = IRStatement.SWITCH(selector=selector, cases=ir_cases)
        ctx.add_statement(stmt)
    
    def _lower_loop(self, ast_stmt, ctx):
        """Lower loop statement."""
        from ...ir import Statement as IRStatement
        
        # ast_stmt.data is a LoopStatement
        loop_data = ast_stmt.data
        
        # Lower loop body
        ctx.push_block()
        for stmt in loop_data.body:
            self._lower_statement(stmt, ctx)
        body = ctx.pop_block()
        
        # Create loop statement
        stmt = IRStatement.LOOP(body=body, continuing=None)
        ctx.add_statement(stmt)
    
    def _lower_while(self, ast_stmt, ctx):
        """Lower while loop."""
        # While is just a loop with a conditional break at the start
        # Handled during lowering if we want to transform it here
        pass
    
    def _lower_for(self, ast_stmt, ctx):
        """Lower for loop."""
        # For is just a loop with init, condition check, and update
        pass
    
    def _lower_break(self, ast_stmt, ctx):
        """Lower break statement."""
        from ...ir import Statement as IRStatement
        
        stmt = IRStatement.BREAK
        ctx.add_statement(stmt)
    
    def _lower_continue(self, ast_stmt, ctx):
        """Lower continue statement."""
        from ...ir import Statement as IRStatement
        
        stmt = IRStatement.CONTINUE
        ctx.add_statement(stmt)
    
    def _lower_return(self, ast_stmt, ctx):
        """Lower return statement."""
        from ...ir import Statement as IRStatement
        
        # ast_stmt.data is a ReturnStatement
        return_data = ast_stmt.data
        value = None
        if return_data.value:
            value = self._lower_expression(return_data.value, ctx)
        
        stmt = IRStatement.RETURN(value=value)
        ctx.add_statement(stmt)
    
    def _lower_discard(self, ast_stmt, ctx):
        """Lower discard statement."""
        from ...ir import Statement as IRStatement
        
        stmt = IRStatement.KILL
        ctx.add_statement(stmt)
    
    def _lower_call_stmt(self, ast_stmt, ctx):
        """Lower function call statement."""
        # Lower call expression and add as statement
        # ast_stmt.data is the call expression itself
        self._lower_expression(ast_stmt.data, ctx)
    
    def _lower_block(self, ast_stmt, ctx):
        """Lower block statement."""
        from ...ir import Statement as IRStatement
        
        # ast_stmt.data is a BlockStatement
        block_data = ast_stmt.data
        
        # Push new scope
        ctx.push_block()
        
        for stmt in block_data.statements:
            self._lower_statement(stmt, ctx)
        
        # Pop scope
        block = ctx.pop_block()
        
        # Create block statement
        stmt = IRStatement.BLOCK(block)
        ctx.add_statement(stmt)

    
    # Add all methods to class
    lowerer_class._lower_literal = _lower_literal
    lowerer_class._lower_identifier = _lower_identifier
    lowerer_class._lower_binary = _lower_binary
    lowerer_class._lower_unary = _lower_unary
    lowerer_class._lower_call = _lower_call
    lowerer_class._lower_member = _lower_member
    lowerer_class._lower_index = _lower_index
    lowerer_class._lower_constructor = _lower_constructor
    
    lowerer_class._lower_var_stmt = _lower_var_stmt
    lowerer_class._lower_assignment = _lower_assignment
    lowerer_class._lower_if = _lower_if
    lowerer_class._lower_switch = _lower_switch
    lowerer_class._lower_loop = _lower_loop
    lowerer_class._lower_while = _lower_while
    lowerer_class._lower_for = _lower_for
    lowerer_class._lower_break = _lower_break
    lowerer_class._lower_continue = _lower_continue
    lowerer_class._lower_return = _lower_return
    lowerer_class._lower_discard = _lower_discard
    lowerer_class._lower_call_stmt = _lower_call_stmt
    lowerer_class._lower_block = _lower_block
    
    return lowerer_class
