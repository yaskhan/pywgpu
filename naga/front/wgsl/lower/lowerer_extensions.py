"""
Additional lowering methods for WGSL lowerer.

This module contains helper methods for lowering expressions and statements.
"""


def add_lowering_methods(lowerer_class):
    """Add lowering helper methods to Lowerer class."""
    
    # Expression lowering methods
    def _lower_literal(self, ast_expr, ctx):
        """Lower literal expression."""
        from ....ir import Expression as IRExpression, ExpressionType, Literal
        
        # ast_expr.data is a LiteralExpression
        value = ast_expr.data.value
        
        if isinstance(value, bool):
            lit = Literal.bool_(value)
        elif isinstance(value, int):
            lit = Literal.i32(value)
        elif isinstance(value, float):
            lit = Literal.f32(value)
        else:
            from ..number import Number, NumberType
            if isinstance(value, Number):
                num_val = value.value
                num_type = value.type_
                
                if num_type == NumberType.I32:
                    lit = Literal.i32(int(num_val))
                elif num_type == NumberType.U32:
                    lit = Literal.u32(int(num_val))
                elif num_type == NumberType.F32:
                    lit = Literal.f32(float(num_val))
                elif num_type == NumberType.F16:
                    lit = Literal.f16(float(num_val))
                elif num_type == NumberType.ABSTRACT_INT:
                    lit = Literal.abstract_int(int(num_val))
                elif num_type == NumberType.ABSTRACT_FLOAT:
                    lit = Literal.abstract_float(float(num_val))
                else:
                    lit = Literal.f32(float(num_val))
            else:
                # Assume it's a parsed number object or something similar
                lit = Literal.f32(float(str(value)))
        
        expr = IRExpression(
            type=ExpressionType.LITERAL,
            literal=lit
        )
        
        return ctx.add_expression(expr)
    
    def _lower_identifier(self, ast_expr, ctx):
        """Lower identifier expression."""
        from ....ir import Expression as IRExpression, ExpressionType
        
        # ast_expr.data is an Ident
        ident_name = ast_expr.data.name
        
        # Resolve identifier to variable/constant/function
        # Check local variables first
        if ident_name in ctx.local_table:
            local_handle = ctx.local_table[ident_name]
            expr = IRExpression(
                type=ExpressionType.LOCAL_VARIABLE,
                local_variable=local_handle
            )
            return ctx.add_expression(expr)
        
        # Check global variables
        if ident_name in self.var_map:
            var_handle = self.var_map[ident_name]
            expr = IRExpression(
                type=ExpressionType.GLOBAL_VARIABLE,
                global_variable=var_handle
            )
            return ctx.add_expression(expr)
        
        # Check constants
        if ident_name in self.const_map:
            const_handle = self.const_map[ident_name]
            expr = IRExpression(
                type=ExpressionType.CONSTANT,
                constant=const_handle
            )
            return ctx.add_expression(expr)
        
        # Check functions
        if ident_name in self.function_map:
            func_handle = self.function_map[ident_name]
            return func_handle
        
        return None
    
    def _lower_binary(self, ast_expr, ctx):
        """Lower binary operation."""
        from ....ir import Expression as IRExpression, BinaryOperator, ExpressionType
        
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
            "|": BinaryOperator.INCLUSIVE_OR, # NAGA uses INCLUSIVE_OR
            "^": BinaryOperator.EXCLUSIVE_OR, # NAGA uses EXCLUSIVE_OR
            "<<": BinaryOperator.SHIFT_LEFT,
            ">>": BinaryOperator.SHIFT_RIGHT,
        }
        op = op_map.get(binary_data.op, BinaryOperator.ADD)
        
        expr = IRExpression(
            type=ExpressionType.BINARY,
            binary_left=left_handle,
            binary_op=op,
            binary_right=right_handle
        )
        
        return ctx.add_expression(expr)
    
    def _lower_unary(self, ast_expr, ctx):
        """Lower unary operation."""
        from ....ir import Expression as IRExpression, UnaryOperator, ExpressionType
        
        # ast_expr.data is a UnaryExpression
        unary_data = ast_expr.data
        operand_handle = self._lower_expression(unary_data.expr, ctx)
        
        # Map AST operator to IR UnaryOperator
        op_map = {
            "-": UnaryOperator.NEGATE,
            "!": UnaryOperator.LOGICAL_NOT,
            "~": UnaryOperator.BITWISE_NOT,
        }
        op = op_map.get(unary_data.op, UnaryOperator.NEGATE)
        
        expr = IRExpression(
            type=ExpressionType.UNARY,
            unary_op=op,
            unary_expr=operand_handle
        )
        
        return ctx.add_expression(expr)


    def _lower_call(self, ast_expr, ctx):
        """Lower function call."""
        from ....ir import Expression as IRExpression
        
        # ast_expr.data is a CallExpression
        call_data = ast_expr.data
        
        # Check for built-in functions if it's an identifier
        from ..ast import ExpressionKind
        if call_data.function.kind == ExpressionKind.IDENT:
            name = call_data.function.data.name
            
            # 1. Math Functions
            math_fn = self.builtin_resolver.resolve_math(name)
            if math_fn:
                args = [self._lower_expression(arg, ctx) for arg in call_data.arguments]
                expr = IRExpression(
                    type=None, # Will be set by validator or inferred
                    math_fun=math_fn,
                    math_arg=args[0] if len(args) > 0 else None,
                    math_arg1=args[1] if len(args) > 1 else None,
                    math_arg2=args[2] if len(args) > 2 else None,
                    math_arg3=args[3] if len(args) > 3 else None,
                )
                from ....ir import ExpressionType
                # We need to set the type field correctly
                object.__setattr__(expr, 'type', ExpressionType.MATH)
                return ctx.add_expression(expr)
            
            # 2. Relational Functions
            rel_fn = self.builtin_resolver.resolve_relational(name)
            if rel_fn:
                arg = self._lower_expression(call_data.arguments[0], ctx)
                expr = IRExpression(
                    type=None,
                    relational_fun=rel_fn,
                    relational_argument=arg
                )
                from ....ir import ExpressionType
                object.__setattr__(expr, 'type', ExpressionType.RELATIONAL)
                return ctx.add_expression(expr)
            
            # 3. Derivatives
            deriv = self.builtin_resolver.resolve_derivative(name)
            if deriv:
                axis, ctrl = deriv
                arg = self._lower_expression(call_data.arguments[0], ctx)
                expr = IRExpression(
                    type=None,
                    derivative_axis=axis,
                    derivative_ctrl=ctrl,
                    derivative_expr=arg
                )
                from ....ir import ExpressionType
                object.__setattr__(expr, 'type', ExpressionType.DERIVATIVE)
                return ctx.add_expression(expr)
                
            # 4. Selection
            if self.builtin_resolver.resolve_select(name):
                # select(reject, accept, condition)
                reject = self._lower_expression(call_data.arguments[0], ctx)
                accept = self._lower_expression(call_data.arguments[1], ctx)
                cond = self._lower_expression(call_data.arguments[2], ctx)
                expr = IRExpression(
                    type=None,
                    select_condition=cond,
                    select_accept=accept,
                    select_reject=reject
                )
                from ....ir import ExpressionType
                object.__setattr__(expr, 'type', ExpressionType.SELECT)
                return ctx.add_expression(expr)
                
            # 5. Texture Functions
            tex_fn = self.builtin_resolver.resolve_texture(name)
            if tex_fn:
                from ....ir import SampleLevel
                args = [self._lower_expression(arg, ctx) for arg in call_data.arguments]
                
                if tex_fn.startswith("sample"):
                    # textureSample(texture, sampler, coordinate, ...)
                    level = SampleLevel.auto()
                    array_index = None
                    offset = None
                    
                    if tex_fn == "sample_bias":
                        # textureSampleBias(texture, sampler, coordinate, bias, ...)
                        level = SampleLevel.bias(args[3])
                        if len(args) > 4: offset = args[4]
                    elif tex_fn == "sample_compare":
                        # textureSampleCompare(texture, sampler, coordinate, compare, ...)
                        level = SampleLevel.zero() # Comparison usually at level 0
                        # Comparison value is handled differently in NAGA?
                        # Actually image_sample_compare exists
                    elif tex_fn == "sample_level":
                        # textureSampleLevel(texture, sampler, coordinate, level, ...)
                        level = SampleLevel.exact(args[3])
                        if len(args) > 4: offset = args[4]
                    elif tex_fn == "sample_grad":
                        # textureSampleGrad(texture, sampler, coordinate, grad_x, grad_y, ...)
                        level = SampleLevel.gradient(args[3], args[4])
                        if len(args) > 5: offset = args[5]
                    
                    expr = IRExpression(
                        type=None,
                        image_sample_image=args[0],
                        image_sample_sampler=args[1],
                        image_sample_coordinate=args[2],
                        image_sample_level=level,
                        image_sample_offset=offset
                    )
                    from ....ir import ExpressionType
                    object.__setattr__(expr, 'type', ExpressionType.IMAGE_SAMPLE)
                    return ctx.add_expression(expr)
                    
                elif tex_fn == "load":
                    # textureLoad(texture, coordinate, ...)
                    level = None
                    sample = None
                    if len(args) > 2:
                        # Depends on texture type, could be level or sample
                        level = args[2] # Simplified
                        
                    expr = IRExpression(
                        type=None,
                        image_load_image=args[0],
                        image_load_coordinate=args[1],
                        image_load_level=level,
                        image_load_sample=sample
                    )
                    from ....ir import ExpressionType
                    object.__setattr__(expr, 'type', ExpressionType.IMAGE_LOAD)
                    return ctx.add_expression(expr)
                    
                elif tex_fn.startswith("query"):
                    # textureDimensions(texture, ...)
                    from ....ir import ImageQuery, ImageQueryType
                    query_type = ImageQueryType.SIZE
                    level = None
                    if tex_fn == "query_num_levels": query_type = ImageQueryType.NUM_LEVELS
                    elif tex_fn == "query_num_layers": query_type = ImageQueryType.NUM_LAYERS
                    elif tex_fn == "query_num_samples": query_type = ImageQueryType.NUM_SAMPLES
                    elif tex_fn == "query_size" and len(args) > 1:
                        level = args[1]
                    if query_type == ImageQueryType.SIZE:
                        query = ImageQuery.new_size(level)
                    elif query_type == ImageQueryType.NUM_LEVELS:
                        query = ImageQuery.new_num_levels()
                    elif query_type == ImageQueryType.NUM_LAYERS:
                        query = ImageQuery.new_num_layers()
                    elif query_type == ImageQueryType.NUM_SAMPLES:
                        query = ImageQuery.new_num_samples()
                    else:
                        query = ImageQuery(type=query_type)
                    
                    expr = IRExpression(
                        type=None,
                        image_query_image=args[0],
                        image_query_query=query
                    )
                    from ....ir import ExpressionType
                    object.__setattr__(expr, 'type', ExpressionType.IMAGE_QUERY)
                    return ctx.add_expression(expr)
                
                elif tex_fn == "store":
                    # textureStore(texture, coordinate, value)
                    from ....ir import Statement as IRStatement
                    stmt = IRStatement(
                        type=None, # Will be set in constructor or IRStatement type?
                        # Wait, Statement doesn't have an IMAGE_STORE factory yet?
                        # image_store_image=args[0], ...
                    )
                    # I'll use the raw constructor for now since I didn't add the factory
                    from ....ir import StatementType
                    stmt = IRStatement(
                        type=StatementType.IMAGE_STORE,
                        image_store_image=args[0],
                        image_store_coordinate=args[1],
                        image_store_value=args[2]
                    )
                    ctx.add_statement(stmt)
                    return None # textureStore returns void

            # 6. Atomic Functions
            atomic_fun = self.builtin_resolver.resolve_atomic(name)
            if atomic_fun:
                args = [self._lower_expression(arg, ctx) for arg in call_data.arguments]
                # atomicAdd(pointer, value)
                # 1. Create AtomicResult expression
                from ....ir import Expression as IRExpression, ExpressionType
                ptr_handle = args[0]
                ptr_inner = self._resolve_type(ptr_handle, ctx)
                
                from .conversion import TypeConverter
                converter = TypeConverter(self.module)
                result_ty_handle = 0
                
                from ....ir import TypeInnerType, TypeInner
                if ptr_inner.type == TypeInnerType.POINTER:
                    base_handle = ptr_inner.pointer.base
                    base_inner = self.module.types[base_handle].inner
                    if base_inner.type == TypeInnerType.ATOMIC:
                        scalar = base_inner.atomic
                        result_ty_handle = converter.get_handle_for_type_inner(TypeInner.new_scalar(scalar))
                    else:
                        result_ty_handle = base_handle
                
                result_expr = IRExpression(
                    type=ExpressionType.ATOMIC_RESULT,
                    atomic_result_ty=result_ty_handle
                )
                result_handle = ctx.add_expression(result_expr)
                
                # 2. Create Atomic statement
                from ....ir import Statement as IRStatement, StatementType
                stmt = IRStatement.new_atomic(
                    pointer=ptr_handle,
                    fun=atomic_fun,
                    value=args[1],
                    result=result_handle
                )
                ctx.add_statement(stmt)
                return result_handle

            # 7. arrayLength
            if self.builtin_resolver.resolve_array_length(name):
                arg = self._lower_expression(call_data.arguments[0], ctx)
                from ....ir import Expression as IRExpression, ExpressionType
                expr = IRExpression(
                    type=ExpressionType.ARRAY_LENGTH,
                    array_length=arg
                )
                return ctx.add_expression(expr)

        # Resolve function (fallback to user function)
        function_handle = self._lower_expression(call_data.function, ctx)
        
        # Lower arguments
        # arguments = [self._lower_expression(arg, ctx) for arg in call_data.arguments]
        
        # Create call expression
        expr = IRExpression(
            type=None,
            call_result=function_handle
        )
        from ....ir import ExpressionType
        object.__setattr__(expr, 'type', ExpressionType.CALL_RESULT)
        
        return ctx.add_expression(expr)

    
    def _lower_member(self, ast_expr, ctx):
        """Lower member access."""
        from ....ir import Expression as IRExpression, SwizzleComponent, VectorSize, TypeInnerType
        
        # ast_expr.data is a MemberExpression
        member_data = ast_expr.data
        base_handle = self._lower_expression(member_data.base, ctx)
        member_name = member_data.member.name
        
        # Resolve base type
        base_inner = self._resolve_type(base_handle, ctx)
        
        swizzle_map = {
            'x': SwizzleComponent.X, 'y': SwizzleComponent.Y, 'z': SwizzleComponent.Z, 'w': SwizzleComponent.W,
            'r': SwizzleComponent.X, 'g': SwizzleComponent.Y, 'b': SwizzleComponent.Z, 'a': SwizzleComponent.W,
        }
        
        # 1. Check if it's a swizzle (on vector)
        is_swizzle = all(c in swizzle_map for c in member_name)
        if is_swizzle and base_inner.type == TypeInnerType.VECTOR:
            if len(member_name) == 1:
                # Single component swizzle -> ACCESS_INDEX
                member_index = swizzle_map[member_name[0]].value
                expr = IRExpression(
                    type=None,
                    access_base=base_handle,
                    access_index_value=member_index
                )
                from ....ir import ExpressionType
                object.__setattr__(expr, 'type', ExpressionType.ACCESS_INDEX)
                return ctx.add_expression(expr)
            else:
                # Multi-component swizzle -> SWIZZLE
                pattern = [swizzle_map[c] for c in member_name]
                size_map = {2: VectorSize.BI, 3: VectorSize.TRI, 4: VectorSize.QUAD}
                expr = IRExpression(
                    type=None,
                    swizzle_size=size_map[len(member_name)],
                    swizzle_vector=base_handle,
                    swizzle_pattern=pattern
                )
                from ....ir import ExpressionType
                object.__setattr__(expr, 'type', ExpressionType.SWIZZLE)
                return ctx.add_expression(expr)

        # 2. Check if it's a struct member
        if base_inner.type == TypeInnerType.STRUCT:
            for i, member in enumerate(base_inner.struct.members):
                if member.name == member_name:
                    expr = IRExpression(
                        type=None,
                        access_base=base_handle,
                        access_index_value=i
                    )
                    from ....ir import ExpressionType
                    object.__setattr__(expr, 'type', ExpressionType.ACCESS_INDEX)
                    return ctx.add_expression(expr)

        # Fallback (e.g. error case or unresolved type)
        return base_handle


    
    def _lower_index(self, ast_expr, ctx):
        """Lower index access."""
        from ....ir import Expression as IRExpression, ExpressionType, LiteralType
        
        # ast_expr.data is an IndexExpression
        index_data = ast_expr.data
        base_handle = self._lower_expression(index_data.base, ctx)
        index_handle = self._lower_expression(index_data.index, ctx)
        
        # Check if index is a constant
        index_expr = ctx.function.expressions[index_handle]
        
        if index_expr.type == ExpressionType.LITERAL:
            lit = index_expr.literal
            if lit.type in (LiteralType.I32, LiteralType.U32, LiteralType.ABSTRACT_INT):
                expr = IRExpression(
                    type=ExpressionType.ACCESS_INDEX,
                    access_base=base_handle,
                    access_index_value=int(lit.value)
                )
                return ctx.add_expression(expr)
        
        # Dynamic index
        expr = IRExpression(
            type=ExpressionType.ACCESS,
            access_base=base_handle,
            access_index=index_handle
        )
        return ctx.add_expression(expr)

    
    def _lower_constructor(self, ast_expr, ctx):
        """Lower constructor expression."""
        from .construction import ConstructorHandler
        
        # ast_expr.data is a ConstructExpression
        construct_data = ast_expr.data
        
        # Check if the "type" is actually a function call (mis-parsed as constructor)
        from ..ast import ExpressionKind
        # Simple types/functions can be dicts with 'name' (from expression_parser.py)
        if isinstance(construct_data.ty, dict) and "name" in construct_data.ty:
            name = construct_data.ty["name"]
            if (self.builtin_resolver.resolve_math(name) or 
                self.builtin_resolver.resolve_relational(name) or
                self.builtin_resolver.resolve_atomic(name) or
                self.builtin_resolver.resolve_texture(name) or
                self.builtin_resolver.resolve_array_length(name) or
                self.builtin_resolver.resolve_select(name)):
                # Delegate to _lower_call
                from ..ast import CallExpression, Expression as ASTExpression, ExpressionKind
                # Create a temporary IDENT expression for the function name
                from ..ast import Ident
                func_expr = ASTExpression(
                    kind=ExpressionKind.IDENT,
                    data=Ident(name=name, span=construct_data.ty.get("span", (0, 0))),
                    span=construct_data.ty.get("span", (0, 0))
                )
                call_expr = CallExpression(
                    function=func_expr,
                    arguments=construct_data.arguments
                )
                wrapped_call = ASTExpression(
                    kind=ExpressionKind.CALL,
                    data=call_expr,
                    span=ast_expr.span
                )
                return self._lower_call(wrapped_call, ctx)
                
        # We use a new handler instance or a shared one from self
        handler = ConstructorHandler(self)
        constructor_type = self._lower_type(construct_data.ty)
        arguments = [self._lower_expression(arg, ctx) for arg in construct_data.arguments]
        
        return handler.handle_constructor(constructor_type, arguments, ctx)

    
    # Statement lowering methods
    def _lower_var_stmt(self, ast_stmt, ctx):
        """Lower variable declaration statement."""
        from ....ir import Statement as IRStatement, LocalVariable, StatementType
        
        # ast_stmt.data is a VarDecl
        var_decl = ast_stmt.data
        var_name = var_decl.name.name
        var_type = self._lower_type(var_decl.type_) if var_decl.type_ else None
        
        initializer = None
        if var_decl.initializer:
            initializer = self._lower_expression(var_decl.initializer, ctx)
            
        # Type inference for let/var without explicit type
        if var_type is None and initializer is not None:
            inner = self._resolve_type(initializer, ctx)
            from .conversion import TypeConverter
            converter = TypeConverter(self.module)
            var_type = converter.get_handle_for_type_inner(inner)
        
        # Add to function context and get handle
        handle = ctx.function.add_local_var(var_name, var_type, initializer)
        ctx.local_table[var_name] = handle
        
        # Local variables are added to the function arena, no statement needed in IR
        # unless we need an explicit EMIT for something, but typically not for declarations.
        pass
    
    def _lower_assignment(self, ast_stmt, ctx):
        """Lower assignment statement."""
        from ....ir import Statement as IRStatement, StatementType
        
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
        stmt = IRStatement.new_store(pointer=pointer, value=value)
        ctx.add_statement(stmt)
    
    def _lower_if(self, ast_stmt, ctx):
        """Lower if statement."""
        from ....ir import Statement as IRStatement
        
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
        stmt = IRStatement.new_if(condition=condition, accept=accept, reject=reject)
        ctx.add_statement(stmt)
    
    def _lower_switch(self, ast_stmt, ctx):
        """Lower switch statement."""
        from ....ir import Statement as IRStatement, SwitchCase
        
        # ast_stmt.data is a SwitchStatement
        switch_data = ast_stmt.data
        selector = self._lower_expression(switch_data.selector, ctx)
        
        ir_cases = []
        default_block = None
        for case in switch_data.cases:
            # Lower case body
            ctx.push_block()
            for stmt in case["body"]:
                self._lower_statement(stmt, ctx)
            body = ctx.pop_block()
            
            # Map values
            if case["kind"] == "default":
                default_block = body
            else:
                # Resolve case value (must be constant)
                from ..ast import ExpressionKind
                ast_val = case["value"]
                value = None
                if ast_val.kind == ExpressionKind.LITERAL:
                    value = int(ast_val.data.value)
                
                ir_cases.append(SwitchCase(value=value, body=body, fall_through=False))
        
        # Create switch statement
        stmt = IRStatement.new_switch(selector=selector, cases=ir_cases, default=default_block)
        ctx.add_statement(stmt)
    
    def _lower_loop(self, ast_stmt, ctx):
        """Lower loop statement."""
        from ....ir import Statement as IRStatement
        
        # ast_stmt.data is a LoopStatement
        loop_data = ast_stmt.data
        
        # Lower loop body
        ctx.push_block()
        for stmt in loop_data.body:
            self._lower_statement(stmt, ctx)
        body = ctx.pop_block()
        
        # Create loop statement
        stmt = IRStatement(type=StatementType.LOOP, loop_body=body, loop_continuing=None)
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
        from ....ir import Statement as IRStatement
        
        stmt = IRStatement.new_break()
        ctx.add_statement(stmt)
    
    def _lower_continue(self, ast_stmt, ctx):
        """Lower continue statement."""
        from ....ir import Statement as IRStatement
        
        stmt = IRStatement.new_continue()
        ctx.add_statement(stmt)
    
    def _lower_return(self, ast_stmt, ctx):
        """Lower return statement."""
        from ....ir import Statement as IRStatement
        
        # ast_stmt.data is a ReturnStatement
        return_data = ast_stmt.data
        value = None
        if return_data.value:
            value = self._lower_expression(return_data.value, ctx)
        
        stmt = IRStatement.new_return(value=value)
        ctx.add_statement(stmt)
    
    def _lower_discard(self, ast_stmt, ctx):
        """Lower discard statement."""
        from ....ir import Statement as IRStatement
        
        stmt = IRStatement.new_kill()
        ctx.add_statement(stmt)
    
    def _lower_call_stmt(self, ast_stmt, ctx):
        """Lower function call statement."""
        # Lower call expression and add as statement
        # ast_stmt.data is the call expression itself
        self._lower_expression(ast_stmt.data, ctx)
    
    def _lower_block(self, ast_stmt, ctx):
        """Lower block statement."""
        from ....ir import Statement as IRStatement
        
        # ast_stmt.data is a BlockStatement
        block_data = ast_stmt.data
        
        # Push new scope
        ctx.push_block()
        
        for stmt in block_data.statements:
            self._lower_statement(stmt, ctx)
        
        # Pop scope
        block = ctx.pop_block()
        
        # Create block statement
        stmt = IRStatement.new_block(block)
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
