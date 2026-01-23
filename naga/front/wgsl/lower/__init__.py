"""
WGSL AST to NAGA IR lowering.

Translated from wgpu-trunk/naga/src/front/wgsl/lower/mod.rs

This module converts the WGSL AST into NAGA's intermediate representation.
"""

from typing import Any, Dict, List, Optional
from ....ir import Module, Function, Expression, Statement, Type
from ..ast import TranslationUnit, GlobalDecl
from ..index import Index
from .lowerer_extensions import add_lowering_methods
from .type_resolver import add_type_resolver_methods


@add_type_resolver_methods
@add_lowering_methods
class Lowerer:
    """
    Lowers WGSL AST to NAGA IR.
    
    This is the final stage of WGSL parsing that converts the analyzed
    AST into executable IR.
    """
    
    def __init__(self, index: Index):
        """
        Initialize the lowerer with a semantic index.
        
        Args:
            index: Semantic index with dependency ordering
        """
        self.index = index
        self.module: Optional[Module] = None
        
        # Lookup tables for resolved declarations
        self.type_map: Dict[Any, Any] = {}
        self.const_map: Dict[Any, Any] = {}
        self.var_map: Dict[Any, Any] = {}
        self.function_map: Dict[Any, Any] = {}
        
        from .builtins import BuiltInResolver
        self.builtin_resolver = BuiltInResolver()

    
    def lower(self, tu: TranslationUnit) -> Module:
        """
        Lower a translation unit to a NAGA module.
        
        Args:
            tu: Translation unit (parsed AST)
            
        Returns:
            Complete NAGA IR Module
            
        Raises:
            ParseError: If lowering fails
        """
        self.module = Module()
        
        # Process global declarations in dependency order
        for decl_handle in self.index.visit_ordered():
            decl = tu.decls[decl_handle]
            self._lower_global_decl(decl)
        
        return self.module
    
    def _lower_global_decl(self, decl: GlobalDecl) -> None:
        """
        Lower a global declaration.
        
        Args:
            decl: Global declaration to lower
        """
        from ..ast import (
            FunctionDecl, VarDecl, ConstDecl, OverrideDecl,
            StructDecl, TypeAlias, ConstAssert
        )
        
        # Match on declaration kind
        if isinstance(decl.kind, StructDecl):
            # Lower struct type definition
            struct_type = self._lower_struct(decl.kind)
            self.type_map[decl.kind.name.name] = struct_type
        
        elif isinstance(decl.kind, TypeAlias):
            # Lower type alias
            aliased_type = self._lower_type(decl.kind.type_)
            self.type_map[decl.kind.name.name] = aliased_type
        
        elif isinstance(decl.kind, ConstDecl):
            # Lower constant declaration
            const_handle = self._lower_const(decl.kind)
            self.const_map[decl.kind.name.name] = const_handle
        
        elif isinstance(decl.kind, OverrideDecl):
            # Lower pipeline-overridable constant
            override_handle = self._lower_override(decl.kind)
            # Store in module.overrides
            if self.module:
                # TODO: Add to module.overrides when IR supports it
                pass
        
        elif isinstance(decl.kind, VarDecl):
            # Lower global variable
            var_handle = self._lower_global_var(decl.kind)
            self.var_map[decl.kind.name.name] = var_handle
        
        elif isinstance(decl.kind, FunctionDecl):
            # Lower function declaration
            func_handle = self._lower_function(decl.kind)
            self.function_map[decl.kind.name.name] = func_handle
        
        elif isinstance(decl.kind, ConstAssert):
            # Evaluate and verify const assertion
            self._lower_const_assert(decl.kind)
        
        else:
            # Unknown declaration type
            pass
    
    def _lower_type(self, ast_type: Any) -> Any:
        """
        Lower an AST type to IR type.
        
        Args:
            ast_type: AST type
            
        Returns:
            IR type handle
        """
        from .conversion import TypeConverter
        
        if ast_type is None:
            return None
        
        converter = TypeConverter(self.module)
        type_handle, type_inner = converter.convert_type(ast_type)
        return type_handle
    
    def _lower_expression(self, ast_expr: Any, ctx: Any) -> Any:
        """
        Lower an AST expression to IR expression.
        
        Args:
            ast_expr: AST expression
            ctx: Expression context
            
        Returns:
            IR expression handle
        """
        from ..ast import Expression, ExpressionKind
        
        if ast_expr is None:
            return None
        
        # Match on expression kind
        if ast_expr.kind == ExpressionKind.LITERAL:
            # Lower literal expression
            return self._lower_literal(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.IDENT:
            # Lower identifier (resolve to variable/constant)
            return self._lower_identifier(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.BINARY:
            # Lower binary operation
            return self._lower_binary(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.UNARY:
            # Lower unary operation
            return self._lower_unary(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.CALL:
            # Lower function call
            return self._lower_call(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.MEMBER:
            # Lower member access
            return self._lower_member(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.INDEX:
            # Lower index access
            return self._lower_index(ast_expr, ctx)
        
        elif ast_expr.kind == ExpressionKind.CONSTRUCT:
            # Lower constructor
            return self._lower_constructor(ast_expr, ctx)
        
        else:
            # Unknown expression kind
            return None
    
    def _lower_statement(self, ast_stmt: Any, ctx: Any) -> None:
        """
        Lower an AST statement to IR statement.
        
        Args:
            ast_stmt: AST statement
            ctx: Statement context
        """
        from ..ast import Statement, StatementKind
        
        if ast_stmt is None:
            return
        
        # Match on statement kind
        if ast_stmt.kind == StatementKind.VAR_DECL:
            # Lower variable declaration
            self._lower_var_stmt(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.ASSIGNMENT:
            # Lower assignment
            self._lower_assignment(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.IF:
            # Lower if statement
            self._lower_if(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.SWITCH:
            # Lower switch statement
            self._lower_switch(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.LOOP:
            # Lower loop
            self._lower_loop(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.WHILE:
            # Lower while loop
            self._lower_while(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.FOR:
            # Lower for loop
            self._lower_for(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.BREAK:
            # Lower break
            self._lower_break(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.CONTINUE:
            # Lower continue
            self._lower_continue(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.RETURN:
            # Lower return
            self._lower_return(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.DISCARD:
            # Lower discard
            self._lower_discard(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.CALL:
            # Lower function call statement
            self._lower_call_stmt(ast_stmt, ctx)
        
        elif ast_stmt.kind == StatementKind.BLOCK:
            # Lower block
            self._lower_block(ast_stmt, ctx)
        
        else:
            # Unknown statement kind
            pass
    
    def _lower_function(self, ast_func: Any) -> Any:
        """
        Lower an AST function to IR function.
        
        Args:
            ast_func: AST function declaration
            
        Returns:
            IR function handle
        """
        from ....ir import Function, Expression, Statement, FunctionArgument, TypeInner
        from .context import StatementContext
        
        from ....ir import Function, Block
        # Create new function
        func = Function(name=ast_func.name.name, result=None, body=Block.new())
        
        # Process function parameters
        for i, param in enumerate(ast_func.parameters):
            param_type = self._lower_type(param['type'])
            # Create function argument
            arg = func.add_argument(name=param['name'].name, ty=param_type, binding=None)
            # Store in local map if we have a context (but we populate local_table later)
        
        # Process return type
        if ast_func.return_type:
            from ....ir import FunctionResult
            func.result = FunctionResult(ty=self._lower_type(ast_func.return_type), binding=None)
        
        # Detect entry point
        for attr in ast_func.attributes:
            if attr.name in ('vertex', 'fragment', 'compute'):
                # In NAGA IR, we might wrap this in an EntryPoint object
                # For now, just mark the stage
                pass
        
        # Lower function body
        ctx = StatementContext(func)
        # Add arguments to local table
        for i, arg in enumerate(func.arguments):
            ctx.local_table[arg.name] = i # Simplified handle mapping
            
        for stmt in ast_func.body:
            self._lower_statement(stmt, ctx)
        
        # Finalize function body
        func.body.append_block(ctx.block_stack[0])
        
        # Add to module
        if self.module:
            self.module.functions.append(func)
            handle = len(self.module.functions) - 1
            return handle
        
        return func
    
    def _lower_struct(self, ast_struct: Any) -> Any:
        """Lower struct type definition."""
        from .conversion import TypeConverter
        from ....ir import StructMember
        
        converter = TypeConverter(self.module)
        members = []
        for ast_member in ast_struct.members:
            inner_ty = self._lower_type(ast_member.type_)
            members.append(StructMember(
                name=ast_member.name.name,
                ty=inner_ty,
                offset=0 # Layout will be handled later
            ))
            
        return converter.convert_struct(members)
    
    def _lower_const(self, ast_const: Any) -> Any:
        """Lower constant declaration."""
        from ....ir import Constant
        # TODO: Evaluate constant expression
        # For now, just create a placeholder constant
        const = Constant(name=ast_const.name.name, ty=self._lower_type(ast_const.type_), init=None)
        if self.module:
            self.module.constants.append(const)
            return len(self.module.constants) - 1
        return None
    
    def _lower_override(self, ast_override: Any) -> Any:
        """Lower pipeline-overridable constant."""
        # TODO: Create override in NAGA IR (OverridableConstant)
        return None
    
    def _lower_global_var(self, ast_var: Any) -> Any:
        """Lower global variable."""
        from ....ir import GlobalVariable, AddressSpace, StorageAccess
        
        # Determine address space and access mode
        space = AddressSpace.PRIVATE
        if ast_var.address_space:
            from .conversion import resolve_address_space
            space = resolve_address_space(ast_var.address_space)
            
        access = StorageAccess.LOAD | StorageAccess.STORE
        if ast_var.access_mode:
            from .conversion import resolve_storage_access
            access = resolve_storage_access(ast_var.access_mode)
            
        # Process bindings
        group = None
        binding = None
        for attr in ast_var.attributes:
            if attr.name == 'group':
                group = int(attr.arguments[0])
            elif attr.name == 'binding':
                binding = int(attr.arguments[0])
        
        # Validation: Check for required bindings
        if space in (AddressSpace.UNIFORM, AddressSpace.STORAGE, AddressSpace.HANDLE):
            if group is None or binding is None:
                from ..error import ParseError
                raise ParseError(
                    message=f"resource variable in address space {space.value} must have @group and @binding attributes",
                    labels=[(ast_var.name.span[0], ast_var.name.span[1], "")],
                    notes=[]
                )
        
        # Lower initializer
        init = None
        if ast_var.initializer:
            # For global constants/vars, we need a constant expression context
            from .context import ExpressionContext
            ctx = ExpressionContext(self.module)
            init = self._lower_expression(ast_var.initializer, ctx)
                
        var = GlobalVariable(
            name=ast_var.name.name,
            space=space,
            binding={'group': group, 'binding': binding} if binding is not None else None,
            ty=self._lower_type(ast_var.type_),
            init=init
        )
        
        if self.module:
            self.module.global_variables.append(var)
            return len(self.module.global_variables) - 1
            
        return None
    
    def _lower_const_assert(self, ast_assert: Any) -> None:
        """Evaluate and verify const assertion."""
        # In a real implementation, we'd evaluate the condition
        pass

