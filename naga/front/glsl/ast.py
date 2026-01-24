"""
GLSL AST (Abstract Syntax Tree) node definitions.

This module provides AST node definitions for GLSL syntax tree representation,
including expressions, statements, declarations, and types.
"""

from typing import Any, Optional, List, Dict, Union
from enum import Enum
from dataclasses import dataclass
from typing import Protocol


class GlobalLookupKind(Enum):
    """Types of global lookup operations."""
    VARIABLE = "variable"
    CONSTANT = "constant"
    OVERRIDE = "override"
    BLOCK_SELECT = "block_select"


@dataclass
class GlobalLookup:
    """Information about global symbol lookup."""
    kind: GlobalLookupKind
    entry_arg: Optional[int]
    mutable: bool


@dataclass
class ParameterInfo:
    """Information about function parameters."""
    qualifier: 'ParameterQualifier'
    depth: bool


class FunctionKind(Enum):
    """Types of function implementations."""
    CALL = "call"
    MACRO = "macro"


@dataclass
class Overload:
    """Function overload information."""
    parameters: List[Any]
    parameters_info: List[ParameterInfo]
    kind: FunctionKind
    defined: bool
    internal: bool
    void: bool


class BuiltinVariations(Enum):
    """Builtin function variation flags."""
    STANDARD = 1 << 0
    DOUBLE = 1 << 1
    CUBE_TEXTURES_ARRAY = 1 << 2
    D2_MULTI_TEXTURES_ARRAY = 1 << 3


@dataclass
class FunctionDeclaration:
    """Function declaration information."""
    overloads: List[Overload]
    variations: BuiltinVariations


@dataclass
class EntryArg:
    """Entry point argument information."""
    name: Optional[str]
    binding: Any
    handle: Any
    storage: 'StorageQualifier'


@dataclass
class VariableReference:
    """Variable reference information."""
    expr: Any
    load: bool
    mutable: bool
    constant: Optional[tuple[Any, Any]]
    entry_arg: Optional[int]


class HirExprKind(Enum):
    """High-level IR expression types."""
    SEQUENCE = "sequence"
    ACCESS = "access"
    SELECT = "select"
    LITERAL = "literal"
    BINARY = "binary"
    UNARY = "unary"
    VARIABLE = "variable"
    CALL = "call"
    CONDITIONAL = "conditional"
    ASSIGN = "assign"
    PREPOSTFIX = "prepostfix"
    METHOD = "method"


@dataclass
class HirExpr:
    """High-level IR expression."""
    kind: HirExprKind
    meta: Any


@dataclass
class FunctionCall:
    """Function call information."""
    kind: 'FunctionCallKind'
    args: List[HirExpr]


class StorageQualifier(Enum):
    """Storage qualifier types."""
    ADDRESS_SPACE = "address_space"
    INPUT = "input"
    OUTPUT = "output"
    CONST = "const"


class StructLayout(Enum):
    """Struct layout qualifiers."""
    STD140 = "std140"
    STD430 = "std430"


# A precision hint used in GLSL declarations.
#
# Precision hints can be used to either speed up shader execution or control
# the precision of arithmetic operations.
#
# To use a precision hint simply add it before the type in the declaration.
# ```glsl
# mediump float a;
# ```
#
# The default when no precision is declared is `highp` which means that all
# operations operate with the type defined width.
#
# For `mediump` and `lowp` operations follow the spir-v
# [`RelaxedPrecision`][RelaxedPrecision] decoration semantics.
#
# [RelaxedPrecision]: https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#_a_id_relaxedprecisionsection_a_relaxed_precision
class Precision(Enum):
    """
    Precision qualifiers for types and variables.
    
    Precision hints control the precision of arithmetic operations and can be
    used to optimize shader execution. They correspond to SPIR-V RelaxedPrecision
    decoration.
    """
    LOW = "lowp"
    MEDIUM = "mediump"
    HIGH = "highp"


class ParameterQualifier(Enum):
    """Parameter direction qualifiers."""
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    CONST = "const"


class Profile(Enum):
    """GLSL profile types."""
    CORE = "core"


# Additional AST node classes for complete GLSL representation
class ExprNode:
    """Base class for expression nodes."""
    def __init__(self):
        self.meta = None


class BinaryExpr(ExprNode):
    """Binary expression node."""
    def __init__(self, left: ExprNode, op: str, right: ExprNode):
        super().__init__()
        self.left = left
        self.op = op
        self.right = right


class UnaryExpr(ExprNode):
    """Unary expression node."""
    def __init__(self, op: str, expr: ExprNode):
        super().__init__()
        self.op = op
        self.expr = expr


class LiteralExpr(ExprNode):
    """Literal expression node."""
    def __init__(self, value: Any, type_name: str):
        super().__init__()
        self.value = value
        self.type_name = type_name


class VariableExpr(ExprNode):
    """Variable reference expression node."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class CallExpr(ExprNode):
    """Function call expression node."""
    def __init__(self, name: str, args: List[ExprNode]):
        super().__init__()
        self.name = name
        self.args = args


class SelectExpr(ExprNode):
    """Field selection expression node."""
    def __init__(self, base: ExprNode, field: str):
        super().__init__()
        self.base = base
        self.field = field


class IndexExpr(ExprNode):
    """Array indexing expression node."""
    def __init__(self, base: ExprNode, index: ExprNode):
        super().__init__()
        self.base = base
        self.index = index


class ConditionalExpr(ExprNode):
    """Ternary conditional expression node."""
    def __init__(self, condition: ExprNode, accept: ExprNode, reject: ExprNode):
        super().__init__()
        self.condition = condition
        self.accept = accept
        self.reject = reject


class AssignExpr(ExprNode):
    """Assignment expression node."""
    def __init__(self, target: ExprNode, value: ExprNode):
        super().__init__()
        self.target = target
        self.value = value


class PrePostfixExpr(ExprNode):
    """Prefix/postfix expression node."""
    def __init__(self, op: str, expr: ExprNode, postfix: bool):
        super().__init__()
        self.op = op
        self.expr = expr
        self.postfix = postfix


class MethodExpr(ExprNode):
    """Method call expression node."""
    def __init__(self, expr: ExprNode, name: str, args: List[ExprNode]):
        super().__init__()
        self.expr = expr
        self.name = name
        self.args = args


class SequenceExpr(ExprNode):
    """Sequence of expressions node."""
    def __init__(self, exprs: List[ExprNode]):
        super().__init__()
        self.exprs = exprs


class DeclNode:
    """Base class for declaration nodes."""
    def __init__(self):
        self.meta = None


class VarDecl(DeclNode):
    """Variable declaration node."""
    def __init__(self, type_spec: Any, name: str, init: Optional[ExprNode] = None, precision: Optional[Precision] = None):
        super().__init__()
        self.type_spec = type_spec
        self.name = name
        self.init = init
        self.precision = precision


class FuncDecl(DeclNode):
    """Function declaration node."""
    def __init__(self, return_type: Any, name: str, params: List[Any], body: Optional['BlockStmt'] = None, precision: Optional[Precision] = None):
        super().__init__()
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body
        self.precision = precision


class StructDecl(DeclNode):
    """Struct declaration node."""
    def __init__(self, name: str, members: List[Any]):
        super().__init__()
        self.name = name
        self.members = members


class StmtNode:
    """Base class for statement nodes."""
    def __init__(self):
        self.meta = None


class BlockStmt(StmtNode):
    """Block statement node."""
    def __init__(self, stmts: List[StmtNode]):
        super().__init__()
        self.stmts = stmts


class ExprStmt(StmtNode):
    """Expression statement node."""
    def __init__(self, expr: ExprNode):
        super().__init__()
        self.expr = expr


class DeclStmt(StmtNode):
    """Declaration statement node."""
    def __init__(self, decl: DeclNode):
        super().__init__()
        self.decl = decl


class ReturnStmt(StmtNode):
    """Return statement node."""
    def __init__(self, expr: Optional[ExprNode] = None):
        super().__init__()
        self.expr = expr


class IfStmt(StmtNode):
    """If statement node."""
    def __init__(self, condition: ExprNode, then_stmt: StmtNode, else_stmt: Optional[StmtNode] = None):
        super().__init__()
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt


class WhileStmt(StmtNode):
    """While loop statement node."""
    def __init__(self, condition: ExprNode, body: StmtNode):
        super().__init__()
        self.condition = condition
        self.body = body


class ForStmt(StmtNode):
    """For loop statement node."""
    def __init__(self, init: Optional[DeclNode], condition: Optional[ExprNode], 
                 iter: Optional[ExprNode], body: StmtNode):
        super().__init__()
        self.init = init
        self.condition = condition
        self.iter = iter
        self.body = body