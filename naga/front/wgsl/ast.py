"""
WGSL AST (Abstract Syntax Tree) definitions.

Translated from wgpu-trunk/naga/src/front/wgsl/parse/ast.rs

This module defines the AST structure for WGSL source code.
"""

from typing import Any, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


@dataclass
class Ident:
    """Identifier with source location."""
    name: str
    span: tuple[int, int]


@dataclass
class Dependency:
    """A reference from one declaration to another."""
    ident: str
    usage: tuple[int, int]


class GlobalDeclKind(Enum):
    """Kind of global declaration."""
    FN = "fn"
    VAR = "var"
    CONST = "const"
    OVERRIDE = "override"
    STRUCT = "struct"
    TYPE = "type"
    CONST_ASSERT = "const_assert"


@dataclass
class GlobalDecl:
    """
    A global declaration in the translation unit.
    
    Attributes:
        kind: The kind and data of this declaration
        dependencies: Other declarations this one depends on
    """
    kind: Any  # Specific type depends on GlobalDeclKind
    dependencies: List[Dependency]


@dataclass
class TranslationUnit:
    """
    Top-level WGSL translation unit.
    
    Represents a complete WGSL shader module.
    
    Attributes:
        decls: All global declarations in source order
        directives: Preprocessor-like directives (enable extensions, etc.)
    """
    decls: List[GlobalDecl]
    directives: List[Any]


@dataclass
class FunctionDecl:
    """
    Function declaration.
    
    Attributes:
        name: Function name
        parameters: Function parameters
        return_type: Return type annotation
        body: Function body statements
        attributes: Function attributes (@vertex, @fragment, etc.)
    """
    name: Ident
    parameters: List[Any]
    return_type: Optional[Any]
    body: List[Any]
    attributes: List[Any]


@dataclass
class VarDecl:
    """
    Variable declaration (global or local).
    
    Attributes:
        name: Variable name
        type_: Type annotation
        address_space: Address space (function, private, workgroup, etc.)
        access_mode: Access mode (read, write, read_write)
        initializer: Initial value expression
        attributes: Variable attributes (@group, @binding, etc.)
    """
    name: Ident
    type_: Optional[Any]
    address_space: Optional[str]
    access_mode: Optional[str]
    initializer: Optional[Any]
    attributes: List[Any]


@dataclass
class ConstDecl:
    """
    Constant declaration.
    
    Attributes:
        name: Constant name
        type_: Type annotation
        initializer: Constant value expression
    """
    name: Ident
    type_: Optional[Any]
    initializer: Any


@dataclass
class OverrideDecl:
    """
    Pipeline-overridable constant declaration.
    
    Attributes:
        name: Override name
        type_: Type annotation
        initializer: Default value expression
        id: Optional override ID
    """
    name: Ident
    type_: Optional[Any]
    initializer: Optional[Any]
    id: Optional[int]


@dataclass
class StructDecl:
    """
    Struct type declaration.
    
    Attributes:
        name: Struct name
        members: Struct members
    """
    name: Ident
    members: List['StructMember']


@dataclass
class StructMember:
    """
    Member of a struct.
    
    Attributes:
        name: Member name
        type_: Member type
        attributes: Member attributes (@location, @builtin, etc.)
    """
    name: Ident
    type_: Any
    attributes: List[Any]


@dataclass
class TypeAlias:
    """
    Type alias declaration.
    
    Attributes:
        name: Alias name
        type_: Aliased type
    """
    name: Ident
    type_: Any


@dataclass
class ConstAssert:
    """
    Const assertion (compile-time check).
    
    Attributes:
        condition: Boolean expression that must be true
    """
    condition: Any


# Expression kinds data
@dataclass
class BinaryExpression:
    left: 'Expression'
    op: str
    right: 'Expression'

@dataclass
class UnaryExpression:
    op: str
    expr: 'Expression'

@dataclass
class CallExpression:
    function: 'Expression'
    arguments: List['Expression']

@dataclass
class IndexExpression:
    base: 'Expression'
    index: 'Expression'

@dataclass
class MemberExpression:
    base: 'Expression'
    member: Ident

@dataclass
class ConstructExpression:
    ty: Any # Type AST node
    arguments: List['Expression']

@dataclass
class LiteralExpression:
    value: Any # int, float, bool, or parsed number object

class ExpressionKind(Enum):
    """Kind of expression."""
    LITERAL = "literal"
    IDENT = "ident"
    BINARY = "binary"
    UNARY = "unary"
    CALL = "call"
    INDEX = "index"
    MEMBER = "member"
    CONSTRUCT = "construct"


@dataclass
class Expression:
    """
    Expression in the AST.
    
    Attributes:
        kind: Expression kind (Enum)
        data: The actual data for this kind of expression
        span: Source location
    """
    kind: ExpressionKind
    data: Any
    span: tuple[int, int]


# Statement kinds data
@dataclass
class IfStatement:
    condition: Expression
    accept: List['Statement']
    reject: List['Statement']

@dataclass
class SwitchStatement:
    selector: Expression
    cases: List[Any] # Case data

@dataclass
class LoopStatement:
    body: List['Statement']

@dataclass
class ReturnStatement:
    value: Optional[Expression]

@dataclass
class AssignmentStatement:
    lhs: Expression
    op: str
    rhs: Expression

@dataclass
class BlockStatement:
    statements: List['Statement']

class StatementKind(Enum):
    """Kind of statement."""
    BLOCK = "block"
    IF = "if"
    SWITCH = "switch"
    LOOP = "loop"
    WHILE = "while"
    FOR = "for"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    DISCARD = "discard"
    ASSIGNMENT = "assignment"
    CALL = "call"
    VAR_DECL = "var_decl"


@dataclass
class Statement:
    """
    Statement in the AST.
    
    Attributes:
        kind: Statement kind (Enum)
        data: The actual data for this kind of statement
        span: Source location
    """
    kind: StatementKind
    data: Any
    span: tuple[int, int]



# Attribute types

@dataclass
class Attribute:
    """
    Attribute on a declaration.
    
    Examples: @vertex, @group(0), @binding(1), @location(0)
    
    Attributes:
        name: Attribute name
        arguments: Attribute arguments
        span: Source location
    """
    name: str
    arguments: List[Any]
    span: tuple[int, int]
