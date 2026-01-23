"""
Parser for GLSL type declarations.

This module handles parsing of type definitions, struct types, and type qualifiers.
"""

from typing import Any, Optional, List, Dict
from enum import Enum


class TypeKind(Enum):
    """Types of type declarations."""
    BASIC = "basic"
    ARRAY = "array"
    STRUCT = "struct"
    FUNCTION = "function"
    POINTER = "pointer"


class TypeParser:
    """Parser for GLSL type declarations and definitions."""
    
    def __init__(self, lexer: Any):
        self.lexer = lexer
        self.errors: List[str] = []
        self.type_definitions: Dict[str, Any] = {}
        self.struct_definitions: Dict[str, Any] = {}
    
    def parse_type_specifier(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse a type specifier."""
        token = ctx.peek(frontend)
        if token is None:
            return None
        
        from ..token import TokenValue
        
        # Handle identifiers as potential type names
        if token.value == TokenValue.IDENTIFIER:
            type_name = self.parse_type_name(ctx, frontend)
            return self.resolve_type_name(type_name, frontend)
        
        # Handle keywords that are types
        if token.value == TokenValue.VOID:
            ctx.bump(frontend)
            return "void"
        
        if token.value == TokenValue.STRUCT:
            return self.parse_struct_type(ctx, frontend)
            
        return None
    
    def parse_struct_type(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """Parse a struct type definition."""
        from ..token import TokenValue
        from ....ir import Type, TypeInner, Struct, StructMember
        
        ctx.expect(frontend, TokenValue.STRUCT)
        
        # Optional struct name
        token = ctx.peek(frontend)
        struct_name = None
        if token and token.value == TokenValue.IDENTIFIER:
            struct_name, _ = ctx.expect_ident(frontend)
            
        ctx.expect(frontend, TokenValue.LEFT_BRACE)
        
        members = []
        while ctx.peek(frontend) and ctx.peek(frontend).value != TokenValue.RIGHT_BRACE:
            # Parse member declaration
            # In GLSL, members can have layout qualifiers and type qualifiers
            member_layout = frontend.declaration_parser.parse_layout_qualifier(ctx, frontend)
            member_qualifiers = frontend.declaration_parser.parse_type_qualifier(ctx, frontend)
            
            member_type_handle = self.parse_type_specifier(ctx, frontend)
            if member_type_handle is None:
                # Error recovery
                ctx.bump(frontend)
                continue
                
            member_name, _ = ctx.expect_ident(frontend)
            
            # Check for array dimensions on member
            if ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.LEFT_BRACKET:
                member_type_handle = self.parse_array_dimensions(ctx, frontend, member_type_handle)
                
            ctx.expect(frontend, TokenValue.SEMICOLON)
            
            # Create StructMember
            # Naga IR StructMember needs offset and binding, but those are often calculated later
            # For now, let's just store the type and name.
            # Real Naga calculated offsets based on std140/std430.
            member = StructMember(
                name=member_name,
                ty=member_type_handle,
                binding=None,
                offset=0 # Placeholder
            )
            members.append(member)
            
        ctx.expect(frontend, TokenValue.RIGHT_BRACE)
        
        # Create Struct type in NAGA IR
        struct_inner = TypeInner.new_struct(members=members, span=0) # Span placeholder
        new_type = Type(name=struct_name, inner=struct_inner)
        
        module = frontend.module
        handle = len(module.types)
        module.types.append(new_type)
        
        if struct_name:
            self.type_definitions[struct_name] = handle
            self.struct_definitions[struct_name] = handle
            
        return handle
    
    def parse_array_dimensions(self, ctx: Any, frontend: Any, base_type_handle: int) -> int:
        """Parse array dimensions and return the new type handle."""
        from ..token import TokenValue
        from ....ir import Type, TypeInner, ArraySize
        
        current_type_handle = base_type_handle
        
        while ctx.peek(frontend) and ctx.peek(frontend).value == TokenValue.LEFT_BRACKET:
            ctx.bump(frontend)
            
            # TODO: Handle constant expressions for array size
            token = ctx.peek(frontend)
            size = None
            if token and token.value == TokenValue.INT_CONSTANT:
                size = token.data.value
                ctx.bump(frontend)
            
            ctx.expect(frontend, TokenValue.RIGHT_BRACKET)
            
            # Create Array type in NAGA IR
            # In Naga, Array is a TypeInner
            if size is not None:
                array_size = ArraySize.new_constant(size)
            else:
                array_size = ArraySize.new_dynamic()
                
            # We need the base type's stride. For now, assume a default or use offset logic if available.
            # Real Naga implementation calculates stride based on layout.
            stride = 0 # Placeholder, should be calculated
            
            new_inner = TypeInner.new_array(
                base=current_type_handle,
                size=array_size,
                stride=stride
            )
            
            new_type = Type(name=None, inner=new_inner)
            
            module = frontend.module
            new_handle = len(module.types)
            module.types.append(new_type)
            current_type_handle = new_handle
            
        return current_type_handle

    def parse_array_type(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """
        Parse an array type.
        
        Args:
            ctx: Parsing context
            frontend: GLSL parser frontend
            
        Returns:
            Array type information or None
        """
        # TODO: Complete array type parsing
        # Should handle:
        # - Fixed-size arrays
        # - Unsized arrays
        # - Multi-dimensional arrays
        # - Array of structs
        # - Array qualifiers (restrict, etc.)
        
        return None
    
    def parse_type_qualifier(self, ctx: Any, frontend: Any) -> Optional[Any]:
        """
        Parse type qualifiers.
        
        Args:
            ctx: Parsing context
            frontend: GLSL parser frontend
            
        Returns:
            Type qualifier information or None
        """
        # TODO: Complete type qualifier parsing
        # Should handle:
        # - Precision qualifiers (highp, mediump, lowp)
        # - Storage qualifiers (const, uniform, buffer, etc.)
        # - Interpolation qualifiers (flat, smooth, centroid, sample)
        # - Invariant qualifiers
        # - Memory qualifiers (coherent, volatile, restrict, etc.)
        # - Layout qualifiers (location, binding, offset, align, etc.)
        
        return None
    
    def parse_type_name(self, ctx: Any, frontend: Any) -> Optional[str]:
        """Parse a type name (identifier)."""
        token = ctx.peek(frontend)
        if token is None:
            return None
        
        from ..token import TokenValue
        if token.value == TokenValue.IDENTIFIER:
            name, _ = ctx.expect_ident(frontend)
            return name
            
        return None

    def resolve_type_name(self, type_name: str, frontend: Any) -> Optional[int]:
        """Resolve a type name to a type handle in the module."""
        # Check if already resolved
        if type_name in self.type_definitions:
            return self.type_definitions[type_name]
            
        from ..types import parse_type
        ty = parse_type(type_name)
        
        if ty is not None:
            # Add to module and return index
            module = frontend.module
            handle = len(module.types)
            module.types.append(ty)
            self.type_definitions[type_name] = handle
            return handle
            
        return None
    
    def validate_type_compatibility(self, type1: Any, type2: Any) -> bool:
        """
        Validate type compatibility for operations.
        
        Args:
            type1: First type
            type2: Second type
            
        Returns:
            True if types are compatible
        """
        # TODO: Complete type compatibility validation
        # Should handle:
        # - Exact type matches
        # - Implicit conversion compatibility
        # - Assignment compatibility
        # - Arithmetic operation compatibility
        # - Comparison operation compatibility
        
        # TODO: These next ones seem incorrect to me
        # Some format mappings may be incorrect (e.g., "rgb10_a2ui" might not map correctly).
        # Review and fix any incorrect storage format mappings.
        
        return False