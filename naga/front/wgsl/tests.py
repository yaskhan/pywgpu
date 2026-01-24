"""
WGSL frontend tests.

Translated from wgpu-trunk/naga/src/front/wgsl/tests.rs

This module contains tests for the WGSL parser and frontend.
"""

import pytest
from . import parse_str, ParseError
from .lexer import Lexer, TokenKind
from .number import parse_number, NumberType
from .conv import get_scalar_type, map_built_in
from .directive import parse_enable_directive
from .enable_extension import ImplementedEnableExtension
from .language_extension import ImplementedLanguageExtension


class TestLexer:
    """Tests for the WGSL lexer."""
    
    def test_keywords(self):
        """Test keyword tokenization."""
        lexer = Lexer("fn var const struct")
        tokens = list(lexer)
        
        assert tokens[0].kind == TokenKind.FN
        assert tokens[1].kind == TokenKind.VAR
        assert tokens[2].kind == TokenKind.CONST
        assert tokens[3].kind == TokenKind.STRUCT
    
    def test_identifiers(self):
        """Test identifier tokenization."""
        lexer = Lexer("myVar my_func MyStruct")
        tokens = list(lexer)
        
        assert all(t.kind == TokenKind.IDENT for t in tokens[:-1])
        assert tokens[0].value == "myVar"
        assert tokens[1].value == "my_func"
        assert tokens[2].value == "MyStruct"
    
    def test_operators(self):
        """Test operator tokenization."""
        lexer = Lexer("+ - * / == != < > <= >=")
        tokens = list(lexer)
        
        assert tokens[0].kind == TokenKind.PLUS
        assert tokens[1].kind == TokenKind.MINUS
        assert tokens[2].kind == TokenKind.STAR
        assert tokens[3].kind == TokenKind.FORWARD_SLASH
    
    def test_comments(self):
        """Test comment handling."""
        lexer = Lexer("// line comment\nvar /* block comment */ x")
        tokens = list(lexer)
        
        # Comments should be skipped
        assert tokens[0].kind == TokenKind.VAR
        assert tokens[1].kind == TokenKind.IDENT


class TestNumberParsing:
    """Tests for number literal parsing."""
    
    def test_decimal_integer(self):
        """Test decimal integer parsing."""
        num = parse_number("123", (0, 3))
        assert num.value == 123
        assert num.type_ == NumberType.ABSTRACT_INT
    
    def test_hex_integer(self):
        """Test hexadecimal integer parsing."""
        num = parse_number("0x1A2B", (0, 6))
        assert num.value == 0x1A2B
        assert num.type_ == NumberType.ABSTRACT_INT
    
    def test_float(self):
        """Test float parsing."""
        num = parse_number("1.5", (0, 3))
        assert num.value == 1.5
        assert num.type_ == NumberType.ABSTRACT_FLOAT
    
    def test_scientific_notation(self):
        """Test scientific notation."""
        num = parse_number("1.5e10", (0, 6))
        assert num.value == 1.5e10
        assert num.type_ == NumberType.ABSTRACT_FLOAT
    
    def test_typed_integer(self):
        """Test typed integer literals."""
        num_i = parse_number("123i", (0, 4))
        assert num_i.type_ == NumberType.I32
        
        num_u = parse_number("123u", (0, 4))
        assert num_u.type_ == NumberType.U32
    
    def test_typed_float(self):
        """Test typed float literals."""
        num_f = parse_number("1.5f", (0, 4))
        assert num_f.type_ == NumberType.F32
        
        num_h = parse_number("1.5h", (0, 4))
        assert num_h.type_ == NumberType.F16


class TestConversion:
    """Tests for type and keyword conversion."""
    
    def test_scalar_types(self):
        """Test scalar type mapping."""
        assert get_scalar_type(set(), (0, 0), "bool") is not None
        assert get_scalar_type(set(), (0, 0), "i32") is not None
        assert get_scalar_type(set(), (0, 0), "u32") is not None
        assert get_scalar_type(set(), (0, 0), "f32") is not None
    
    def test_builtins(self):
        """Test built-in mapping."""
        from ...ir import BuiltIn
        
        assert map_built_in(set(), "vertex_index", (0, 0)) == BuiltIn.VERTEX_INDEX
        assert map_built_in(set(), "position", (0, 0)) == BuiltIn.POSITION
        assert map_built_in(set(), "frag_depth", (0, 0)) == BuiltIn.FRAG_DEPTH


class TestDirectives:
    """Tests for directive parsing."""
    
    def test_enable_directive(self):
        """Test enable directive parsing."""
        directive = parse_enable_directive(["f16"])
        assert ImplementedEnableExtension.F16 in directive.extensions
    
    def test_unknown_extension(self):
        """Test unknown extension error."""
        with pytest.raises(ParseError):
            parse_enable_directive(["unknown_extension"])


class TestParser:
    """Tests for the WGSL parser."""
    
    def test_empty_shader(self):
        """Test parsing empty shader."""
        # This should fail - no entry point
        with pytest.raises(ParseError):
            parse_str("")
    
    def test_simple_vertex_shader(self):
        """Test parsing simple vertex shader."""
        source = """
        @vertex
        fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        """
        
        module = parse_str(source)
        assert module is not None
        assert len(module.entry_points) == 1
        assert module.entry_points[0].name == "main"
    
    def test_struct_declaration(self):
        """Test parsing struct declaration."""
        source = """
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec4<f32>,
        }
        
        @vertex
        fn main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
        """
        
        module = parse_str(source)
        assert module is not None
        # Should have VertexOutput and vec4<f32> etc.
        assert len(module.types) >= 1

    
    def test_error_reporting(self):
        """Test error message formatting."""
        source = "fn invalid syntax here"
        
        try:
            parse_str(source)
        except ParseError as e:
            error_msg = e.emit_to_string(source)
            assert "error:" in error_msg
            assert "fn" in error_msg


class TestLowerer:
    """Tests for AST to IR lowering."""
    
    def test_type_conversion(self):
        """Test type conversion."""
        source = "struct S { a: f32, b: array<i32, 4> }"
        module = parse_str(source)
        # Struct S and f32, i32, array<i32, 4>
        assert len(module.types) >= 4
        
    def test_expression_lowering(self):
        """Test expression lowering."""
        source = """
        fn dummy() {
            let x = 1.0 + 2.0;
            let y = abs(-x);
        }
        """
        module = parse_str(source)
        func = module.functions[0]
        # Should have several expressions for literals, binary, and math call
        assert len(func.expressions) >= 5
        
    def test_statement_lowering(self):
        """Test statement lowering."""
        source = """
        fn dummy() {
            if (true) {
                return;
            }
        }
        """
        module = parse_str(source)
        func = module.functions[0]
        # Should have an If statement
        from ...ir import StatementType
        assert func.body[0].type == StatementType.IF


# Integration tests
class TestIntegration:
    """Integration tests with complete WGSL shaders."""
    
    def test_triangle_shader(self):
        """Test complete triangle shader."""
        source = """
        @vertex
        fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
            let x = f32(i32(in_vertex_index) - 1);
            let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
            return vec4<f32>(x, y, 0.0, 1.0);
        }
        
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        """
        
        module = parse_str(source)
        assert len(module.entry_points) == 2
        assert any(ep.name == "vs_main" for ep in module.entry_points)
        assert any(ep.name == "fs_main" for ep in module.entry_points)
    
    def test_compute_shader(self):
        """Test compute shader."""
        source = """
        @group(0) @binding(0)
        var<storage, read_write> data: array<u32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            data[global_id.x] = global_id.x;
        }
        """
        
        module = parse_str(source)
        assert len(module.entry_points) == 1
        assert module.entry_points[0].name == "main"
        assert len(module.global_variables) == 1




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
