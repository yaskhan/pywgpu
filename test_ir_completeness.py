#!/usr/bin/env python3
"""
Test script to verify the completeness of IR implementations.
Tests the core data structures and methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from naga.ir import (
    Block, Type, ScalarKind, VectorSize, Constant, 
    Function, FunctionResult, Module
)
from naga.ir.block import Block as CoreBlock
from naga.ir.type import Scalar, Vector, Matrix, Array, Struct, StructMember, Image, ImageDimension

def test_block_methods():
    """Test Block class methods."""
    print("Testing Block methods...")
    
    # Test basic creation and methods
    block = Block.new()
    assert block.is_empty()
    assert len(block) == 0
    
    # Test with_capacity
    block2 = Block.with_capacity(10)
    assert len(block2) == 0
    
    # Test append and length
    from naga.span import Span
    block.append("statement1", Span())
    assert len(block) == 1
    assert not block.is_empty()
    
    # Test extend_block
    block3 = Block.from_vec(["stmt2", "stmt3"])
    block.extend_block(block3)
    assert len(block) == 3
    
    # Test iteration
    statements = list(block)
    assert len(statements) == 3
    
    # Test cull
    block.cull(1, 2)
    assert len(block) == 2
    
    # Test splicing
    block4 = Block.from_vec(["new_stmt"])
    block.splice(1, 1, block4)
    assert len(block) == 3
    
    # Test span iteration
    span_pairs = list(block.span_iter())
    assert len(span_pairs) == 3
    
    print("✓ Block methods test passed")

def test_type_methods():
    """Test Type class methods."""
    print("Testing Type methods...")
    
    # Test scalar creation
    scalar = Type.scalar(ScalarKind.FLOAT, 4)
    assert scalar.inner.value == "scalar"
    assert hasattr(scalar, '_scalar')
    
    # Test vector creation
    vector = Type.vector(VectorSize.TRI, ScalarKind.FLOAT, 4)
    assert vector.inner.value == "vector"
    assert hasattr(vector, '_vector')
    
    # Test matrix creation
    matrix = Type.matrix(VectorSize.QUAD, VectorSize.QUAD, 4)
    assert matrix.inner.value == "matrix"
    assert hasattr(matrix, '_matrix')
    
    # Test array creation
    array = Type.array(1, 10)
    assert array.inner.value == "array"
    assert hasattr(array, '_array')
    
    # Test struct creation
    member = StructMember("field1", 1, 0)
    struct = Type.struct("MyStruct", [member])
    assert struct.inner.value == "struct"
    
    # Test image creation
    image = Type.image(ImageDimension.D2, False, "sampled")
    assert image.inner.value == "image"
    
    print("✓ Type methods test passed")

def test_constant_methods():
    """Test Constant class methods."""
    print("Testing Constant methods...")
    
    # Test literal
    lit = Constant.literal(1, 42)
    assert lit.name is None
    assert lit.ty == 1
    assert lit.value == 42
    
    # Test composite
    comp = Constant.composite(2, [1, 2, 3])
    assert comp.name is None
    assert comp.ty == 2
    assert hasattr(comp, 'components')
    
    # Test zero
    zero = Constant.zero(3)
    assert zero.name is None
    assert zero.ty == 3
    assert zero.value == 0
    
    print("✓ Constant methods test passed")

def test_function_methods():
    """Test Function class methods."""
    print("Testing Function methods...")
    
    body = Block.new()
    result = FunctionResult(0, None)
    func = Function.new("test_func", result, body)
    
    # Test argument addition
    arg = func.add_argument("x", 1, None)
    assert arg.name == "x"
    assert arg.ty == 1
    assert len(func.arguments) == 1
    
    # Test local variable addition
    var = func.add_local_var("temp", 2, None)
    assert var.name == "temp"
    assert var.ty == 2
    assert len(func.local_variables) == 1
    
    # Test expression addition
    expr_idx = func.add_expression("some_expr")
    assert expr_idx == 0
    assert len(func.expressions) == 1
    
    # Test named expressions
    func.set_named_expression("result", 0)
    assert "result" in func.named_expressions
    
    print("✓ Function methods test passed")

def test_module_methods():
    """Test Module class methods."""
    print("Testing Module methods...")
    
    module = Module()
    
    # Test type addition
    ty_idx = module.add_type("float", "scalar")
    assert ty_idx == 0
    assert len(module.types) == 1
    
    # Test constant addition
    const_idx = module.add_constant("PI", 0, 3.14)
    assert const_idx == 0
    assert len(module.constants) == 1
    
    # Test function addition
    body = Block.new()
    func = module.add_function("main", None, body)
    assert len(module.functions) == 1
    assert func.name == "main"
    
    # Test entry point addition
    entry = module.add_entry_point("main", "compute", func)
    assert len(module.entry_points) == 1
    assert entry.name == "main"
    assert entry.stage == "compute"
    
    # Test named expressions
    module.set_named_expression("global", 5)
    assert "global" in module.named_expressions
    
    print("✓ Module methods test passed")

def main():
    """Run all tests."""
    print("Testing IR completeness...\n")
    
    try:
        test_block_methods()
        test_type_methods()
        test_constant_methods()
        test_function_methods()
        test_module_methods()
        
        print("\n✅ All IR completeness tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)