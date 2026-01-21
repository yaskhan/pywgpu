# Naga Proc Module - Constant Evaluator Implementation

## Overview

This directory contains processing utilities for Naga IR (Intermediate Representation), including the constant expression evaluator.

## Files

### Core Processing Utilities

| File | Purpose | Status |
|-------|----------|--------|
| `constant_evaluator.py` | Main constant expression evaluator (Python port of 161KB Rust file) | ✅ Framework Complete, TODO implementations |
| `type_methods.py` | Type-related helper functions (cross product, bit ops) | ✅ Implemented |
| `component_wise.py` | Component-wise operation infrastructure | 🚧 Framework Complete |
| `constant_evaluator_math.py` | Math function implementations | 🚧 Framework Complete |

### Existing Utilities

| File | Purpose | Status |
|-------|----------|--------|
| `namer.py` | Generate unique identifiers for expressions | ✅ Complete |
| `typifier.py` | Track expression types without creating arena entries | ✅ Complete |
| `layouter.py` | Compute type layouts and alignment | ✅ Complete |
| `keyword_set.py` | Manage reserved keyword sets | ✅ Complete |
| `emitter.py` | Manage expression emission for code generation | ✅ Complete |
| `terminator.py` | Ensure blocks have return statements | ✅ Complete |
| `index.py` | Bounds checking and indexing utilities | ✅ Complete |

## Constant Evaluator

### Purpose

The `ConstantEvaluator` class evaluates constant expressions at compile time, reducing them to their computed values. This is a critical component for shader optimization and validation.

### Implementation Status

#### Completed (Framework)

- **ExpressionKindTracker**: Tracks whether expressions are const, override, or runtime
- **ConstantEvaluator** class structure with all required fields
- **Factory methods**: `for_wgsl_module()`, `for_glsl_module()`
- **Evaluation pipeline**: `try_eval_and_append()` → `try_eval_and_append_impl()`
- **Error handling**: `ConstantEvaluatorError` exception with details

#### Partially Implemented

- **Expression matching**: All `ExpressionType` variants are matched in dispatcher
- **Basic expression passthrough**: `Literal`, `ZeroValue`, `Constant` expressions work
- **Expression registration**: `register_evaluated_expr()` with validation hooks

#### Needs TODO Implementation

The following methods are stubbed with `NotImplementedError`:

1. **Literal Evaluation**:
   - `literal.ty_inner()` - Return TypeInner for a Literal
   - `LiteralVector.from_literal()` / `from_literal_vec()` - Type conversion

2. **Zero Value Handling**:
   - `eval_zero_value_impl()` - Complete for all TypeInner variants
   - Vector, Matrix, Array, Struct zero generation

3. **Flattening**:
   - `flatten_compose()` - Recursively expand nested structures
   - Handle Splat expansion in compositions

4. **Unary Operations**:
   - `unary_op()` - All `UnaryOperator` variants
   - Negate, LogicalNot, BitwiseNot

5. **Binary Operations**:
   - `binary_op()` - All `BinaryOperator` variants
   - Arithmetic, comparison, bitwise, shift operations
   - Overflow/underflow checking

6. **Math Functions** (`constant_evaluator_math.py`):
   - Comparison: Abs, Min, Max, Clamp, Saturate
   - Trigonometry: Sin, Cos, Tan, inverses, hyperbolics
   - Decomposition: Ceil, Floor, Round, Fract, Trunc
   - Exponent: Exp, Log, Pow, Sqrt, InverseSqrt
   - Bit ops: CountLeadingZeros, CountOneBits, ReverseBits, etc.
   - Vector ops: Dot, Cross, Length, Distance, Normalize

7. **Advanced Expressions**:
   - `select()` - Conditional selection
   - `swizzle()` - Component reordering
   - `access()` - Composite access

### Expression Types Supported

| Expression Type | Status | Notes |
|----------------|--------|-------|
| Literal | ✅ | Passthrough |
| Constant | ✅ | See-through to initializer |
| ZeroValue | ✅ | Basic structure |
| Compose | 🚧 | Structure complete, needs recursive eval |
| Splat | 🚧 | Structure complete |
| AccessIndex | 🚧 | Basic structure |
| Access | 🚧 | Structure complete |
| Swizzle | 🚧 | Structure complete |
| Unary | 🚧 | Structure complete |
| Binary | 🚧 | Structure complete |
| Math | 🚧 | Structure complete |
| As (cast) | ⚠️ | Only structure, no implementation |
| Select | 🚧 | Structure complete |
| Relational | 🚧 | Structure complete (All/Any) |
| ArrayLength | 🚧 | Basic structure |
| Load | ❌ | Not supported (returns error) |
| ImageSample | ❌ | Not supported (returns error) |
| Derivative | ❌ | Not supported (returns error) |

Legend:
- ✅ Implemented
- 🚧 Partial/Stubbed
- ⚠️ Limited
- ❌ Not applicable

## Type Methods

### Implemented Functions

- `cross_product(a, b)`: 3D vector cross product
- `first_trailing_bit(value, signed)`: Find first set bit from LSB
- `first_leading_bit(value, signed)`: Find first set bit from MSB
- `extract_scalar_components(literals)`: Extract scalar values
- `flatten_compose()`: Flatten nested compositions (stub)

### Type Resolution

The module provides `TypeResolution` as a union of:
- `TypeResolutionHandle`: References a type handle
- `TypeResolutionValue`: Contains an inline `TypeInner`

This allows expressions to reference types either by handle or inline value.

## Component-Wise Operations

The `component_wise.py` module provides infrastructure for:

1. **Component-wise extractors**:
   - `component_wise_scalar()`: For all numeric types
   - `component_wise_float()`: For floating-point only
   - `component_wise_concrete_int()`: For i32/u32 only
   - `component_wise_signed()`: For signed numbers only

2. **Math function metadata**:
   - `math_function_arg_count()`: Returns expected argument count
   - Categorizes functions by argument count (1, 2, or 3)

3. **Helper functions**:
   - `flatten_compose_to_literals()`: Extract literals from compositions
   - `extract_vector_literals()`: Get literals from vector expressions

## Usage Example

```python
from naga import Module
from naga.proc import ConstantEvaluator, ExpressionKindTracker, Layouter
from naga import Expression

# Create evaluator for WGSL module
tracker = ExpressionKindTracker.from_arena(module.global_expressions)
layouter = Layouter()
evaluator = ConstantEvaluator.for_wgsl_module(
    module,
    tracker,
    layouter,
    in_override_ctx=False,
)

# Evaluate an expression
expr = Expression(
    type=ExpressionType.BINARY,
    binary_op=BinaryOperator.ADD,
    binary_left=left_handle,
    binary_right=right_handle,
)
result_handle = evaluator.try_eval_and_append(expr, span)

# Check the result
result_expr = evaluator.expressions[result_handle]
```

## Architecture Notes

### Expression Matching

The implementation uses Python's pattern matching (PEP 634) to dispatch
expression evaluation based on `Expression.type`. This matches the Rust
`match` statement structure closely.

### Arena-Based Expressions

All expressions are stored in arenas (`Arena[Expression]`) and referenced by
handles (`Handle[Expression]`). This mirrors the Rust approach to
enable efficient manipulation of large IR structures.

### Type Resolution

Types are stored in a `UniqueArena[Type]` and can be referenced:
1. By handle (`Handle[Type]`) - for named/composite types
2. Inline (`TypeInner`) - for simple scalar types

### Constness Tracking

`ExpressionKindTracker` maintains constness information for each expression:
- **CONST**: Can be evaluated at compile time
- **OVERRIDE**: Pipeline-overridable constant
- **RUNTIME**: Requires runtime evaluation

## Testing Strategy

### Unit Tests

Each math function and operator should have tests covering:

1. **Scalar inputs**: Verify single-value operations
2. **Vector inputs**: Verify component-wise operations
3. **Mixed inputs**: Verify coercion rules
4. **Edge cases**: Overflow, NaN, zeros
5. **Error paths**: Verify proper error reporting

### Integration Tests

Test the full constant evaluation pipeline with:

1. **Simple expressions**: Literals, basic arithmetic
2. **Complex expressions**: Nested compositions, casts
3. **WGSL mode**: Const vs override vs runtime contexts
4. **GLSL mode**: Specialization constant handling

## Future Work

### High Priority

1. **Complete literal evaluation**: Add `ty_inner()` method to `Literal` types
2. **Implement zero values**: Complete `eval_zero_value_impl()`
3. **Implement flattening**: Complete `flatten_compose()` recursion

### Medium Priority

4. **Implement unary operations**: All `UnaryOperator` variants
5. **Implement binary operations**: All `BinaryOperator` variants
6. **Implement math functions**: All `MathFunction` variants

### Low Priority

7. **Implement select**: Full scalar and vector support
8. **Implement swizzle**: Pattern matching and reordering
9. **Implement access**: Dynamic index handling
10. **Implement casting**: All type conversions

## References

- Original Rust: `/tmp/wgpu/naga/src/proc/constant_evaluator.rs` (4103 lines)
- Naga IR: `naga/ir/` directory
- WGSL Spec: https://www.w3.org/TR/WGSL/
- GLSL Spec: https://www.khronos.org/registry/OpenGL-Refpages/glsl4.5/
