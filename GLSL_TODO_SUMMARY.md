# GLSL Frontend TODO Annotations Summary

This document summarizes the TODO comments that have been properly annotated and documented in the Python GLSL frontend code, based on translating from the Rust wgpu-trunk/naga/src/front/glsl implementation.

## Files Modified

### 1. naga/front/glsl/ast.py
**Line 132-160**: Precision qualifiers
- **TODO**: Encode precision hints in the IR
- **Context**: Precision qualifiers (highp, mediump, lowp) used in GLSL declarations
- **Reference**: wgpu-trunk/naga/src/front/glsl/ast.rs line 343
- **Notes**: Precision hints control the precision of arithmetic operations and can be used to optimize shader execution. They correspond to SPIR-V RelaxedPrecision decoration.

### 2. naga/front/glsl/functions.py
**Line 37-47**: Matrix width casts
- **TODO**: Matrix width casts require special handling
- **Context**: Expression::As operation doesn't support matrix width casts
- **Reference**: wgpu-trunk/naga/src/front/glsl/functions.rs line 222
- **Notes**: Conversions between matrices of different dimensions need to be decomposed into component-wise operations.

### 3. naga/front/glsl/builtins.py
**Line 65-68**: Bias with depth samplers
- **TODO**: glsl supports using bias with depth samplers but naga doesn't
- **Context**: GLSL allows bias parameters with depth/shadow samplers
- **Reference**: wgpu-trunk/naga/src/front/glsl/builtins.rs line 183
- **Notes**: When bias=true and shadow=true, that variation is skipped.

**Line 92-96**: modf and frexp functions
- **TODO**: https://github.com/gfx-rs/naga/issues/2526
- **Context**: Functions that require multiple return values
- **Reference**: wgpu-trunk/naga/src/front/glsl/builtins.rs line 1395
- **Notes**: modf splits floats into integer and fractional parts, frexp splits into mantissa and exponent.

### 4. naga/front/glsl/parser_main.py
**Line 117-120**: Extension handling
- **TODO**: Proper extension handling
- **Context**: Checking for extension support, handling behaviors like warn, handling the "all" extension
- **Reference**: wgpu-trunk/naga/src/front/glsl/parser.rs line 315
- **Notes**: Extensions can be required, enabled, warned about, or disabled.

**Line 196-200**: Pragma directives
- **TODO**: handle some common pragmas?
- **Context**: Common GLSL pragmas like optimize(on/off), debug(on/off)
- **Reference**: wgpu-trunk/naga/src/front/glsl/parser.rs line 402
- **Notes**: Vendor-specific pragmas should be preserved or ignored depending on target backend.

### 5. naga/front/glsl/offset.py
**Line 78-80**: Matrices array
- **TODO**: Matrices array
- **Context**: Arrays of matrices need special handling according to rule (5)
- **Reference**: wgpu-trunk/naga/src/front/glsl/offset.rs line 73
- **Notes**: Matrices are treated as arrays of column vectors.

**Line 87-89**: Row major matrices
- **TODO**: Row major matrices
- **Context**: Only column-major matrices currently supported
- **Reference**: wgpu-trunk/naga/src/front/glsl/offset.rs line 111
- **Notes**: Row-major matrices would store R row vectors with C components each.

### 6. naga/front/glsl/types.py
**Line 117-119**: Format and kind matching
- **TODO**: Check that the texture format and the kind match
- **Context**: Validate that format specifier matches kind specifier
- **Reference**: wgpu-trunk/naga/src/front/glsl/types.rs line 159
- **Notes**: For example, "uimage2D" should have unsigned format matching the "u" prefix.

**Line 268-271**: Multisampled storage images
- **TODO**: glsl support multisampled storage images, naga doesn't
- **Context**: GLSL allows multisampled storage images (image2DMS, image2DMSArray)
- **Reference**: wgpu-trunk/naga/src/front/glsl/types.rs line 167
- **Notes**: Naga's IR doesn't currently support this combination.

### 7. naga/front/glsl/variables.py
**Line 148-151**: Location counter
- **TODO**: glslang seems to use a counter for variables without explicit location
- **Context**: glslang assigns sequential location numbers even if it causes collisions
- **Reference**: wgpu-trunk/naga/src/front/glsl/variables.rs line 430
- **Notes**: Naga currently defaults to location 0.

**Line 339-342**: Writeonly images without format
- **TODO**: glsl supports images without format qualifier if they are `writeonly`
- **Context**: GLSL allows writeonly storage images to omit format qualifier
- **Reference**: wgpu-trunk/naga/src/front/glsl/variables.rs line 575
- **Notes**: Naga requires format qualifiers for all storage images.

### 8. naga/front/glsl/parser/declarations.py
**Line 49-50**: Layout arguments
- **TODO**: Accept layout arguments
- **Context**: Struct members should support layout qualifiers like offset, align, etc.
- **Reference**: wgpu-trunk/naga/src/front/glsl/parser/declarations.rs line 624
- **Notes**: Layout qualifiers control memory layout of struct members.

**Line 52-54**: Type qualifiers
- **TODO**: type_qualifier
- **Context**: Struct members should support type qualifiers
- **Reference**: wgpu-trunk/naga/src/front/glsl/parser/declarations.rs line 636
- **Notes**: Includes precision, interpolation, invariant qualifiers.

### 9. naga/front/glsl/parser/functions.py
**Line 52-57**: Implicit conversions
- **TODO**: Implicit conversions
- **Context**: Support for implicit type conversions in function arguments
- **Reference**: wgpu-trunk/naga/src/front/glsl/parser/functions.rs line 99
- **Notes**: Integer to float, boolean to integer, widening, precision conversions.

### 10. naga/front/glsl/parser/types.py
**Line 174-176**: Incorrect format mappings
- **TODO**: These next ones seem incorrect to me
- **Context**: Some storage format mappings may be incorrect (e.g., "rgb10_a2ui")
- **Reference**: wgpu-trunk/naga/src/front/glsl/parser/types.rs line 448
- **Notes**: Review and fix incorrect storage format mappings.

## Summary

All TODOs have been properly documented with:
1. Clear reference to the original Rust code location
2. Context about what the TODO refers to
3. Explanation of the limitation or missing feature
4. Technical details about what would be needed for implementation

The TODO comments follow the same structure and meaning as the original wgpu-trunk/naga Rust implementation, ensuring structural identity between the Python and Rust codebases.

## Implementation Status

All TODO comments are now properly annotated and documented. The Python code structure mirrors the Rust codebase layout, with each TODO preserving the original intent and context from the wgpu-trunk implementation.

No TODOs have been "filled in" with actual implementations, as per the project requirements - the focus was on properly documenting what needs to be implemented by translating the comments and context from the Rust source code.
