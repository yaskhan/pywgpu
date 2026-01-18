# Naga Validator

This module implements the Python port of the Naga shader validator from the `wgpu-rs` project.

## Overview

The validator ensures that a Naga IR `Module` is structurally correct and semantically valid according to the Naga IR specification. It performs various checks including:

- Type correctness (scalar/vector/matrix constraints, struct layouts)
- Constant expression correctness
- Function argument/result validity
- Entry point stage and binding constraints
- Control flow uniformity (if enabled)
- Resource binding uniqueness

## Architecture

The validator is structured to mirror the Rust implementation in `wgpu/naga/src/valid/mod.rs`:

```
naga/valid/
├── __init__.py          # Public API exports
├── validator.py         # Main Validator class
├── flags.py             # ValidationFlags, Capabilities, TypeFlags
├── errors.py            # ValidationError and related exceptions
├── module_info.py       # ModuleInfo, FunctionInfo, ExpressionInfo
└── README.md            # This file
```

## Usage

### Basic Validation

```python
from naga.ir.module import Module
from naga.valid import Validator

# Create a module
module = Module()
module.add_type("float", "scalar")

# Validate it
validator = Validator()
info = validator.validate(module)

# Use validation results
print(f"Validated {len(info.type_flags)} types")
```

### Custom Validation Flags

```python
from naga.valid import Validator, ValidationFlags

# Only validate expressions and constants
flags = ValidationFlags.EXPRESSIONS | ValidationFlags.CONSTANTS
validator = Validator(flags=flags)
```

### Capability Constraints

```python
from naga.valid import Validator, Capabilities

# Allow 64-bit float and integer operations
caps = Capabilities.FLOAT64 | Capabilities.SHADER_INT64
validator = Validator(capabilities=caps)
```

## Components

### Validator

The main validator class that orchestrates all validation checks.

**Key Methods:**
- `validate(module)` - Validate a module and return ModuleInfo
- `validate_resolved_overrides(module)` - Validate with all overrides resolved
- `reset()` - Reset internal state for reuse

### ValidationFlags

IntFlag enum controlling which validation stages to perform:

- `EXPRESSIONS` - Validate expressions
- `BLOCKS` - Validate statements and blocks
- `CONTROL_FLOW_UNIFORMITY` - Check uniformity requirements
- `STRUCT_LAYOUTS` - Validate host-shareable struct layouts
- `CONSTANTS` - Validate constants
- `BINDINGS` - Validate resource bindings

### Capabilities

IntFlag enum specifying allowed shader capabilities:

- `FLOAT64` - 64-bit floating point
- `SHADER_INT64` - 64-bit integers
- `SUBGROUP` - Subgroup operations
- `RAY_QUERY` - Ray tracing queries
- `MESH_SHADER` - Mesh/task shaders
- And many more...

### TypeFlags

IntFlag enum describing type properties:

- `CONSTRUCTIBLE` - Type can be constructed
- `HOST_SHAREABLE` - Type can be shared with host
- `IO_SHAREABLE` - Type can be used for I/O
- `DATA` - Type can hold data
- `SIZED` - Type has a known size
- `COPY` - Type can be copied
- `ARGUMENT` - Type can be a function argument

### ModuleInfo

Returned by validation, contains analysis results:

```python
@dataclass
class ModuleInfo:
    type_flags: list[TypeFlags]
    functions: list[FunctionInfo]
    entry_points: list[FunctionInfo]
    const_expression_types: list[object]
```

## Implementation Status

This is a **skeleton implementation** that provides the core validation infrastructure:

✅ **Implemented:**
- Module structure validation
- Type validation with flag computation
- Constant validation
- Function and entry point validation
- Global variable validation
- Handle reference validation

🚧 **Planned:**
- Full expression validation
- Statement validation
- Control flow uniformity analysis
- Layout computation and validation
- Binding uniqueness checks
- Built-in usage validation

## Design Principles

Following AGENTS.md guidelines:

1. **Type Safety**: No `Any` types where possible, explicit type annotations
2. **Structural Identity**: Mirrors Rust crate structure from `wgpu/naga`
3. **Google Style Docstrings**: All public APIs documented
4. **IntFlag for Bitflags**: Using Python's IntFlag to mirror Rust bitflags
5. **Dataclasses**: Using dataclasses for structured data

## Testing

The validator includes comprehensive tests covering:

- Basic validation workflows
- Type and constant validation
- Function and entry point validation
- Validation flags and capabilities
- Error detection for invalid modules
- ModuleInfo getter methods

Run tests with:

```bash
cd /home/engine/project
python3 -c "from naga.valid import Validator; ..."
```

## References

- Original Rust implementation: `wgpu/naga/src/valid/mod.rs`
- WebGPU Shading Language: https://www.w3.org/TR/WGSL/
- Naga repository: https://github.com/gfx-rs/wgpu
