# Naga Backend Implementation

This directory contains the Python implementation of shader backends for the Naga IR (Intermediate Representation) system. These backends convert high-level shader code into various target languages and formats.

## Overview

The backends directory implements the following shader compilation targets:

### WGSL Backend (`wgsl/`)
- **Purpose**: WebGPU Shading Language output
- **File**: `wgsl/__init__.py`
- **Features**: 
  - Full WGSL syntax support
  - Enable declarations (f16, dual_source_blending, etc.)
  - Function and entry point generation
  - Type system translation
  - Statement and expression rendering

### GLSL Backend (`glsl/`)
- **Purpose**: OpenGL Shading Language output
- **File**: `glsl/__init__.py`
- **Features**:
  - Multiple GLSL versions (140, 150, 330, 400, 410, 420, 430, 440, 450, 460)
  - Core and ES profiles support
  - Extension handling
  - Varying and uniform declarations
  - Shader stage-specific output (vertex, fragment, compute)

### HLSL Backend (`hlsl/`)
- **Purpose**: High-Level Shading Language output (DirectX)
- **File**: `hlsl/__init__.py`
- **Features**:
  - Multiple shader models (5.0, 5.1, 6.0-6.7)
  - Shader stage-specific functions (VS, PS, CS, HS, DS, GS)
  - Constant buffer and structured buffer support
  - Semantic mapping for built-in variables
  - Matrix layout handling (row/column major)

### MSL Backend (`msl/`)
- **Purpose**: Metal Shading Language output
- **File**: `msl/__init__.py`
- **Features**:
  - Metal-specific syntax and annotations
  - Thread group and grid coordination
  - Texture and sampler declarations
  - Attribute annotations (`[[attribute]]`)
  - Mesh and task shader support

### SPIR-V Backend (`spv/`)
- **Purpose**: Standard Portable Intermediate Representation
- **File**: `spv/__init__.py`
- **Features**:
  - Binary SPIR-V generation
  - Capability management
  - Type system mapping
  - Instruction generation
  - Storage class handling

## Architecture

Each backend follows a consistent pattern:

1. **Writer Class**: Main class that handles the conversion
2. **Options**: Configuration for target-specific settings
3. **Helper Classes**: Type mappings, name generators, etc.
4. **Utility Functions**: Standalone conversion functions

### Common Interface

All backends inherit from the base `Writer` class and implement:

```python
def write(self, module: Any, info: Any) -> Any:
    """Write the complete module to target format"""
    
def finish(self) -> str:
    """Return the complete generated code"""
```

## Usage Examples

### WGSL Output
```python
from naga.back import write_wgsl_string, WriterFlags

wgsl_code = write_wgsl_string(module, info, WriterFlags.EXPLICIT_TYPES)
```

### GLSL Output
```python
from naga.back import write_glsl_string, GlslOptions, GlslOptions.Version, GlslOptions.Profile

options = GlslOptions(GlslOptions.Version.V430, GlslOptions.Profile.Core)
glsl_code = write_glsl_string(module, info, options, pipeline_options)
```

### HLSL Output
```python
from naga.back import write_hlsl_string, HlslOptions, HlslOptions.ShaderModel

options = HlslOptions(HlslOptions.ShaderModel.SM_6_0)
hlsl_code = write_hlsl_string(module, info, options, "main", HlslOptions.ShaderStage.Vertex)
```

### MSL Output
```python
from naga.back import write_msl_string, MslOptions

options = MslOptions()
msl_code = write_msl_string(module, info, options, "main", MslOptions.ShaderStage.Vertex)
```

### SPIR-V Binary Output
```python
from naga.back import write_spirv_binary, SpvOptions

options = SpvOptions()
spirv_binary = write_spirv_binary(module, info, options)
```

## Implementation Status

### ✅ Completed
- [x] WGSL backend - Full implementation
- [x] GLSL backend - Core implementation with version support
- [x] HLSL backend - Basic implementation with shader model support
- [x] MSL backend - Metal-specific syntax and structure
- [x] SPIR-V backend - Binary generation framework
- [x] Base Writer class with common interface
- [x] Type mapping systems
- [x] Name generation and conflict resolution
- [x] Entry point handling for different shader stages

### 🚧 In Progress
- [ ] Full expression and statement coverage
- [ ] Advanced optimization passes
- [ ] Complete feature detection and capability management
- [ ] Comprehensive error handling and validation

### 📋 Planned
- [ ] Performance optimizations
- [ ] Advanced debugging and logging
- [ ] Integration with actual Naga IR parser
- [ ] Test suite with real shader examples
- [ ] Documentation and usage guides

## Development Notes

### Mapping from Rust Implementation
These Python backends are direct ports of the Rust implementations found in:
- `wgpu/naga/src/back/wgsl/`
- `wgpu/naga/src/back/glsl/`
- `wgpu/naga/src/back/hlsl/`
- `wgpu/naga/src/back/msl/`
- `wgpu/naga/src/back/spv/`

### Key Differences from Rust
1. **Type System**: Python's dynamic typing vs Rust's static typing
2. **Memory Management**: Python's garbage collection vs Rust's ownership
3. **Error Handling**: Python exceptions vs Rust's Result type
4. **Performance**: Interpreted vs compiled execution

### Future Enhancements
1. **Cython Integration**: For performance-critical sections
2. **NumPy Integration**: For efficient data handling
3. **Parallel Processing**: Multi-threaded shader compilation
4. **Advanced Optimizations**: Backend-specific optimization passes

## Testing

Each backend includes baseline implementations that can be extended with comprehensive test suites. The implementations provide basic structure for:

- Type translation correctness
- Shader stage-specific output
- Error handling and validation
- Performance benchmarks

## Contributing

When adding new features or fixing bugs:

1. Ensure consistency with the Rust reference implementation
2. Maintain the established naming conventions
3. Add appropriate error handling
4. Include documentation for new features
5. Test with real shader examples when possible

## License

This implementation follows the same licensing as the original Naga project.