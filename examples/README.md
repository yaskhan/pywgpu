# pywgpu Examples

This directory contains examples demonstrating the pywgpu API, which is a Python port of the wgpu-rs WebGPU implementation.

## Examples

### Basic Examples

1. **01_hello_compute.py** - Simple compute shader example
   - Demonstrates basic GPU compute operations
   - Doubles numbers using a compute shader
   - Shows data transfer between CPU and GPU
   - Introduces core pywgpu concepts

2. **02_hello_window.py** - Window creation and basic rendering
   - Shows how to initialize pywgpu
   - Demonstrates device and adapter creation
   - Placeholder for full window management (requires window backend)

3. **03_hello_triangle.py** - Basic triangle rendering
   - Demonstrates vertex buffer creation and data upload
   - Shows shader module compilation
   - Introduces render pipeline creation
   - Basic rendering concepts

4. **04_buffer_operations.py** - Buffer management and operations
   - Creates different buffer types (vertex, index, uniform, storage)
   - Demonstrates buffer usage flags
   - Shows data upload and mapping operations
   - Buffer-to-buffer copy operations

5. **05_shader_module.py** - Shader creation and compilation
   - Creates vertex and fragment shaders
   - Demonstrates compute shader modules
   - Shows shaders with uniform data
   - Introduces WGSL shader language concepts

### Running Examples

To run an example:

```bash
python examples/01_hello_compute.py 1.0 2.0 3.0 4.0
python examples/02_hello_window.py
python examples/03_hello_triangle.py
python examples/04_buffer_operations.py
python examples/05_shader_module.py
```

### Example Concepts

#### Compute Operations (01_hello_compute.py)
- Instance creation and adapter request
- Device initialization with compute support
- Storage buffer creation for input/output data
- Compute shader with workgroup dispatch
- Data transfer between CPU and GPU

#### Rendering Pipeline (03_hello_triangle.py)
- Vertex buffer creation and data upload
- Shader module compilation from WGSL code
- Render pipeline configuration
- Vertex attribute layout definition
- Color target state setup

#### Buffer Management (04_buffer_operations.py)
- Multiple buffer types with different usage flags
- Buffer mapping and unmapping operations
- Structured data packing/unpacking
- GPU-to-CPU data transfer
- Command encoder usage for buffer operations

#### Shader Programming (05_shader_module.py)
- WGSL shader language syntax
- Vertex and fragment shader creation
- Compute shader programming
- Uniform data binding
- Shader compilation and validation

## Architecture

The pywgpu library follows the same structure as the original wgpu project:

- **pywgpu** - User-facing API (analogous to `wgpu` crate)
- **pywgpu-core** - Engine implementation (analogous to `wgpu-core`)
- **pywgpu-hal** - Hardware abstraction layer (analogous to `wgpu-hal`)
- **pywgpu-types** - Shared type definitions (analogous to `wgpu-types`)
- **naga** - Shader translator (analogous to `naga`)

## Key Concepts

- **Instance** - Entry point for the wgpu API
- **Adapter** - Represents a physical GPU
- **Device** - Connection to a graphics/compute device
- **Queue** - Command submission interface
- **Buffers** - GPU memory for data storage
- **Shaders** - Programs that run on the GPU (WGSL format)
- **Pipelines** - Ready-to-go program state for the GPU

## Implementation Status

This is a work-in-progress Python implementation. Many methods are currently stubs and will be implemented as the backend support is added.

### Completed Method Implementations

- **Instance.create_surface()** - Surface creation with backend delegation
- **Instance.poll_all_devices()** - Device polling with fallback implementation
- **Instance.generate_report()** - Instance state reporting
- **Device.create_bind_group()** - Bind group creation with backend delegation
- **Device.create_bind_group_layout()** - Bind group layout creation
- **Device.create_pipeline_layout()** - Pipeline layout creation
- **Device.create_render_bundle_encoder()** - Render bundle encoder creation
- **Device.create_query_set()** - Query set creation
- **Device.create_compute_pipeline()** - Compute pipeline creation
- **Device.create_render_pipeline()** - Render pipeline creation
- **Device.create_pipeline_cache()** - Pipeline cache creation
- **Device.destroy()** - Device cleanup with resource management
- **Device.set_device_lost_callback()** - Device lost callback setup
- **Device.on_uncaptured_error()** - Error callback setup

## Dependencies

- Python 3.12+
- Pydantic V2
- asyncio (for async operations)
- struct (for data packing/unpacking)

## Contributing

When adding new examples, follow the patterns established in the existing examples and maintain compatibility with the wgpu API design principles.