# PyWGPU HAL Examples

This directory contains Python translations of wgpu-hal examples from the Rust `wgpu-trunk` project.

## Examples

### 1. raw_gles.py
**Original:** `wgpu-trunk/wgpu-hal/examples/raw-gles.rs`

Demonstrates interop with raw GLES contexts - how to hook up pywgpu-hal to an existing OpenGL ES context and render into it.

**Requirements:**
```bash
pip install glfw PyOpenGL
```

**Run:**
```bash
python examples/hal/raw_gles.py
```

### 2. halmark.py (Simplified)
**Original:** `wgpu-trunk/wgpu-hal/examples/halmark/main.rs` (898 lines)

A simplified version of the halmark benchmark that demonstrates basic HAL usage by rendering moving sprites.

**Features:**
- Device initialization
- Buffer creation and mapping
- Texture creation
- Render pipeline setup
- Command encoding and submission

**Run:**
```bash
python examples/hal/halmark.py
```

### 3. ray_traced_triangle.py (Simplified)
**Original:** `wgpu-trunk/wgpu-hal/examples/ray-traced-triangle/main.rs` (1200 lines)

A simplified ray tracing example demonstrating acceleration structure usage.

**Features:**
- Bottom-level acceleration structure (BLAS) creation
- Top-level acceleration structure (TLAS) creation
- Ray tracing pipeline
- Compute shader with ray queries

**Requirements:**
- GPU with ray tracing support
- DXR (DirectX 12) or Vulkan ray tracing

**Run:**
```bash
python examples/hal/ray_traced_triangle.py
```

## Notes

The original Rust examples are very comprehensive and include:
- Detailed error handling
- Platform-specific code paths
- Advanced features like pipeline caching
- Complex resource management

These Python translations focus on the core concepts and are simplified for clarity and educational purposes.

## Original Examples

The original Rust examples can be found in:
- `wgpu-trunk/wgpu-hal/examples/raw-gles.rs`
- `wgpu-trunk/wgpu-hal/examples/halmark/`
- `wgpu-trunk/wgpu-hal/examples/ray-traced-triangle/`

## Architecture

All examples follow this pattern:

1. **Initialization**
   - Create HAL instance
   - Enumerate adapters
   - Open device and queue

2. **Resource Creation**
   - Buffers (vertex, uniform, etc.)
   - Textures
   - Shaders
   - Pipelines

3. **Rendering Loop**
   - Acquire surface texture
   - Encode commands
   - Submit to queue
   - Present

4. **Cleanup**
   - Destroy resources
   - Drop device and instance
