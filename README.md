# pywgpu

**pywgpu** is a high-performance Python port of the [wgpu-rs](https://github.com/gfx-rs/wgpu) project. It provides a modern, safe, and strictly-typed interface for graphics and compute on the GPU, following the WebGPU standard.

## 🏗 Architecture

The project is an exact structural copy of the original `wgpu` repository, divided into the following components:

- **`pywgpu-hal`**: Hardware Abstraction Layer. Implements low-level backends (Vulkan, Metal, DX12, GLES) via CFFI.
- **`pywgpu-core`**: The "brain" of the library. Handles resource management, state tracking, and validation.
- **`pywgpu`**: High-level, user-facing Pythonic API (equivalent to the `wgpu` crate).
- **`naga`**: A pure Python port of the Naga shader translator (WGSL/SPIR-V translation and validation).
- **`pywgpu-types`**: Shared data structures and descriptors powered by **Pydantic V2**.

## 🛠 Tech Stack

- **Python 3.12+**: Leveraging modern features like `typing.Protocol` and improved generics.
- **Strict Static Typing**: 100% type coverage. Verified with `pyright` in strict mode.
- **Pydantic V2**: Robust validation for all GPU descriptors and state objects.
- **CFFI**: High-performance interaction with native graphics APIs.
- **NumPy**: Efficient zero-copy data handling for buffers and textures.

## 📄 Documentation

- [**SPECIFICATION.md**](./SPECIFICATION.md): Detailed technical requirements and architecture goals.
- [**AGENTS.md**](./AGENTS.md): Guidelines and standards for AI-driven development.

## 🚀 Status

Currently in the **Skeleton Stage**. The directory structure and core interfaces are defined. logic implementation is the next phase.

## 📜 Principles

1. **Architectural Identity**: Every module mirrors its Rust counterpart.
2. **Type Safety First**: No `Any`, no implicit conversions.
3. **Performance**: Minimal overhead between Python and the GPU.
