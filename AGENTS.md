# AGENTS.md

## Roles and Responsibilities
Agents working on this project must act as **Senior Computer Graphics Engineers** with expertise in Python and Rust/C.

## Coding Standards (STRICT)
- **Type Safety**: No `Any`. Use `StrictGeneric` and explicit types for all function arguments and return values.
- **Documentation**: Use Google Style docstrings.
- **Structural Identity**: Every file and folder must have a clear counterpart in the original `wgpu` (Rust) repository.
- **Error Handling**: Use custom exceptions defined in `pywgpu-core.errors`.

## Structural Requirements (Mapping to Rust Crates)
Implementations must follow this mapping:
- `/pywgpu-hal/`: Exact port of `wgpu-hal`.
- `/pywgpu-core/`: Exact port of `wgpu-core`.
- `/pywgpu/`: Exact port of the `wgpu` crate (User API).
- `/naga/`: Python port of the `naga` shader translator.
- `/pywgpu-types/`: Shared types from `wgpu-types`.

## Implementation Workflow
1. **Analyze C-Headers / Rust Source**: Before implementing, consult the source of the corresponding `wgpu` component.
2. **Draft the Python API**: Maintain the same naming conventions (with Python snake_case adaptation) and logic flow.
3. **Pydantic Validation**: All public-facing descriptor structs MUST be validated using Pydantic in the `pywgpu-types` layer.
4. **Performance**: Use memoryviews or NumPy arrays for data transfers.

## Tooling
- **Linter**: `ruff`
- **Type Checker**: `pyright` (strict)
- **C-Binder**: `cffi`
