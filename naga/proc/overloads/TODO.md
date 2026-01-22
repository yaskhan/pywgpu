# Naga Overload Resolution Implementation Task

This directory is a Python port of `naga/src/proc/overloads` from the Rust `wgpu` repository.
It implements overload resolution for builtin functions in Naga IR.

## Progress Tracking

- [x] `rule.py`: `Rule` and `Conclusion` types. (COMPLETED)
- [x] `overload_set.py`: `OverloadSet` Protocol (trait) definition. (COMPLETED)
- [x] `constructor_set.py`: `ConstructorSet` implementation. (COMPLETED)
- [x] `regular.py`: `Regular` overload set implementation. (COMPLETED)
- [x] `list.py`: `List` overload set implementation. (COMPLETED)
- [x] `any_overload_set.py`: `AnyOverloadSet` enum/union. (COMPLETED)
- [x] `scalar_set.py`: `ScalarSet` utility. (COMPLETED)
- [x] `mathfunction.py`: Math function overloads registry. (COMPLETED)
- [x] `utils.py`: Helpers for constructing overload sets. (COMPLETED)
- [x] `one_bits_iter.py`: Bit manipulation utility. (COMPLETED)
- [x] `__init__.py`: Package exports. (COMPLETED)

## Implementation Notes

The implementation follows the Rust source in `wgpu-trunk/naga/src/proc/overloads/` as closely as possible, adapting Rust's `bitflags` and `enum` patterns to Python's `IntFlag` and `Union` of dataclasses.

- `OverloadSet` is implemented as a `typing.Protocol`.
- `Regular` and `ListOverloadSet` (renamed from `List`) implement `OverloadSet`.
- `ConstructorSet` and `ScalarSet` use `enum.IntFlag` to mimic Rust's `bitflags`.
- `Conclusion` is a `Union` of `ConclusionValue` and `ConclusionPredeclared`.
- `AnyOverloadSet` is a `Union` of `Regular` and `ListOverloadSet`.
- `get_math_function_overloads` provides the registry mapping `ir.MathFunction` to its `OverloadSet`.
