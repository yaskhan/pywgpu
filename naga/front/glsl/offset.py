"""GLSL std140/std430 layout offset calculation.

This module is a Python translation of `wgpu-trunk/naga/src/front/glsl/offset.rs`.

It provides `calculate_offset`, which computes per-member alignment and size
information and, for arrays/structs, returns an updated type handle whose
stride/span reflect the chosen layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from naga.arena import Handle, UniqueArena
from naga.ir import ArraySizeType, ScalarKind, StructMember, Type, TypeInner, TypeInnerType, VectorSize
from naga.proc.layouter import Alignment
from naga.span import Span

from .ast import StructLayout


@dataclass(frozen=True, slots=True)
class TypeAlignSpan:
    """Type handle plus alignment and span information."""

    ty: Handle[Type]
    align: int
    span: int


def _vector_alignment(width: int, size: VectorSize) -> int:
    if size == VectorSize.BI:
        return 2 * width
    # Tri and Quad both have 4N alignment.
    return 4 * width


def calculate_offset(
    ty: Handle[Type],
    meta: Span,
    layout: StructLayout,
    types: UniqueArena[Type],
    errors: list[str],
) -> TypeAlignSpan:
    """Calculate alignment and span for a type under a given struct layout."""

    inner = types[ty].inner

    # 1. Scalar
    if inner.type == TypeInnerType.SCALAR and inner.scalar is not None:
        width = inner.scalar.width
        return TypeAlignSpan(ty=ty, align=Alignment.from_width(width), span=width)

    # 2/3. Vector
    if inner.type == TypeInnerType.VECTOR and inner.vector is not None:
        width = inner.vector.scalar.width
        align = _vector_alignment(width, inner.vector.size)
        span = inner.vector.size.value * width
        return TypeAlignSpan(ty=ty, align=align, span=span)

    # 4. Array
    if inner.type == TypeInnerType.ARRAY and inner.array is not None:
        # Note: Arrays of matrices are handled indirectly, because matrices have their
        # own rule (5) and array base computation will account for that.
        base_handle = Handle.from_usize(inner.array.base)
        info = calculate_offset(base_handle, meta, layout, types, errors)

        name = types[ty].name

        if layout == StructLayout.STD430:
            align = info.align
            stride = Alignment.round_up(info.align, info.span)
        else:
            align = max(info.align, Alignment.MIN_UNIFORM)
            stride = Alignment.round_up(align, info.span)

        if inner.array.size.type == ArraySizeType.CONSTANT and inner.array.size.constant is not None:
            count = inner.array.size.constant.value
            span = count * stride
        elif inner.array.size.type == ArraySizeType.DYNAMIC:
            span = stride
        else:
            # Pending/unknown sizes should not appear here (Rust uses unreachable!).
            span = stride

        ty_span = types.get_span(ty)
        new_ty = types.insert(
            Type(
                name=name,
                inner=TypeInner.array(
                    base=info.ty.index,
                    size=inner.array.size,
                    stride=stride,
                ),
            ),
            ty_span,
        )

        return TypeAlignSpan(ty=new_ty, align=align, span=span)

    # 5. Column-major matrix
    if inner.type == TypeInnerType.MATRIX and inner.matrix is not None:
        m = inner.matrix
        width = m.scalar.width
        align = _vector_alignment(width, m.rows)

        if layout != StructLayout.STD430:
            align = max(align, Alignment.MIN_UNIFORM)

        if layout == StructLayout.STD140:
            if m.scalar.kind == ScalarKind.FLOAT and m.scalar.width == 2:
                errors.append(
                    f"Unsupported f16 matrix in std140 layout: columns={m.columns.value}, rows={m.rows.value} at {meta}"
                )
            if m.rows == VectorSize.BI:
                errors.append(
                    f"Unsupported matrix with two rows in std140 layout: columns={m.columns.value} at {meta}"
                )

        span = align * m.columns.value
        return TypeAlignSpan(ty=ty, align=align, span=span)

    # 9. Struct
    if inner.type == TypeInnerType.STRUCT and inner.struct is not None:
        span = 0
        align = Alignment.ONE
        members: list[StructMember] = []

        name = types[ty].name

        for member in inner.struct.members:
            info = calculate_offset(Handle.from_usize(member.ty), meta, layout, types, errors)
            member_alignment = info.align
            span = Alignment.round_up(member_alignment, span)
            align = max(member_alignment, align)

            members.append(
                StructMember(
                    name=member.name,
                    ty=info.ty.index,
                    binding=member.binding,
                    offset=span,
                )
            )

            span += info.span

        span = Alignment.round_up(align, span)

        ty_span = types.get_span(ty)
        new_ty = types.insert(
            Type(
                name=name,
                inner=TypeInner.struct_(members=members, span=span),
            ),
            ty_span,
        )

        return TypeAlignSpan(ty=new_ty, align=align, span=span)

    errors.append(f"Invalid struct member type at {meta}")
    return TypeAlignSpan(ty=ty, align=Alignment.ONE, span=0)


class OffsetCalculator:
    """Compatibility wrapper matching the earlier skeleton API."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def calculate_offset(
        self,
        ty: Handle[Type],
        meta: Span,
        layout: StructLayout,
        types: UniqueArena[Type],
    ) -> TypeAlignSpan:
        return calculate_offset(ty, meta, layout, types, self.errors)

    def get_errors(self) -> list[str]:
        return self.errors.copy()

    def clear_errors(self) -> None:
        self.errors.clear()
