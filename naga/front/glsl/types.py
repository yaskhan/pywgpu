"""GLSL frontend type parsing utilities.

This module is a Python translation of `wgpu-trunk/naga/src/front/glsl/types.rs`.
It provides helpers for recognizing GLSL type names and converting them to Naga IR
`Type`/`TypeInner` representations.

Note:
    This is intended for the GLSL frontend and is not a general-purpose GLSL type
    system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from naga.ir import (
    ImageClass,
    ImageDimension,
    Scalar,
    ScalarKind,
    StorageAccess,
    StorageFormat,
    Type,
    TypeInner,
    TypeInnerType,
    VectorSize,
)


_BOOL = Scalar(kind=ScalarKind.BOOL, width=1)
_F16 = Scalar(kind=ScalarKind.FLOAT, width=2)
_F32 = Scalar(kind=ScalarKind.FLOAT, width=4)
_F64 = Scalar(kind=ScalarKind.FLOAT, width=8)
_I32 = Scalar(kind=ScalarKind.SINT, width=4)
_U32 = Scalar(kind=ScalarKind.UINT, width=4)


def _size_parse(text: str) -> Optional[VectorSize]:
    if text == "2":
        return VectorSize.BI
    if text == "3":
        return VectorSize.TRI
    if text == "4":
        return VectorSize.QUAD
    return None


def _kind_width_parse(text: str) -> Optional[Scalar]:
    if text == "":
        return _F32
    if text == "b":
        return _BOOL
    if text == "i":
        return _I32
    if text == "u":
        return _U32
    if text == "d":
        return _F64
    if text == "f16":
        return _F16
    return None


def parse_type(type_name: str) -> Optional[Type]:
    """Parse a GLSL type name into a Naga IR `Type`.

    This mirrors `parse_type` in Rust.

    Args:
        type_name: The type name string as it appears in GLSL.

    Returns:
        A `Type` if recognized, otherwise `None`.
    """

    if type_name == "bool":
        return Type(name=None, inner=TypeInner.scalar(_BOOL))
    if type_name == "float16_t":
        return Type(name=None, inner=TypeInner.scalar(_F16))
    if type_name == "float":
        return Type(name=None, inner=TypeInner.scalar(_F32))
    if type_name == "double":
        return Type(name=None, inner=TypeInner.scalar(_F64))
    if type_name == "int":
        return Type(name=None, inner=TypeInner.scalar(_I32))
    if type_name == "uint":
        return Type(name=None, inner=TypeInner.scalar(_U32))

    if type_name in ("sampler", "samplerShadow"):
        return Type(name=None, inner=TypeInner.sampler(comparison=(type_name == "samplerShadow")))

    # Vector types
    def vec_parse(word: str) -> Optional[Type]:
        parts = word.split("vec")
        if len(parts) != 2:
            return None
        kind, size_text = parts
        scalar = _kind_width_parse(kind)
        size = _size_parse(size_text)
        if scalar is None or size is None:
            return None
        return Type(name=None, inner=TypeInner.vector(size=size, scalar=scalar))

    # Matrix types
    def mat_parse(word: str) -> Optional[Type]:
        parts = word.split("mat")
        if len(parts) != 2:
            return None
        kind, size_text = parts
        scalar = _kind_width_parse(kind)
        if scalar is None:
            return None

        square = _size_parse(size_text)
        if square is not None:
            columns = square
            rows = square
        else:
            dim_parts = size_text.split("x")
            if len(dim_parts) != 2:
                return None
            columns = _size_parse(dim_parts[0])
            rows = _size_parse(dim_parts[1])
            if columns is None or rows is None:
                return None

        return Type(name=None, inner=TypeInner.matrix(columns=columns, rows=rows, scalar=scalar))

    # Texture types
    def texture_parse(word: str) -> Optional[Type]:
        parts = word.split("texture")
        if len(parts) != 2:
            return None
        kind_text, size_text = parts

        def texture_kind(text: str) -> Optional[ScalarKind]:
            if text == "":
                return ScalarKind.FLOAT
            if text == "i":
                return ScalarKind.SINT
            if text == "u":
                return ScalarKind.UINT
            return None

        kind = texture_kind(kind_text)
        if kind is None:
            return None

        def sampled(multi: bool) -> ImageClass:
            return ImageClass.sampled(kind=kind, multi=multi)

        match size_text:
            case "1D":
                dim, arrayed, cls = ImageDimension.D1, False, sampled(False)
            case "1DArray":
                dim, arrayed, cls = ImageDimension.D1, True, sampled(False)
            case "2D":
                dim, arrayed, cls = ImageDimension.D2, False, sampled(False)
            case "2DArray":
                dim, arrayed, cls = ImageDimension.D2, True, sampled(False)
            case "2DMS":
                dim, arrayed, cls = ImageDimension.D2, False, sampled(True)
            case "2DMSArray":
                dim, arrayed, cls = ImageDimension.D2, True, sampled(True)
            case "3D":
                dim, arrayed, cls = ImageDimension.D3, False, sampled(False)
            case "Cube":
                dim, arrayed, cls = ImageDimension.CUBE, False, sampled(False)
            case "CubeArray":
                dim, arrayed, cls = ImageDimension.CUBE, True, sampled(False)
            case _:
                return None

        return Type(name=None, inner=TypeInner.image(dim=dim, arrayed=arrayed, class_=cls))

    # Storage image types
    def image_parse(word: str) -> Optional[Type]:
        parts = word.split("image")
        if len(parts) != 2:
            return None
        kind_text, size_text = parts

        def texture_kind(text: str) -> Optional[ScalarKind]:
            if text == "":
                return ScalarKind.FLOAT
            if text == "i":
                return ScalarKind.SINT
            if text == "u":
                return ScalarKind.UINT
            return None

        # The Rust implementation currently only validates that the kind prefix is one of
        # the supported ones.
        if texture_kind(kind_text) is None:
            return None

        cls = ImageClass.storage(
            format=StorageFormat.R8_UINT,
            access=StorageAccess.LOAD | StorageAccess.STORE,
        )

        # Multisampled storage images are not supported.
        match size_text:
            case "1D":
                dim, arrayed = ImageDimension.D1, False
            case "1DArray":
                dim, arrayed = ImageDimension.D1, True
            case "2D":
                dim, arrayed = ImageDimension.D2, False
            case "2DArray":
                dim, arrayed = ImageDimension.D2, True
            case "3D":
                dim, arrayed = ImageDimension.D3, False
            case _:
                return None

        return Type(name=None, inner=TypeInner.image(dim=dim, arrayed=arrayed, class_=cls))

    return (
        vec_parse(type_name)
        or mat_parse(type_name)
        or texture_parse(type_name)
        or image_parse(type_name)
    )


def scalar_components(inner: TypeInner) -> Optional[Scalar]:
    """Return the scalar component type for scalar/vector/matrix/value-pointer types."""

    if inner.type == TypeInnerType.SCALAR:
        return inner.scalar
    if inner.type == TypeInnerType.VECTOR:
        return inner.vector.scalar if inner.vector is not None else None
    if inner.type == TypeInnerType.VALUE_POINTER:
        return inner.value_pointer.scalar if inner.value_pointer is not None else None
    if inner.type == TypeInnerType.MATRIX:
        return inner.matrix.scalar if inner.matrix is not None else None
    return None


def type_power(scalar: Scalar) -> Optional[int]:
    """Return a total ordering bucket for scalar types used in conversion ranking."""

    if scalar.kind == ScalarKind.SINT:
        return 0
    if scalar.kind == ScalarKind.UINT:
        return 1
    if scalar.kind == ScalarKind.FLOAT and scalar.width == 4:
        return 2
    if scalar.kind == ScalarKind.FLOAT:
        return 3
    # Bool and abstract types are not ordered.
    if scalar.kind in (ScalarKind.BOOL, ScalarKind.ABSTRACT_INT, ScalarKind.ABSTRACT_FLOAT):
        return None
    return None


class TypeParser:
    """Compatibility wrapper matching the earlier skeleton API."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def parse_image_type(self, tokens: list[str]) -> Optional[Type]:
        """Parse an image/texture/sampler type name.

        Args:
            tokens: Token list containing the type name (first element).

        Returns:
            A Naga IR `Type` if recognized, otherwise `None`.
        """

        if not tokens:
            return None

        ty = parse_type(tokens[0])
        if ty is None:
            self.errors.append(f"Unrecognized type: {tokens[0]}")
        return ty

    def get_errors(self) -> list[str]:
        return self.errors.copy()

    def clear_errors(self) -> None:
        self.errors.clear()
