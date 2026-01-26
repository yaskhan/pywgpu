"""
WGSL polyfills.

Provides polyfill functions for WGSL operations that may not be directly supported
or need specialized implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...ir.type import TypeInner


@dataclass
class InversePolyfill:
    """Matrix inverse polyfill."""

    fun_name: str
    """Name of the polyfill function."""

    source: str
    """WGSL source code for the polyfill."""

    @staticmethod
    def find_overload(ty: TypeInner) -> Optional["InversePolyfill"]:
        """Find inverse polyfill overload for a matrix type.

        Args:
            ty: The type inner to find overload for

        Returns:
            InversePolyfill if applicable, None otherwise
        """
        from ...ir.type import TypeInnerType, ScalarKind, VectorSize

        if ty.type != TypeInnerType.MATRIX:
            return None

        mat = ty.matrix
        columns = mat.columns
        rows = mat.rows
        scalar = mat.scalar

        # Only for square matrices with float scalars
        if columns != rows or scalar.kind != ScalarKind.FLOAT:
            return None

        return InversePolyfill.polyfill_overload(columns, scalar.width)

    @staticmethod
    def polyfill_overload(dimension: VectorSize, width: int) -> Optional["InversePolyfill"]:
        """Get polyfill overload for a specific matrix dimension and width.

        Args:
            dimension: Matrix dimension (2x2, 3x3, 4x4)
            width: Scalar width in bytes

        Returns:
            InversePolyfill if applicable, None otherwise
        """
        from ...ir.type import VectorSize

        match (dimension, width):
            case (VectorSize.Bi, 4):
                # 2x2 f32 matrix
                return InversePolyfill(
                    fun_name="_naga_inverse_2x2_f32",
                    source=INVERSE_2X2_F32,
                )
            case (VectorSize.Tri, 4):
                # 3x3 f32 matrix
                return InversePolyfill(
                    fun_name="_naga_inverse_3x3_f32",
                    source=INVERSE_3X3_F32,
                )
            case (VectorSize.Quad, 4):
                # 4x4 f32 matrix
                return InversePolyfill(
                    fun_name="_naga_inverse_4x4_f32",
                    source=INVERSE_4X4_F32,
                )
            case (VectorSize.Bi, 2):
                # 2x2 f16 matrix
                return InversePolyfill(
                    fun_name="_naga_inverse_2x2_f16",
                    source=INVERSE_2X2_F16,
                )
            case (VectorSize.Tri, 2):
                # 3x3 f16 matrix
                return InversePolyfill(
                    fun_name="_naga_inverse_3x3_f16",
                    source=INVERSE_3X3_F16,
                )
            case (VectorSize.Quad, 2):
                # 4x4 f16 matrix
                return InversePolyfill(
                    fun_name="_naga_inverse_4x4_f16",
                    source=INVERSE_4X4_F16,
                )
            case _:
                return None


# Polyfill source code
# 2x2 f32 matrix inverse
INVERSE_2X2_F32 = """
fn _naga_inverse_2x2_f32(m: mat2x2<f32>) -> mat2x2<f32> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let inv_det = 1.0 / det;
    return mat2x2<f32>(
        m[1][1] * inv_det,
        -m[0][1] * inv_det,
        -m[1][0] * inv_det,
        m[0][0] * inv_det
    );
}
"""

# 3x3 f32 matrix inverse
INVERSE_3X3_F32 = """
fn _naga_inverse_3x3_f32(m: mat3x3<f32>) -> mat3x3<f32> {
    let a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    let a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    let a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a01 * a12 - a02 * a11;
    let b10 = a10 * a21 - a11 * a20;
    let b11 = a10 * a22 - a12 * a20;
    let b12 = a11 * a22 - a12 * a21;

    let det = b00 * a22 - b01 * a21 + b02 * a20;
    let inv_det = 1.0 / det;

    return mat3x3<f32>(
        (a11 * a22 - a12 * a21) * inv_det,
        (a02 * a21 - a01 * a22) * inv_det,
        (a01 * a12 - a02 * a11) * inv_det,
        (a12 * a20 - a10 * a22) * inv_det,
        (a00 * a22 - a02 * a20) * inv_det,
        (a02 * a10 - a00 * a12) * inv_det,
        (a10 * a21 - a11 * a20) * inv_det,
        (a01 * a20 - a00 * a21) * inv_det,
        (a00 * a11 - a01 * a10) * inv_det
    );
}
"""

# 4x4 f32 matrix inverse
INVERSE_4X4_F32 = """
fn _naga_inverse_4x4_f32(m: mat4x4<f32>) -> mat4x4<f32> {
    let a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    let a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    let a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    let a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let inv_det = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);

    let c00 = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
    let c01 = (a02 * b10 - a01 * b11 - a03 * b09) * inv_det;
    let c02 = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
    let c03 = (a22 * b04 - a21 * b05 - a23 * b03) * inv_det;
    let c10 = (a12 * b08 - a10 * b11 - a13 * b07) * inv_det;
    let c11 = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
    let c12 = (a32 * b02 - a30 * b05 - a33 * b01) * inv_det;
    let c13 = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;
    let c20 = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
    let c21 = (a01 * b08 - a00 * b10 - a03 * b06) * inv_det;
    let c22 = (a30 * b05 - a31 * b02 + a33 * b00) * inv_det;
    let c23 = (a21 * b02 - a20 * b05 - a23 * b00) * inv_det;
    let c30 = (a11 * b07 - a10 * b09 - a12 * b06) * inv_det;
    let c31 = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
    let c32 = (a31 * b01 - a30 * b02 - a32 * b00) * inv_det;
    let c33 = (a20 * b02 - a21 * b01 + a22 * b00) * inv_det;

    return mat4x4<f32>(
        c00, c01, c02, c03,
        c10, c11, c12, c13,
        c20, c21, c22, c23,
        c30, c31, c32, c33
    );
}
"""

# 2x2 f16 matrix inverse
INVERSE_2X2_F16 = """
fn _naga_inverse_2x2_f16(m: mat2x2<f16>) -> mat2x2<f16> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let inv_det = f16(1.0) / det;
    return mat2x2<f16>(
        m[1][1] * inv_det,
        -m[0][1] * inv_det,
        -m[1][0] * inv_det,
        m[0][0] * inv_det
    );
}
"""

# 3x3 f16 matrix inverse
INVERSE_3X3_F16 = """
fn _naga_inverse_3x3_f16(m: mat3x3<f16>) -> mat3x3<f16> {
    let a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    let a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    let a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a01 * a12 - a02 * a11;
    let b10 = a10 * a21 - a11 * a20;
    let b11 = a10 * a22 - a12 * a20;
    let b12 = a11 * a22 - a12 * a21;

    let det = b00 * a22 - b01 * a21 + b02 * a20;
    let inv_det = f16(1.0) / det;

    return mat3x3<f16>(
        (a11 * a22 - a12 * a21) * inv_det,
        (a02 * a21 - a01 * a22) * inv_det,
        (a01 * a12 - a02 * a11) * inv_det,
        (a12 * a20 - a10 * a22) * inv_det,
        (a00 * a22 - a02 * a20) * inv_det,
        (a02 * a10 - a00 * a12) * inv_det,
        (a10 * a21 - a11 * a20) * inv_det,
        (a01 * a20 - a00 * a21) * inv_det,
        (a00 * a11 - a01 * a10) * inv_det
    );
}
"""

# 4x4 f16 matrix inverse
INVERSE_4X4_F16 = """
fn _naga_inverse_4x4_f16(m: mat4x4<f16>) -> mat4x4<f16> {
    let a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    let a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    let a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    let a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let inv_det = f16(1.0) / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);

    let c00 = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
    let c01 = (a02 * b10 - a01 * b11 - a03 * b09) * inv_det;
    let c02 = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
    let c03 = (a22 * b04 - a21 * b05 - a23 * b03) * inv_det;
    let c10 = (a12 * b08 - a10 * b11 - a13 * b07) * inv_det;
    let c11 = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
    let c12 = (a32 * b02 - a30 * b05 - a33 * b01) * inv_det;
    let c13 = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;
    let c20 = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
    let c21 = (a01 * b08 - a00 * b10 - a03 * b06) * inv_det;
    let c22 = (a30 * b05 - a31 * b02 + a33 * b00) * inv_det;
    let c23 = (a21 * b02 - a20 * b05 - a23 * b00) * inv_det;
    let c30 = (a11 * b07 - a10 * b09 - a12 * b06) * inv_det;
    let c31 = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
    let c32 = (a31 * b01 - a30 * b02 - a32 * b00) * inv_det;
    let c33 = (a20 * b02 - a21 * b01 + a22 * b00) * inv_det;

    return mat4x4<f16>(
        c00, c01, c02, c03,
        c10, c11, c12, c13,
        c20, c21, c22, c23,
        c30, c31, c32, c33
    );
}
"""


__all__ = [
    "InversePolyfill",
    "find_inverse_polyfill",
    "INVERSE_2X2_F32",
    "INVERSE_3X3_F32",
    "INVERSE_4X4_F32",
    "INVERSE_2X2_F16",
    "INVERSE_3X3_F16",
    "INVERSE_4X4_F16",
]


def find_inverse_polyfill(ty: TypeInner) -> Optional[InversePolyfill]:
    """Find inverse polyfill for a type.

    Args:
        ty: Type inner to find polyfill for

    Returns:
        Inverse polyfill if found, None otherwise
    """
    return InversePolyfill.find_overload(ty)
