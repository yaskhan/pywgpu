"""
GLSL type offset and layout calculation.

This module calculates the offset and span for struct members according to
GLSL layout rules (std140, std430, etc.).

The layout rules are defined by the OpenGL spec in section 7.6.2.2 and use
basic machine units (equivalent to bytes).
"""

from typing import Any, Optional, List, Dict
from enum import Enum
from dataclasses import dataclass


class StructLayout(Enum):
    """Struct layout qualifiers."""
    STD140 = "std140"
    STD430 = "std430"
    SHARED = "shared"  # Not supported by SPIR-V
    PACKED = "packed"  # Implementation dependent, alias to std140


@dataclass
class TypeAlignSpan:
    """Information needed for defining a struct member."""
    ty: Any  # Handle to the type
    align: int  # Alignment required by the type
    span: int  # Size of the type


class OffsetCalculator:
    """Calculator for type offsets and spans in GLSL layouts."""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def calculate_offset(self, ty: Any, meta: Any, layout: StructLayout, types: Any) -> TypeAlignSpan:
        """
        Calculate the type, alignment and span of a struct member.
        
        Args:
            ty: Handle to the type
            meta: Metadata about the type location
            layout: Struct layout (std140, std430, etc.)
            types: Type arena/collection
            
        Returns:
            TypeAlignSpan with type, alignment and span information
        """
        # When using the std430 storage layout, shader storage blocks will be laid out in buffer storage
        # identically to uniform and shader storage blocks using the std140 layout, except
        # that the base alignment and stride of arrays of scalars and vectors in rule 4 and of
        # structures in rule 9 are not rounded up a multiple of the base alignment of a vec4.
        
        # Get the inner type information
        type_info = self._get_type_inner(ty, types)
        
        # Calculate alignment and span based on type
        if type_info['kind'] == 'scalar':
            # 1. If the member is a scalar consuming N basic machine units,
            # the base alignment is N.
            align = self._alignment_from_width(type_info['width'])
            span = type_info['width']
            
        elif type_info['kind'] == 'vector':
            # 2. If the member is a two- or four-component vector with components
            # consuming N basic machine units, the base alignment is 2N or 4N, respectively.
            # 3. If the member is a three-component vector with components consuming N
            # basic machine units, the base alignment is 4N.
            align = self._alignment_from_vector_size(type_info['size']) * self._alignment_from_width(type_info['width'])
            span = type_info['size'] * type_info['width']
            
        elif type_info['kind'] == 'array':
            # 4. If the member is an array of scalars or vectors, the base alignment and array
            # stride are set to match the base alignment of a single array element, according
            # to rules (1), (2), and (3), and rounded up to the base alignment of a vec4.
            # TODO: Matrices array
            return self._calculate_array_offset(ty, meta, layout, types)
            
        elif type_info['kind'] == 'matrix':
            # 5. If the member is a column-major matrix with C columns and R rows, the
            # matrix is stored identically to an array of C column vectors with R
            # components each, according to rule (4)
            # TODO: Row major matrices
            return self._calculate_matrix_offset(ty, meta, layout, types)
            
        elif type_info['kind'] == 'struct':
            return self._calculate_struct_offset(ty, meta, layout, types)
            
        else:
            # Invalid struct member type
            self.errors.append(f"Invalid struct member type: {type_info['kind']}")
            align = 1  # Minimum alignment
            span = 0
        
        return TypeAlignSpan(ty=ty, align=align, span=span)
    
    def _get_type_inner(self, ty: Any, types: Any) -> Dict[str, Any]:
        """Get inner type information."""
        # TODO: Implement type inner information extraction
        # This should extract the type kind and parameters
        # from the type handle
        
        return {'kind': 'scalar', 'width': 4, 'size': 1}
    
    def _alignment_from_width(self, width: int) -> int:
        """Calculate alignment from width."""
        return width
    
    def _alignment_from_vector_size(self, size: int) -> int:
        """Calculate alignment from vector size."""
        if size == 2:
            return 2
        elif size == 3:
            return 4  # Three-component vectors have 4N alignment
        elif size == 4:
            return 4
        else:
            return size
    
    def _calculate_array_offset(self, ty: Any, meta: Any, layout: StructLayout, types: Any) -> TypeAlignSpan:
        """
        Calculate offset for array types.
        
        Args:
            ty: Array type handle
            meta: Metadata
            layout: Struct layout
            types: Type collection
            
        Returns:
            TypeAlignSpan for the array
        """
        # TODO: Matrices array
        # This TODO is about handling arrays of matrices specifically.
        # The current implementation handles arrays of scalars and vectors,
        # but arrays of matrices need special handling.
        
        # Get array base type and size
        array_info = self._get_array_info(ty, types)
        base = array_info['base']
        size = array_info['size']
        
        # Calculate offset for base type
        base_info = self.calculate_offset(base, meta, layout, types)
        
        # Calculate stride
        # See comment at the beginning of the function
        if layout == StructLayout.STD430:
            align = base_info.align
            stride = self._round_up_align(base_info.align, base_info.span)
        else:
            align = max(base_info.align, self._min_uniform_alignment())
            stride = self._round_up_align(align, base_info.span)
        
        # Calculate total span
        if size == 'dynamic':
            span = stride
        elif isinstance(size, int):
            span = size * stride
        else:
            span = stride
        
        return TypeAlignSpan(ty=ty, align=align, span=span)
    
    def _calculate_matrix_offset(self, ty: Any, meta: Any, layout: StructLayout, types: Any) -> TypeAlignSpan:
        """
        Calculate offset for matrix types.
        
        Args:
            ty: Matrix type handle
            meta: Metadata
            layout: Struct layout
            types: Type collection
            
        Returns:
            TypeAlignSpan for the matrix
        """
        # TODO: Row major matrices
        # This TODO is about supporting row-major matrices.
        # Currently only column-major matrices are supported.
        # Row-major matrices are stored differently in memory.
        
        # Get matrix information
        matrix_info = self._get_matrix_info(ty, types)
        columns = matrix_info['columns']
        rows = matrix_info['rows']
        scalar = matrix_info['scalar']
        
        # Calculate alignment for column-major matrix
        align = self._alignment_from_vector_size(rows) * self._alignment_from_width(scalar['width'])
        
        # See comment at the beginning of the function
        if layout != StructLayout.STD430:
            align = max(align, self._min_uniform_alignment())
        
        # Check for unsupported matrix types in std140 layout
        if layout == StructLayout.STD140:
            if scalar['kind'] == 'f16':
                self.errors.append(f"Unsupported f16 matrix in std140 layout: {columns}x{rows}")
            if rows == 2:  # Bi rows (2-component vectors)
                self.errors.append(f"Unsupported matrix with two rows in std140 layout: {columns} columns")
        
        span = align * columns
        
        return TypeAlignSpan(ty=ty, align=align, span=span)
    
    def _calculate_struct_offset(self, ty: Any, meta: Any, layout: StructLayout, types: Any) -> TypeAlignSpan:
        """
        Calculate offset for struct types.
        
        Args:
            ty: Struct type handle
            meta: Metadata
            layout: Struct layout
            types: Type collection
            
        Returns:
            TypeAlignSpan for the struct
        """
        span = 0
        align = 1
        
        # Get struct members
        struct_info = self._get_struct_info(ty, types)
        members = struct_info['members']
        
        calculated_members = []
        
        for member in members:
            # Calculate offset for each member
            member_info = self.calculate_offset(member['ty'], meta, layout, types)
            
            member_alignment = member_info.align
            span = self._round_up_align(member_alignment, span)
            align = max(member_alignment, align)
            
            # Update member with calculated offset
            calculated_member = member.copy()
            calculated_member['ty'] = member_info.ty
            calculated_member['offset'] = span
            calculated_members.append(calculated_member)
            
            span += member_info.span
        
        span = self._round_up_align(align, span)
        
        return TypeAlignSpan(ty=ty, align=align, span=span)
    
    def _get_array_info(self, ty: Any, types: Any) -> Dict[str, Any]:
        """Get array type information."""
        # TODO: Implement array info extraction
        return {'base': ty, 'size': 1}
    
    def _get_matrix_info(self, ty: Any, types: Any) -> Dict[str, Any]:
        """Get matrix type information."""
        # TODO: Implement matrix info extraction
        return {'columns': 4, 'rows': 4, 'scalar': {'kind': 'f32', 'width': 4}}
    
    def _get_struct_info(self, ty: Any, types: Any) -> Dict[str, Any]:
        """Get struct type information."""
        # TODO: Implement struct info extraction
        return {'members': []}
    
    def _round_up_align(self, align: int, span: int) -> int:
        """Round up span to alignment boundary."""
        return ((span + align - 1) // align) * align
    
    def _min_uniform_alignment(self) -> int:
        """Get minimum uniform alignment (vec4 alignment)."""
        return 16  # vec4 alignment
    
    def get_errors(self) -> List[str]:
        """Get accumulated errors."""
        return self.errors.copy()
    
    def clear_errors(self) -> None:
        """Clear accumulated errors."""
        self.errors.clear()