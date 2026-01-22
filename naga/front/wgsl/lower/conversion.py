"""
Type conversion from WGSL AST to NAGA IR.

Translated from wgpu-trunk/naga/src/front/wgsl/lower/conversion.rs

This module handles conversion of WGSL types to NAGA IR types.
"""

from typing import Any, Optional, Tuple
from ...ir import (
    Type, TypeInner, Scalar, ScalarKind, VectorSize, 
    ArraySize, AddressSpace, StorageAccess, ImageDimension,
    ImageClass, StorageFormat
)
from ..conv import (
    get_scalar_type, map_address_space, map_storage_access,
    map_storage_format
)


class TypeConverter:
    """
    Converts WGSL AST types to NAGA IR types.
    """
    
    def __init__(self, module: Any):
        """
        Initialize type converter.
        
        Args:
            module: The module being built
        """
        self.module = module
        self.type_cache: dict[Any, Any] = {}
    
    def convert_type(self, ast_type: Any) -> Tuple[Any, TypeInner]:
        """
        Convert an AST type to IR type.
        
        Args:
            ast_type: AST type to convert
            
        Returns:
            Tuple of (type handle, type inner)
            
        Raises:
            ParseError: If type conversion fails
        """
        # Check cache
        if ast_type in self.type_cache:
            return self.type_cache[ast_type]
        
        # Handle dict-style AST types (from parser)
        type_inner = None
        if isinstance(ast_type, dict):
            type_name = ast_type.get('name', '')
            type_params = ast_type.get('params', [])
            
            # Scalar types
            scalar = get_scalar_type(set(), (0, 0), type_name)
            if scalar:
                type_inner = self.convert_scalar(scalar)
            
            # Vector types
            elif type_name.startswith('vec'):
                size_map = {'vec2': VectorSize.BI, 'vec3': VectorSize.TRI, 'vec4': VectorSize.QUAD}
                if type_name in size_map:
                    size = size_map[type_name]
                    # Get component type from params
                    if type_params:
                        comp_handle, comp_inner = self.convert_type(type_params[0])
                        # This is a bit simplified, NAGA would extract the scalar kind
                        scalar = Scalar(kind=ScalarKind.FLOAT, width=4) 
                    else:
                        scalar = Scalar(kind=ScalarKind.FLOAT, width=4)
                    
                    type_inner = self.convert_vector(size, scalar)
            
            # Matrix types
            elif type_name.startswith('mat'):
                # Parse matrix dimensions (e.g., mat3x3, mat4x4)
                type_inner = self.convert_matrix(
                    VectorSize.QUAD, 
                    VectorSize.QUAD,
                    Scalar(kind=ScalarKind.FLOAT, width=4)
                )
            
            # Array types
            elif type_name == 'array':
                if type_params:
                    base_handle, base_inner = self.convert_type(type_params[0])
                    type_inner = self.convert_array(base_handle, ArraySize.DYNAMIC)
            
            # Pointer types
            elif type_name == 'ptr':
                if len(type_params) >= 2:
                    ast_space = type_params[0].get('name') if isinstance(type_params[0], dict) else str(type_params[0])
                    space = resolve_address_space(ast_space)
                    base_handle, base_inner = self.convert_type(type_params[1])
                    type_inner = self.convert_pointer(base_handle, space)
            
            # Atomic types
            elif type_name == 'atomic':
                if type_params:
                    scalar = Scalar(kind=ScalarKind.SINT, width=4) 
                    type_inner = self.convert_atomic(scalar)
            
            # Sampler types
            elif type_name == 'sampler':
                type_inner = self.convert_sampler(comparison=False)
            
            elif type_name == 'sampler_comparison':
                type_inner = self.convert_sampler(comparison=True)
            
            # Texture types
            elif type_name.startswith('texture_'):
                dim = ImageDimension.D2 # Default
                if '_1d' in type_name: dim = ImageDimension.D1
                elif '_3d' in type_name: dim = ImageDimension.D3
                elif '_cube' in type_name: dim = ImageDimension.CUBE
                
                is_storage = 'storage' in type_name
                is_multisampled = 'multisampled' in type_name
                is_depth = 'depth' in type_name
                
                if is_storage:
                    class_ = ImageClass.STORAGE(format=StorageFormat.RGBA8_UNORM, access=StorageAccess.WRITE)
                elif is_depth:
                    class_ = ImageClass.DEPTH(multi=is_multisampled)
                else:
                    class_ = ImageClass.SAMPLED(kind=ScalarKind.FLOAT, multi=is_multisampled)
                
                type_inner = self.convert_image(dim=dim, arrayed='_array' in type_name, class_=class_)
        
        if type_inner:
            # Add to module types and return handle
            if self.module:
                # Check if this type inner already exists in module.types
                for i, existing_type in enumerate(self.module.types):
                    if existing_type.inner == type_inner:
                        result = (i, type_inner)
                        self.type_cache[ast_type] = result
                        return result
                
                # Add new type
                from ...ir import Type
                new_type = Type(name=None, inner=type_inner)
                self.module.types.append(new_type)
                handle = len(self.module.types) - 1
                result = (handle, type_inner)
                self.type_cache[ast_type] = result
                return result
            else:
                return (None, type_inner)
        
        # Unknown type
        raise NotImplementedError(f"Type conversion not implemented for: {ast_type}")
    
    def convert_scalar(self, scalar: Scalar) -> TypeInner:
        """
        Convert a scalar type.
        
        Args:
            scalar: Scalar type
            
        Returns:
            TypeInner for scalar
        """
        return TypeInner.SCALAR(scalar)
    
    def convert_vector(self, size: VectorSize, scalar: Scalar) -> TypeInner:
        """
        Convert a vector type.
        
        Args:
            size: Vector size (2, 3, or 4)
            scalar: Component scalar type
            
        Returns:
            TypeInner for vector
        """
        return TypeInner.VECTOR(size=size, scalar=scalar)
    
    def convert_matrix(
        self, 
        columns: VectorSize, 
        rows: VectorSize, 
        scalar: Scalar
    ) -> TypeInner:
        """
        Convert a matrix type.
        
        Args:
            columns: Number of columns
            rows: Number of rows
            scalar: Component scalar type
            
        Returns:
            TypeInner for matrix
        """
        return TypeInner.MATRIX(columns=columns, rows=rows, scalar=scalar)
    
    def convert_array(
        self, 
        base: Any, 
        size: ArraySize
    ) -> TypeInner:
        """
        Convert an array type.
        
        Args:
            base: Base type handle
            size: Array size (constant or dynamic)
            
        Returns:
            TypeInner for array
        """
        return TypeInner.ARRAY(base=base, size=size, stride=None)
    
    def convert_struct(self, members: list) -> TypeInner:
        """
        Convert a struct type.
        
        Args:
            members: List of struct members
            
        Returns:
            TypeInner for struct
        """
        # TODO: Convert struct members
        return TypeInner.STRUCT(members=members, span=0)
    
    def convert_pointer(
        self, 
        base: Any, 
        space: AddressSpace
    ) -> TypeInner:
        """
        Convert a pointer type.
        
        Args:
            base: Pointee type handle
            space: Address space
            
        Returns:
            TypeInner for pointer
        """
        return TypeInner.POINTER(base=base, space=space)
    
    def convert_atomic(self, scalar: Scalar) -> TypeInner:
        """
        Convert an atomic type.
        
        Args:
            scalar: Atomic scalar type
            
        Returns:
            TypeInner for atomic
        """
        return TypeInner.ATOMIC(scalar)
    
    def convert_image(
        self,
        dim: ImageDimension,
        arrayed: bool,
        class_: ImageClass
    ) -> TypeInner:
        """
        Convert an image/texture type.
        
        Args:
            dim: Image dimension (1D, 2D, 3D, Cube)
            arrayed: Whether the image is arrayed
            class_: Image class (sampled, depth, storage)
            
        Returns:
            TypeInner for image
        """
        return TypeInner.IMAGE(dim=dim, arrayed=arrayed, class_=class_)
    
    def convert_sampler(self, comparison: bool) -> TypeInner:
        """
        Convert a sampler type.
        
        Args:
            comparison: Whether this is a comparison sampler
            
        Returns:
            TypeInner for sampler
        """
        return TypeInner.SAMPLER(comparison=comparison)


def resolve_address_space(ast_space: Optional[str]) -> AddressSpace:
    """
    Resolve address space from AST.
    
    Args:
        ast_space: AST address space string or None for default
        
    Returns:
        AddressSpace enum value
    """
    if ast_space is None:
        return AddressSpace.FUNCTION
    
    return map_address_space(ast_space, (0, 0))


def resolve_storage_access(ast_access: Optional[str]) -> StorageAccess:
    """
    Resolve storage access mode from AST.
    
    Args:
        ast_access: AST access mode string or None for default
        
    Returns:
        StorageAccess flags
    """
    if ast_access is None:
        return StorageAccess.LOAD | StorageAccess.STORE
    
    return map_storage_access(ast_access, (0, 0))


def resolve_storage_format(ast_format: str) -> StorageFormat:
    """
    Resolve storage format from AST.
    
    Args:
        ast_format: AST storage format string
        
    Returns:
        StorageFormat enum value
    """
    return map_storage_format(ast_format, (0, 0))
