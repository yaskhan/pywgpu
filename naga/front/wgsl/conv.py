"""
WGSL type and keyword conversion utilities.

Translated from wgpu-trunk/naga/src/front/wgsl/parse/conv.rs

This module provides conversion functions for mapping WGSL keywords
and identifiers to their NAGA IR equivalents.
"""

from typing import Optional, Tuple
from ...ir import Scalar, ScalarKind, BuiltIn, Interpolation, Sampling, StorageFormat
from ...ir import AddressSpace, StorageAccess


def get_scalar_type(extensions: set, span: Tuple[int, int], word: str) -> Optional[Scalar]:
    """
    Get scalar type from WGSL keyword.
    
    Args:
        extensions: Set of enabled extensions
        span: Source location for error reporting
        word: WGSL type keyword
        
    Returns:
        Scalar type or None if not a scalar type keyword
    """
    scalar_map = {
        'bool': Scalar(kind=ScalarKind.BOOL, width=1),
        'i32': Scalar(kind=ScalarKind.SINT, width=4),
        'u32': Scalar(kind=ScalarKind.UINT, width=4),
        'f32': Scalar(kind=ScalarKind.FLOAT, width=4),
    }
    
    # f16 requires extension
    if word == 'f16':
        # TODO: Check if f16 extension is enabled
        return Scalar(kind=ScalarKind.FLOAT, width=2)
    
    return scalar_map.get(word)


def map_built_in(extensions: set, word: str, span: Tuple[int, int]) -> BuiltIn:
    """
    Map WGSL built-in name to BuiltIn enum.
    
    Args:
        extensions: Set of enabled extensions
        word: Built-in name
        span: Source location for error reporting
        
    Returns:
        BuiltIn value
        
    Raises:
        ParseError: If built-in name is unknown
    """
    builtin_map = {
        # Vertex shader inputs
        'vertex_index': BuiltIn.VERTEX_INDEX,
        'instance_index': BuiltIn.INSTANCE_INDEX,
        'base_vertex': BuiltIn.BASE_VERTEX,
        'base_instance': BuiltIn.BASE_INSTANCE,
        'draw_id': BuiltIn.DRAW_ID,
        
        # Vertex shader outputs / Fragment shader inputs
        'position': BuiltIn.POSITION,
        'clip_distance': BuiltIn.CLIP_DISTANCE,
        'cull_distance': BuiltIn.CULL_DISTANCE,
        'point_size': BuiltIn.POINT_SIZE,
        
        # Fragment shader inputs
        'front_facing': BuiltIn.FRONT_FACING,
        'frag_depth': BuiltIn.FRAG_DEPTH,
        'primitive_index': BuiltIn.PRIMITIVE_INDEX,
        'sample_index': BuiltIn.SAMPLE_INDEX,
        'sample_mask': BuiltIn.SAMPLE_MASK,
        
        # Compute shader inputs
        'global_invocation_id': BuiltIn.GLOBAL_INVOCATION_ID,
        'local_invocation_id': BuiltIn.LOCAL_INVOCATION_ID,
        'local_invocation_index': BuiltIn.LOCAL_INVOCATION_INDEX,
        'workgroup_id': BuiltIn.WORKGROUP_ID,
        'workgroup_size': BuiltIn.WORKGROUP_SIZE,
        'num_workgroups': BuiltIn.NUM_WORKGROUPS,
        
        # Subgroup operations
        'num_subgroups': BuiltIn.NUM_SUBGROUPS,
        'subgroup_id': BuiltIn.SUBGROUP_ID,
        'subgroup_size': BuiltIn.SUBGROUP_SIZE,
        'subgroup_invocation_id': BuiltIn.SUBGROUP_INVOCATION_ID,
        
        # View index
        'view_index': BuiltIn.VIEW_INDEX,
    }
    
    if word not in builtin_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown built-in: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return builtin_map[word]


def map_interpolation(word: str, span: Tuple[int, int]) -> Interpolation:
    """
    Map WGSL interpolation mode to Interpolation enum.
    
    Args:
        word: Interpolation mode name
        span: Source location for error reporting
        
    Returns:
        Interpolation value
        
    Raises:
        ParseError: If interpolation mode is unknown
    """
    interpolation_map = {
        'perspective': Interpolation.PERSPECTIVE,
        'linear': Interpolation.LINEAR,
        'flat': Interpolation.FLAT,
    }
    
    if word not in interpolation_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown interpolation: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return interpolation_map[word]


def map_sampling(word: str, span: Tuple[int, int]) -> Sampling:
    """
    Map WGSL sampling mode to Sampling enum.
    
    Args:
        word: Sampling mode name
        span: Source location for error reporting
        
    Returns:
        Sampling value
        
    Raises:
        ParseError: If sampling mode is unknown
    """
    sampling_map = {
        'center': Sampling.CENTER,
        'centroid': Sampling.CENTROID,
        'sample': Sampling.SAMPLE,
        'first': Sampling.FIRST,
        'either': Sampling.EITHER,
    }
    
    if word not in sampling_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown sampling: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return sampling_map[word]


def map_address_space(word: str, span: Tuple[int, int]) -> AddressSpace:
    """
    Map WGSL address space to AddressSpace enum.
    
    Args:
        word: Address space name
        span: Source location for error reporting
        
    Returns:
        AddressSpace value
        
    Raises:
        ParseError: If address space is unknown
    """
    address_space_map = {
        'function': AddressSpace.FUNCTION,
        'private': AddressSpace.PRIVATE,
        'workgroup': AddressSpace.WORKGROUP,
        'uniform': AddressSpace.UNIFORM,
        'storage': AddressSpace.STORAGE,
        'handle': AddressSpace.HANDLE,
    }
    
    if word not in address_space_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown address space: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return address_space_map[word]


def map_storage_access(word: str, span: Tuple[int, int]) -> StorageAccess:
    """
    Map WGSL storage access mode to StorageAccess flags.
    
    Args:
        word: Access mode name
        span: Source location for error reporting
        
    Returns:
        StorageAccess flags
        
    Raises:
        ParseError: If access mode is unknown
    """
    access_map = {
        'read': StorageAccess.LOAD,
        'write': StorageAccess.STORE,
        'read_write': StorageAccess.LOAD | StorageAccess.STORE,
    }
    
    if word not in access_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown access mode: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return access_map[word]


def map_storage_format(word: str, span: Tuple[int, int]) -> StorageFormat:
    """
    Map WGSL storage format to StorageFormat enum.
    
    Args:
        word: Storage format name
        span: Source location for error reporting
        
    Returns:
        StorageFormat value
        
    Raises:
        ParseError: If storage format is unknown
    """
    format_map = {
        # 8-bit formats
        'r8unorm': StorageFormat.R8_UNORM,
        'r8snorm': StorageFormat.R8_SNORM,
        'r8uint': StorageFormat.R8_UINT,
        'r8sint': StorageFormat.R8_SINT,
        
        # 16-bit formats
        'r16uint': StorageFormat.R16_UINT,
        'r16sint': StorageFormat.R16_SINT,
        'r16float': StorageFormat.R16_FLOAT,
        'rg8unorm': StorageFormat.RG8_UNORM,
        'rg8snorm': StorageFormat.RG8_SNORM,
        'rg8uint': StorageFormat.RG8_UINT,
        'rg8sint': StorageFormat.RG8_SINT,
        
        # 32-bit formats
        'r32uint': StorageFormat.R32_UINT,
        'r32sint': StorageFormat.R32_SINT,
        'r32float': StorageFormat.R32_FLOAT,
        'rg16uint': StorageFormat.RG16_UINT,
        'rg16sint': StorageFormat.RG16_SINT,
        'rg16float': StorageFormat.RG16_FLOAT,
        'rgba8unorm': StorageFormat.RGBA8_UNORM,
        'rgba8snorm': StorageFormat.RGBA8_SNORM,
        'rgba8uint': StorageFormat.RGBA8_UINT,
        'rgba8sint': StorageFormat.RGBA8_SINT,
        'bgra8unorm': StorageFormat.BGRA8_UNORM,
        
        # Packed 32-bit formats
        'rgb10a2uint': StorageFormat.RGB10A2_UINT,
        'rgb10a2unorm': StorageFormat.RGB10A2_UNORM,
        'rg11b10ufloat': StorageFormat.RG11B10_UFLOAT,
        
        # 64-bit formats
        'rg32uint': StorageFormat.RG32_UINT,
        'rg32sint': StorageFormat.RG32_SINT,
        'rg32float': StorageFormat.RG32_FLOAT,
        'rgba16uint': StorageFormat.RGBA16_UINT,
        'rgba16sint': StorageFormat.RGBA16_SINT,
        'rgba16float': StorageFormat.RGBA16_FLOAT,
        
        # 128-bit formats
        'rgba32uint': StorageFormat.RGBA32_UINT,
        'rgba32sint': StorageFormat.RGBA32_SINT,
        'rgba32float': StorageFormat.RGBA32_FLOAT,
    }
    
    if word not in format_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown storage format: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return format_map[word]


def map_conservative_depth(word: str, span: Tuple[int, int]) -> str:
    """
    Map WGSL conservative depth mode.
    
    Args:
        word: Conservative depth mode name
        span: Source location for error reporting
        
    Returns:
        Conservative depth mode string
        
    Raises:
        ParseError: If mode is unknown
    """
    depth_map = {
        'greater_equal': 'greater_equal',
        'less_equal': 'less_equal',
        'unchanged': 'unchanged',
    }
    
    if word not in depth_map:
        from .error import ParseError
        raise ParseError(
            message=f"unknown conservative depth: '{word}'",
            labels=[(span[0], span[1], "")],
            notes=[]
        )
    
    return depth_map[word]
