"""
Type generators for special types in Naga modules.

This module provides functions to generate special types used by various
language features, particularly ray tracing types.
"""

from typing import Optional
from ..arena import Handle
from ..ir import Module, Type, TypeInner, Scalar, ScalarKind, VectorSize, StructMember, ArraySize
from ..span import Span


def generate_ray_desc_type(module: Module) -> Handle[Type]:
    """
    Populate this module's SpecialTypes.ray_desc type.
    
    SpecialTypes.ray_desc is the type of the descriptor operand of
    an Initialize RayQuery statement. In WGSL, it is a struct type
    referred to as RayDesc.
    
    Backends consume values of this type to drive platform APIs, so if you
    change any its fields, you must update the backends to match.
    
    Args:
        module: The module to add the type to
        
    Returns:
        Handle to the ray_desc type
    """
    if module.special_types.ray_desc is not None:
        return module.special_types.ray_desc
    
    # Create component types
    ty_flag = module.types.insert(
        Type(name=None, inner=TypeInner.Scalar(Scalar(kind=ScalarKind.UINT, width=4))),
        Span.UNDEFINED,
    )
    ty_scalar = module.types.insert(
        Type(name=None, inner=TypeInner.Scalar(Scalar(kind=ScalarKind.FLOAT, width=4))),
        Span.UNDEFINED,
    )
    ty_vector = module.types.insert(
        Type(
            name=None,
            inner=TypeInner.Vector(
                size=VectorSize.TRI,
                scalar=Scalar(kind=ScalarKind.FLOAT, width=4),
            ),
        ),
        Span.UNDEFINED,
    )
    
    # Create the RayDesc struct
    handle = module.types.insert(
        Type(
            name="RayDesc",
            inner=TypeInner.Struct(
                members=[
                    StructMember(name="flags", ty=ty_flag, binding=None, offset=0),
                    StructMember(name="cull_mask", ty=ty_flag, binding=None, offset=4),
                    StructMember(name="tmin", ty=ty_scalar, binding=None, offset=8),
                    StructMember(name="tmax", ty=ty_scalar, binding=None, offset=12),
                    StructMember(name="origin", ty=ty_vector, binding=None, offset=16),
                    StructMember(name="dir", ty=ty_vector, binding=None, offset=32),
                ],
                span=48,
            ),
        ),
        Span.UNDEFINED,
    )
    
    module.special_types.ray_desc = handle
    return handle


def generate_vertex_return_type(module: Module) -> Handle[Type]:
    """
    Make sure the types for the vertex return are in the module's type arena.
    
    Args:
        module: The module to add the type to
        
    Returns:
        Handle to the vertex return type
    """
    if module.special_types.ray_vertex_return is not None:
        return module.special_types.ray_vertex_return
    
    ty_vec3f = module.types.insert(
        Type(
            name=None,
            inner=TypeInner.Vector(
                size=VectorSize.TRI,
                scalar=Scalar(kind=ScalarKind.FLOAT, width=4),
            ),
        ),
        Span.UNDEFINED,
    )
    array = module.types.insert(
        Type(
            name=None,
            inner=TypeInner.Array(
                base=ty_vec3f,
                size=ArraySize.Constant(3),
                stride=16,
            ),
        ),
        Span.UNDEFINED,
    )
    module.special_types.ray_vertex_return = array
    return array


def generate_ray_intersection_type(module: Module) -> Handle[Type]:
    """
    Populate this module's SpecialTypes.ray_intersection type.
    
    SpecialTypes.ray_intersection is the type of a
    RayQueryGetIntersection expression. In WGSL, it is a struct type
    referred to as RayIntersection.
    
    Backends construct values of this type based on platform APIs, so if you
    change any its fields, you must update the backends to match.
    
    Args:
        module: The module to add the type to
        
    Returns:
        Handle to the ray_intersection type
    """
    if module.special_types.ray_intersection is not None:
        return module.special_types.ray_intersection
    
    # Create component types
    ty_scalar = module.types.insert(
        Type(name=None, inner=TypeInner.Scalar(Scalar(kind=ScalarKind.FLOAT, width=4))),
        Span.UNDEFINED,
    )
    ty_uint = module.types.insert(
        Type(name=None, inner=TypeInner.Scalar(Scalar(kind=ScalarKind.UINT, width=4))),
        Span.UNDEFINED,
    )
    ty_int = module.types.insert(
        Type(name=None, inner=TypeInner.Scalar(Scalar(kind=ScalarKind.SINT, width=4))),
        Span.UNDEFINED,
    )
    ty_vector2 = module.types.insert(
        Type(
            name=None,
            inner=TypeInner.Vector(
                size=VectorSize.BI,
                scalar=Scalar(kind=ScalarKind.FLOAT, width=4),
            ),
        ),
        Span.UNDEFINED,
    )
    ty_bool = module.types.insert(
        Type(name=None, inner=TypeInner.Scalar(Scalar(kind=ScalarKind.BOOL, width=1))),
        Span.UNDEFINED,
    )
    
    # Create the RayIntersection struct
    handle = module.types.insert(
        Type(
            name="RayIntersection",
            inner=TypeInner.Struct(
                members=[
                    StructMember(name="kind", ty=ty_uint, binding=None, offset=0),
                    StructMember(name="t", ty=ty_scalar, binding=None, offset=4),
                    StructMember(name="instance_custom_index", ty=ty_uint, binding=None, offset=8),
                    StructMember(name="instance_id", ty=ty_uint, binding=None, offset=12),
                    StructMember(name="sbt_record_offset", ty=ty_uint, binding=None, offset=16),
                    StructMember(name="geometry_index", ty=ty_uint, binding=None, offset=20),
                    StructMember(name="primitive_index", ty=ty_uint, binding=None, offset=24),
                    StructMember(name="barycentrics", ty=ty_vector2, binding=None, offset=28),
                    StructMember(name="front_face", ty=ty_bool, binding=None, offset=36),
                    StructMember(name="object_to_world", ty=ty_int, binding=None, offset=40),
                    StructMember(name="world_to_object", ty=ty_int, binding=None, offset=44),
                ],
                span=48,
            ),
        ),
        Span.UNDEFINED,
    )
    
    module.special_types.ray_intersection = handle
    return handle


__all__ = [
    "generate_ray_desc_type",
    "generate_vertex_return_type",
    "generate_ray_intersection_type",
]
