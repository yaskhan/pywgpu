"""
GLSL features management.

Contains feature flags and manager for tracking and writing
required GLSL extensions.
"""

from __future__ import annotations

from enum import IntFlag, IntEnum
from typing import Set, Optional


class Features(IntFlag):
    """Structure used to encode additions to GLSL that aren't supported by all versions."""

    BUFFER_STORAGE = 1
    ARRAY_OF_ARRAYS = 1 << 1
    DOUBLE_TYPE = 1 << 2
    FULL_IMAGE_FORMATS = 1 << 3
    MULTISAMPLED_TEXTURES = 1 << 4
    MULTISAMPLED_TEXTURE_ARRAYS = 1 << 5
    CUBE_TEXTURES_ARRAY = 1 << 6
    COMPUTE_SHADER = 1 << 7
    IMAGE_LOAD_STORE = 1 << 8
    CONSERVATIVE_DEPTH = 1 << 9
    NOPERSPECTIVE_QUALIFIER = 1 << 11
    SAMPLE_QUALIFIER = 1 << 12
    CLIP_DISTANCE = 1 << 13
    CULL_DISTANCE = 1 << 14
    SAMPLE_VARIABLES = 1 << 15
    DYNAMIC_ARRAY_SIZE = 1 << 16
    MULTI_VIEW = 1 << 17
    TEXTURE_SAMPLES = 1 << 18
    TEXTURE_LEVELS = 1 << 19
    IMAGE_SIZE = 1 << 20
    DUAL_SOURCE_BLENDING = 1 << 21
    INSTANCE_INDEX = 1 << 22
    TEXTURE_SHADOW_LOD = 1 << 23
    SUBGROUP_OPERATIONS = 1 << 24
    TEXTURE_ATOMICS = 1 << 25
    SHADER_BARYCENTRICS = 1 << 26


class WriterFlags(IntFlag):
    """Configuration flags for the GLSL Writer."""

    ADJUST_COORDINATE_SPACE = 0x1
    TEXTURE_SHADOW_LOD = 0x2
    DRAW_PARAMETERS = 0x4
    INCLUDE_UNUSED_ITEMS = 0x10
    FORCE_POINT_SIZE = 0x20


class FeaturesManager:
    """Helper structure used to store required features and write extensions."""

    def __init__(self):
        self._features: Features = Features(0)

    def request(self, features: Features) -> None:
        """Add to the list of required features."""
        self._features |= features

    def contains(self, features: Features) -> bool:
        """Check if the list of features contains the specified features."""
        return self._features.contains(features)

    def write(self, options: "Options", out: list[str]) -> None:
        """Write all needed extensions.

        Args:
            options: GLSL writer options (must have is_es(), version, is_webgl(), and writer_flags attributes)
            out: List to append extension lines to
        """
        # Duck-typed access to options attributes
        is_es = getattr(options, 'is_es', lambda: False)()
        version = getattr(options, 'version', 0)
        is_webgl = getattr(options, 'is_webgl', lambda: False)()
        writer_flags = getattr(options, 'writer_flags', None)

        def has_flag(flag):
            """Check if writer_flags contains a flag."""
            if writer_flags is None:
                return False
            return (writer_flags & flag) != 0

        if self._features.contains(Features.COMPUTE_SHADER) and not is_es:
            out.append("#extension GL_ARB_compute_shader : require")

        if self._features.contains(Features.BUFFER_STORAGE) and not is_es:
            out.append("#extension GL_ARB_shader_storage_buffer_object : require")

        if self._features.contains(Features.DOUBLE_TYPE) and version < 400:
            out.append("#extension GL_ARB_gpu_shader_fp64 : require")

        if self._features.contains(Features.CUBE_TEXTURES_ARRAY):
            if is_es:
                out.append("#extension GL_EXT_texture_cube_map_array : require")
            elif version < 400:
                out.append("#extension GL_ARB_texture_cube_map_array : require")

        if self._features.contains(Features.MULTISAMPLED_TEXTURE_ARRAYS) and is_es:
            out.append("#extension GL_OES_texture_storage_multisample_2d_array : require")

        if self._features.contains(Features.ARRAY_OF_ARRAYS) and version < 430:
            out.append("#extension ARB_arrays_of_arrays : require")

        if self._features.contains(Features.IMAGE_LOAD_STORE):
            if self._features.contains(Features.FULL_IMAGE_FORMATS) and is_es:
                out.append("#extension GL_NV_image_formats : require")
            if version < 420:
                out.append("#extension GL_ARB_shader_image_load_store : require")

        if self._features.contains(Features.CONSERVATIVE_DEPTH):
            if is_es:
                out.append("#extension GL_EXT_conservative_depth : require")
            if version < 420:
                out.append("#extension GL_ARB_conservative_depth : require")

        if (self._features.contains(Features.CLIP_DISTANCE) or self._features.contains(Features.CULL_DISTANCE)) and is_es:
            out.append("#extension GL_EXT_clip_cull_distance : require")

        if self._features.contains(Features.SAMPLE_VARIABLES) and is_es:
            out.append("#extension GL_OES_sample_variables : require")

        if self._features.contains(Features.MULTI_VIEW):
            if is_webgl:
                out.append("#extension GL_OVR_multiview2 : require")
            else:
                out.append("#extension GL_EXT_multiview : require")

        if self._features.contains(Features.TEXTURE_SAMPLES):
            out.append("#extension GL_ARB_shader_texture_image_samples : require")

        if self._features.contains(Features.TEXTURE_LEVELS) and version < 430:
            out.append("#extension GL_ARB_texture_query_levels : require")

        if self._features.contains(Features.DUAL_SOURCE_BLENDING) and is_es:
            out.append("#extension GL_EXT_blend_func_extended : require")

        if self._features.contains(Features.INSTANCE_INDEX):
            if has_flag(WriterFlags.DRAW_PARAMETERS):
                out.append("#extension GL_ARB_shader_draw_parameters : require")

        if self._features.contains(Features.TEXTURE_SHADOW_LOD):
            out.append("#extension GL_EXT_texture_shadow_lod : require")

        if self._features.contains(Features.SUBGROUP_OPERATIONS):
            out.append("#extension GL_KHR_shader_subgroup_basic : require")
            out.append("#extension GL_KHR_shader_subgroup_vote : require")
            out.append("#extension GL_KHR_shader_subgroup_arithmetic : require")
            out.append("#extension GL_KHR_shader_subgroup_ballot : require")
            out.append("#extension GL_KHR_shader_subgroup_shuffle : require")
            out.append("#extension GL_KHR_shader_subgroup_shuffle_relative : require")
            out.append("#extension GL_KHR_shader_subgroup_quad : require")

        if self._features.contains(Features.TEXTURE_ATOMICS) and is_es:
            out.append("#extension GL_OES_shader_image_atomic : require")

        if self._features.contains(Features.SHADER_BARYCENTRICS):
            out.append("#extension GL_EXT_fragment_shader_barycentric : require")


# Import Options from parent module to avoid circular imports
# This is done lazily to avoid circular import issues


__all__ = [
    "Features",
    "FeaturesManager",
    "WriterFlags",
]
