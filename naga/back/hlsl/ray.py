"""naga.back.hlsl.ray

A port of `wgpu-trunk/naga/src/back/hlsl/ray.rs`.

This module implements the helper routines used by the HLSL backend to support
WGSL ray queries.

In upstream Naga, these are methods on `back::hlsl::Writer`. In this Python
port, they are implemented as methods on :class:`~RayWriter`, which is intended
to be used as a mixin by a full HLSL writer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING, TextIO

from .. import Baked, Level, RayQueryPoint

if TYPE_CHECKING:
    from ...arena import Handle
    from ...ir import Expression, Module
    from .. import FunctionCtx


class RayFlag(IntFlag):
    """Ray flags used when casting rays.

    This matches `crate::RayFlag` in upstream Naga (IR-level ray flags).

    Note: these are distinct from `naga.back.RayFlag` which currently mirrors the
    backend-side `back::RayFlag`.
    """

    FORCE_OPAQUE = 0x1
    FORCE_NO_OPAQUE = 0x2
    TERMINATE_ON_FIRST_HIT = 0x4
    SKIP_CLOSEST_HIT_SHADER = 0x8
    CULL_BACK_FACING = 0x10
    CULL_FRONT_FACING = 0x20
    CULL_OPAQUE = 0x40
    CULL_NO_OPAQUE = 0x80
    SKIP_TRIANGLES = 0x100
    SKIP_AABBS = 0x200


class RayQueryIntersection(IntEnum):
    """Type of a ray query intersection.

    Matches `crate::RayQueryIntersection` in upstream Naga.
    """

    NONE = 0
    TRIANGLE = 1
    GENERATED = 2
    AABB = 3


@dataclass(slots=True)
class RayWriterOptions:
    """Subset of HLSL writer options used by ray helpers."""

    ray_query_initialization_tracking: bool = False


class RayWriter:
    """Implements `wgpu-trunk/naga/src/back/hlsl/ray.rs`.

    The full HLSL backend is expected to provide the following members:

    - ``out``: a text writer
    - ``options``: an options object with ``ray_query_initialization_tracking``
    - ``named_expressions``: a mapping used to assign stable names to baked
      expressions
    - ``write_expr(module, expr, func_ctx)``
    - ``write_type(module, ty_handle)``
    - ``write_value_type(module, type_inner)``

    This module provides the ray query helper functions without committing to a
    particular writer implementation.
    """

    out: TextIO
    options: RayWriterOptions

    def __init__(self, out: TextIO, options: RayWriterOptions) -> None:
        self.out = out
        self.options = options
        self.named_expressions: dict[object, str] = {}

    # https://sakibsaikia.github.io/graphics/2022/01/04/Nan-Checks-In-HLSL.html
    # suggests that isnan may not work, unsure if this has changed.
    def write_not_finite(self, expr: str) -> None:
        self.write_contains_flags(f"asuint({expr})", 0x7F800000)

    def write_nan(self, expr: str) -> None:
        self.out.write("(")
        self.write_not_finite(expr)
        self.out.write(f" && ((asuint({expr}) & 0x7fffff) != 0))")

    def write_contains_flags(self, expr: str, flags: int) -> None:
        self.out.write(f"(({expr} & {flags}) == {flags})")

    # constructs hlsl RayDesc from wgsl RayDesc
    def write_ray_desc_from_ray_desc_constructor_function(self, module: Module) -> None:
        self.out.write("RayDesc RayDescFromRayDesc_(")
        self.write_type(module, module.special_types.ray_desc)  # type: ignore[attr-defined]
        self.out.write(" arg0) {\n")
        self.out.write("    RayDesc ret = (RayDesc)0;\n")
        self.out.write("    ret.Origin = arg0.origin;\n")
        self.out.write("    ret.TMin = arg0.tmin;\n")
        self.out.write("    ret.Direction = arg0.dir;\n")
        self.out.write("    ret.TMax = arg0.tmax;\n")
        self.out.write("    return ret;\n")
        self.out.write("}\n\n")

    def write_committed_intersection_function(self, module: Module) -> None:
        from ...ir import Scalar, ScalarKind, TypeInner
        self.write_type(module, module.special_types.ray_intersection)  # type: ignore[attr-defined]
        self.out.write(" GetCommittedIntersection(")
        self.write_value_type(module, TypeInner.new_ray_query(vertex_return=False))
        self.out.write(" rq, ")
        self.write_value_type(module, TypeInner.new_scalar(Scalar(ScalarKind.UINT, 4)))
        self.out.write(" rq_tracker) {\n")
        self.out.write("    ")
        self.write_type(module, module.special_types.ray_intersection)  # type: ignore[attr-defined]
        self.out.write(" ret = (")
        self.write_type(module, module.special_types.ray_intersection)  # type: ignore[attr-defined]
        self.out.write(")0;\n")

        extra_level = Level(0)
        if self.options.ray_query_initialization_tracking:
            # *Technically*, `CommittedStatus` is valid as long as the ray query is initialized,
            # but the metal backend doesn't support this function unless it has finished traversal.
            self.out.write("    if (")
            self.write_contains_flags("rq_tracker", RayQueryPoint.FINISHED_TRAVERSAL.value)
            self.out.write(") {\n")
            extra_level = extra_level.next()

        self.out.write(f"    {extra_level}ret.kind = rq.CommittedStatus();\n")
        self.out.write(
            f"    {extra_level}if( rq.CommittedStatus() == COMMITTED_NOTHING) {{}} else {{\n"
        )
        self.out.write(f"        {extra_level}ret.t = rq.CommittedRayT();\n")
        self.out.write(
            f"        {extra_level}ret.instance_custom_data = rq.CommittedInstanceID();\n"
        )
        self.out.write(
            f"        {extra_level}ret.instance_index = rq.CommittedInstanceIndex();\n"
        )
        self.out.write(
            f"        {extra_level}ret.sbt_record_offset = rq.CommittedInstanceContributionToHitGroupIndex();\n"
        )
        self.out.write(
            f"        {extra_level}ret.geometry_index = rq.CommittedGeometryIndex();\n"
        )
        self.out.write(
            f"        {extra_level}ret.primitive_index = rq.CommittedPrimitiveIndex();\n"
        )
        self.out.write(
            f"        {extra_level}if( rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT ) {{\n"
        )
        self.out.write(
            f"            {extra_level}ret.barycentrics = rq.CommittedTriangleBarycentrics();\n"
        )
        self.out.write(
            f"            {extra_level}ret.front_face = rq.CommittedTriangleFrontFace();\n"
        )
        self.out.write(f"        {extra_level}}}\n")
        self.out.write(
            f"        {extra_level}ret.object_to_world = rq.CommittedObjectToWorld4x3();\n"
        )
        self.out.write(
            f"        {extra_level}ret.world_to_object = rq.CommittedWorldToObject4x3();\n"
        )
        self.out.write(f"    {extra_level}}}\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write("    }\n")

        self.out.write("    return ret;\n")
        self.out.write("}\n\n")

    def write_candidate_intersection_function(self, module: Module) -> None:
        from ...ir import Scalar, ScalarKind, TypeInner
        self.write_type(module, module.special_types.ray_intersection)  # type: ignore[attr-defined]
        self.out.write(" GetCandidateIntersection(")
        self.write_value_type(module, TypeInner.new_ray_query(vertex_return=False))
        self.out.write(" rq, ")
        self.write_value_type(module, TypeInner.new_scalar(Scalar(ScalarKind.UINT, 4)))
        self.out.write(" rq_tracker) {\n")
        self.out.write("    ")
        self.write_type(module, module.special_types.ray_intersection)  # type: ignore[attr-defined]
        self.out.write(" ret = (")
        self.write_type(module, module.special_types.ray_intersection)  # type: ignore[attr-defined]
        self.out.write(")0;\n")

        extra_level = Level(0)
        if self.options.ray_query_initialization_tracking:
            self.out.write("    if (")
            self.write_contains_flags("rq_tracker", RayQueryPoint.PROCEED.value)
            self.out.write(" && !")
            self.write_contains_flags("rq_tracker", RayQueryPoint.FINISHED_TRAVERSAL.value)
            self.out.write(") {\n")
            extra_level = extra_level.next()

        self.out.write(f"    {extra_level}CANDIDATE_TYPE kind = rq.CandidateType();\n")
        self.out.write(
            f"    {extra_level}if (kind == CANDIDATE_NON_OPAQUE_TRIANGLE) {{\n"
        )
        self.out.write(
            f"        {extra_level}ret.kind = {int(RayQueryIntersection.TRIANGLE)};\n"
        )
        self.out.write(f"        {extra_level}ret.t = rq.CandidateTriangleRayT();\n")
        self.out.write(
            f"        {extra_level}ret.barycentrics = rq.CandidateTriangleBarycentrics();\n"
        )
        self.out.write(
            f"        {extra_level}ret.front_face = rq.CandidateTriangleFrontFace();\n"
        )
        self.out.write(f"    {extra_level}}} else {{\n")
        self.out.write(
            f"        {extra_level}ret.kind = {int(RayQueryIntersection.AABB)};\n"
        )
        self.out.write(f"    {extra_level}}}\n")

        self.out.write(
            f"    {extra_level}ret.instance_custom_data = rq.CandidateInstanceID();\n"
        )
        self.out.write(
            f"    {extra_level}ret.instance_index = rq.CandidateInstanceIndex();\n"
        )
        self.out.write(
            f"    {extra_level}ret.sbt_record_offset = rq.CandidateInstanceContributionToHitGroupIndex();\n"
        )
        self.out.write(
            f"    {extra_level}ret.geometry_index = rq.CandidateGeometryIndex();\n"
        )
        self.out.write(
            f"    {extra_level}ret.primitive_index = rq.CandidatePrimitiveIndex();\n"
        )
        self.out.write(
            f"    {extra_level}ret.object_to_world = rq.CandidateObjectToWorld4x3();\n"
        )
        self.out.write(
            f"    {extra_level}ret.world_to_object = rq.CandidateWorldToObject4x3();\n"
        )

        if self.options.ray_query_initialization_tracking:
            self.out.write("    }\n")

        self.out.write("    return ret;\n")
        self.out.write("}\n\n")

    def write_initialize_function(
        self,
        module: Module,
        level: Level,
        query: Handle[Expression],
        acceleration_structure: Handle[Expression],
        descriptor: Handle[Expression],
        rq_tracker: str,
        func_ctx: FunctionCtx,
    ) -> None:
        base_level = level

        # This prevents variables flowing down a level and causing compile errors.
        self.out.write(f"{level}{{\n")
        level = level.next()

        self.out.write(f"{level}")
        self.write_type(module, module.special_types.ray_desc)  # type: ignore[attr-defined]
        self.out.write(" naga_desc = ")
        self.write_expr(module, descriptor, func_ctx)
        self.out.write(";\n")

        if self.options.ray_query_initialization_tracking:
            # Validate ray extents https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#ray-extents
            self.out.write(f"{level}float naga_tmin = naga_desc.tmin;\n")
            self.out.write(f"{level}float naga_tmax = naga_desc.tmax;\n")
            self.out.write(f"{level}float3 naga_origin = naga_desc.origin;\n")
            self.out.write(f"{level}float3 naga_dir = naga_desc.dir;\n")
            self.out.write(f"{level}uint naga_flags = naga_desc.flags;\n")

            self.out.write(
                f"{level}bool naga_tmin_valid = (naga_tmin >= 0.0) && (naga_tmin <= naga_tmax) && !"
            )
            self.write_nan("naga_tmin")
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_tmax_valid = !")
            self.write_nan("naga_tmax")
            self.out.write(";\n")

            # Unlike Vulkan it seems that for DX12, it seems only NaN components of the origin and direction are invalid
            self.out.write(f"{level}bool naga_origin_valid = !any(")
            self.write_nan("naga_origin")
            self.out.write(");\n")

            self.out.write(f"{level}bool naga_dir_valid = !any(")
            self.write_nan("naga_dir")
            self.out.write(");\n")

            def _contains(flag: RayFlag) -> None:
                self.write_contains_flags("naga_flags", int(flag))

            self.out.write(f"{level}bool naga_contains_opaque = ")
            _contains(RayFlag.FORCE_OPAQUE)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_no_opaque = ")
            _contains(RayFlag.FORCE_NO_OPAQUE)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_cull_opaque = ")
            _contains(RayFlag.CULL_OPAQUE)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_cull_no_opaque = ")
            _contains(RayFlag.CULL_NO_OPAQUE)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_cull_front = ")
            _contains(RayFlag.CULL_FRONT_FACING)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_cull_back = ")
            _contains(RayFlag.CULL_BACK_FACING)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_skip_triangles = ")
            _contains(RayFlag.SKIP_TRIANGLES)
            self.out.write(";\n")

            self.out.write(f"{level}bool naga_contains_skip_aabbs = ")
            _contains(RayFlag.SKIP_AABBS)
            self.out.write(";\n")

            # A textified version of the same in the spirv writer
            def _less_than_two_true(bools: list[str]) -> str:
                if len(bools) <= 1:
                    raise ValueError("Must have multiple booleans!")
                final_parts: list[str] = []
                remaining = list(bools)
                while remaining:
                    last_bool = remaining.pop()
                    for other in remaining:
                        final_parts.append(f" ({last_bool} && {other}) ")
                return "||".join(final_parts)

            self.out.write(
                f"{level}bool naga_contains_skip_triangles_aabbs = {_less_than_two_true(['naga_contains_skip_triangles', 'naga_contains_skip_aabbs'])};\n"
            )
            self.out.write(
                f"{level}bool naga_contains_skip_triangles_cull = {_less_than_two_true(['naga_contains_skip_triangles', 'naga_contains_cull_back', 'naga_contains_cull_front'])};\n"
            )
            self.out.write(
                f"{level}bool naga_contains_multiple_opaque = {_less_than_two_true(['naga_contains_opaque', 'naga_contains_no_opaque', 'naga_contains_cull_opaque', 'naga_contains_cull_no_opaque'])};\n"
            )

            self.out.write(
                f"{level}if (naga_tmin_valid && naga_tmax_valid && naga_origin_valid && naga_dir_valid && !(naga_contains_skip_triangles_aabbs || naga_contains_skip_triangles_cull || naga_contains_multiple_opaque)) {{\n"
            )
            level = level.next()
            self.out.write(
                f"{level}{rq_tracker} = {rq_tracker} | {RayQueryPoint.INITIALIZED.value};\n"
            )

        self.out.write(f"{level}")
        self.write_expr(module, query, func_ctx)
        self.out.write(".TraceRayInline(")
        self.write_expr(module, acceleration_structure, func_ctx)
        self.out.write(", naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{base_level}    }}\n")

        self.out.write(f"{base_level}}}\n")

    def write_proceed(
        self,
        module: Module,
        level: Level,
        query: Handle[Expression],
        result: Handle[Expression],
        rq_tracker: str,
        func_ctx: FunctionCtx,
    ) -> None:
        base_level = level
        name = str(Baked(result))
        self.out.write(f"{level}bool {name} = false;\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{level}{{\n")
            level = level.next()
            self.out.write(f"{level}bool naga_has_initialized = ")
            self.write_contains_flags(rq_tracker, RayQueryPoint.INITIALIZED.value)
            self.out.write(";\n")
            self.out.write(f"{level}bool naga_has_finished = ")
            self.write_contains_flags(rq_tracker, RayQueryPoint.FINISHED_TRAVERSAL.value)
            self.out.write(";\n")
            self.out.write(f"{level}if (naga_has_initialized && !naga_has_finished) {{\n")
            level = level.next()

        self.out.write(f"{level}{name} = ")
        self.write_expr(module, query, func_ctx)
        self.out.write(".Proceed();\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write(
                f"{level}{rq_tracker} = {rq_tracker} | {RayQueryPoint.PROCEED.value};\n"
            )
            self.out.write(
                f"{level}if (!{name}) {{ {rq_tracker} = {rq_tracker} | {RayQueryPoint.FINISHED_TRAVERSAL.value}; }}\n"
            )
            self.out.write(f"{base_level}}}}}\n")

        self.named_expressions[result] = name

    def write_generate_intersection(
        self,
        module: Module,
        level: Level,
        query: Handle[Expression],
        hit_t: Handle[Expression],
        rq_tracker: str,
        func_ctx: FunctionCtx,
    ) -> None:
        base_level = level
        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{level}if (")
            self.write_contains_flags(rq_tracker, RayQueryPoint.PROCEED.value)
            self.out.write(" && !")
            self.write_contains_flags(rq_tracker, RayQueryPoint.FINISHED_TRAVERSAL.value)
            self.out.write(") {\n")
            level = level.next()
            self.out.write(f"{level}CANDIDATE_TYPE naga_kind = ")
            self.write_expr(module, query, func_ctx)
            self.out.write(".CandidateType();\n")
            self.out.write(f"{level}float naga_tmin = ")
            self.write_expr(module, query, func_ctx)
            self.out.write(".RayTMin();\n")
            self.out.write(f"{level}float naga_tcurrentmax = ")
            self.write_expr(module, query, func_ctx)
            self.out.write(".CommittedRayT();\n")
            self.out.write(
                f"{level}if ((naga_kind == CANDIDATE_PROCEDURAL_PRIMITIVE) && (naga_tmin <="
            )
            self.write_expr(module, hit_t, func_ctx)
            self.out.write(") && (")
            self.write_expr(module, hit_t, func_ctx)
            self.out.write(" <= naga_tcurrentmax)) {\n")
            level = level.next()

        self.out.write(f"{level}")
        self.write_expr(module, query, func_ctx)
        self.out.write(".CommitProceduralPrimitiveHit(")
        self.write_expr(module, hit_t, func_ctx)
        self.out.write(");\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{base_level}}}}}\n")

    def write_confirm_intersection(
        self,
        module: Module,
        level: Level,
        query: Handle[Expression],
        rq_tracker: str,
        func_ctx: FunctionCtx,
    ) -> None:
        base_level = level
        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{level}if (")
            self.write_contains_flags(rq_tracker, RayQueryPoint.PROCEED.value)
            self.out.write(" && !")
            self.write_contains_flags(rq_tracker, RayQueryPoint.FINISHED_TRAVERSAL.value)
            self.out.write(") {\n")
            level = level.next()
            self.out.write(f"{level}CANDIDATE_TYPE naga_kind = ")
            self.write_expr(module, query, func_ctx)
            self.out.write(".CandidateType();\n")
            self.out.write(f"{level}if (naga_kind == CANDIDATE_NON_OPAQUE_TRIANGLE) {{\n")
            level = level.next()

        self.out.write(f"{level}")
        self.write_expr(module, query, func_ctx)
        self.out.write(".CommitNonOpaqueTriangleHit();\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{base_level}}}}}\n")

    def write_terminate(
        self,
        module: Module,
        level: Level,
        query: Handle[Expression],
        rq_tracker: str,
        func_ctx: FunctionCtx,
    ) -> None:
        base_level = level
        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{level}if (")
            # RayQuery::Abort() can be called any time after RayQuery::TraceRayInline() has been called.
            self.write_contains_flags(rq_tracker, RayQueryPoint.INITIALIZED.value)
            self.out.write(") {\n")
            level = level.next()

        self.out.write(f"{level}")
        self.write_expr(module, query, func_ctx)
        self.out.write(".Abort();\n")

        if self.options.ray_query_initialization_tracking:
            self.out.write(f"{base_level}}}\n")

    # --- Hooks expected from the main writer ---

    def write_expr(self, module: Module, expr: Handle[Expression], func_ctx: FunctionCtx) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_type(self, module: Module, ty: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def write_value_type(self, module: Module, inner: object) -> None:  # pragma: no cover
        raise NotImplementedError


__all__ = [
    "RayFlag",
    "RayQueryIntersection",
    "RayWriterOptions",
    "RayWriter",
]
