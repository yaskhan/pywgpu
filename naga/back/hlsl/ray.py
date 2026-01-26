"""
Ray query support for HLSL backend.

Implements functions for handling ray tracing operations and queries in HLSL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import io

if TYPE_CHECKING:
    from ...ir.module import Module
    from ...ir.type import TypeInner
    from ...ir.ray_query import RayQueryIntersection
    from .. import Level


class RayWriter:
    """Writer for ray query related code in HLSL."""

    def __init__(self, out: io.StringIO, module: Module, options: Any):
        """
        Initialize ray writer.

        Args:
            out: Output stream
            module: The module being written
            options: Writer options
        """
        self.out = out
        self.module = module
        self.options = options

    def write_not_finite(self, expr: str) -> None:
        """Write a check for non-finite values.

        Args:
            expr: Expression string to check
        """
        self.out.write(f"((asuint({expr}) & 0x7f800000) == 0x7f800000)")

    def write_nan(self, expr: str) -> None:
        """Write a check for NaN values.

        Args:
            expr: Expression string to check
        """
        self.out.write("(")
        self.write_not_finite(expr)
        self.out.write(f" && ((asuint({expr}) & 0x7fffff) != 0))")

    def write_contains_flags(self, expr: str, flags: int) -> None:
        """Write a check for specific flags.

        Args:
            expr: Expression string to check
            flags: Flags to check for
        """
        self.out.write(f"(({expr} & {flags}) == {flags})")

    def write_ray_desc_from_ray_desc_constructor_function(self) -> None:
        """Write function to convert WGSL RayDesc to HLSL RayDesc."""
        ray_desc_type = self._get_type_name("ray_desc")

        self.out.write(f"RayDesc RayDescFromRayDesc_({ray_desc_type} arg0) {{\n")
        self.out.write("    RayDesc ret = (RayDesc)0;\n")
        self.out.write("    ret.Origin = arg0.origin;\n")
        self.out.write("    ret.TMin = arg0.tmin;\n")
        self.out.write("    ret.Direction = arg0.dir;\n")
        self.out.write("    ret.TMax = arg0.tmax;\n")
        self.out.write("    return ret;\n")
        self.out.write("}\n\n")

    def write_committed_intersection_function(self) -> None:
        """Write function to get committed intersection from ray query."""
        ray_intersection_type = self._get_type_name("ray_intersection")

        self.out.write(f"{ray_intersection_type} GetCommittedIntersection(")
        self.out.write("RayQuery<0> rq, ")
        self.out.write("uint rq_tracker) {\n")
        self.out.write(f"    {ray_intersection_type} ret = ({ray_intersection_type})0;\n")

        level = Level(0)
        if self._has_ray_query_tracking():
            # Only valid if ray query is initialized and finished traversal
            self.out.write("    if (")
            from .. import RayQueryPoint
            self.write_contains_flags(
                "rq_tracker",
                RayQueryPoint.FINISHED_TRAVERSAL.value
            )
            self.out.write(") {\n")
            level = level.next()

        self.out.write(f"    {level}ret.kind = rq.CommittedStatus();\n")
        self.out.write(f"    {level}if( rq.CommittedStatus() == COMMITTED_NOTHING) {{}} else {{\n")
        self.out.write(f"        {level}ret.t = rq.CommittedRayT();\n")
        self.out.write(f"        {level}ret.instance_custom_data = rq.CommittedInstanceID();\n")
        self.out.write(f"        {level}ret.instance_index = rq.CommittedInstanceIndex();\n")
        self.out.write(f"        {level}ret.sbt_record_offset = rq.CommittedInstanceContributionToHitGroupIndex();\n")
        self.out.write(f"        {level}ret.geometry_index = rq.CommittedGeometryIndex();\n")
        self.out.write(f"        {level}ret.primitive_index = rq.CommittedPrimitiveIndex();\n")
        self.out.write(f"        {level}if( rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT ) {{\n")
        self.out.write(f"            {level}ret.barycentrics = rq.CommittedTriangleBarycentrics();\n")
        self.out.write(f"            {level}ret.front_face = rq.CommittedTriangleFrontFace();\n")
        self.out.write(f"        {level}}}\n")
        self.out.write(f"        {level}ret.object_to_world = rq.CommittedObjectToWorld4x3();\n")
        self.out.write(f"        {level}ret.world_to_object = rq.CommittedWorldToObject4x3();\n")
        self.out.write(f"    {level}}}\n")

        if self._has_ray_query_tracking():
            self.out.write("    }\n")

        self.out.write("    return ret;\n")
        self.out.write("}\n\n")

    def write_candidate_intersection_function(self) -> None:
        """Write function to get candidate intersection from ray query."""
        ray_intersection_type = self._get_type_name("ray_intersection")

        self.out.write(f"{ray_intersection_type} GetCandidateIntersection(")
        self.out.write("RayQuery<0> rq, ")
        self.out.write("uint rq_tracker) {\n")
        self.out.write(f"    {ray_intersection_type} ret = ({ray_intersection_type})0;\n")

        level = Level(0)
        if self._has_ray_query_tracking():
            # Only valid if ray query has proceeded but not finished traversal
            self.out.write("    if (")
            from .. import RayQueryPoint
            self.write_contains_flags(
                "rq_tracker",
                RayQueryPoint.PROCEED.value
            )
            self.out.write(" && !")
            self.write_contains_flags(
                "rq_tracker",
                RayQueryPoint.FINISHED_TRAVERSAL.value
            )
            self.out.write(") {\n")
            level = level.next()

        self.out.write(f"    {level}CANDIDATE_TYPE kind = rq.CandidateType();\n")
        self.out.write(f"    {level}if (kind == CANDIDATE_NON_OPAQUE_TRIANGLE) {{\n")
        self.out.write(f"        {level}ret.kind = {RayQueryIntersection.TRIANGLE.value};\n")
        self.out.write(f"        {level}ret.t = rq.CandidateTriangleRayT();\n")
        self.out.write(f"        {level}ret.barycentrics = rq.CandidateTriangleBarycentrics();\n")
        self.out.write(f"        {level}ret.front_face = rq.CandidateTriangleFrontFace();\n")
        self.out.write(f"    {level}}} else {{\n")
        self.out.write(f"        {level}ret.kind = {RayQueryIntersection.BOUNDING_BOX.value};\n")
        self.out.write(f"        {level}ret.t = rq.CandidateProceduralPrimitiveRayT();\n")
        self.out.write(f"    {level}}}\n")

        if self._has_ray_query_tracking():
            self.out.write("    }\n")

        self.out.write("    return ret;\n")
        self.out.write("}\n\n")

    def write_ray_desc_type(self) -> None:
        """Write the RayDesc struct type."""
        self.out.write("struct RayDesc {\n")
        self.out.write("    float3 Origin;\n")
        self.out.write("    float  TMin;\n")
        self.out.write("    float3 Direction;\n")
        self.out.write("    float  TMax;\n")
        self.out.write("};\n\n")

    def write_ray_intersection_type(self) -> None:
        """Write the RayIntersection struct type."""
        self.out.write("struct RayIntersection {\n")
        self.out.write("    uint   kind;\n")
        self.out.write("    float  t;\n")
        self.out.write("    float2 barycentrics;\n")
        self.out.write("    bool   front_face;\n")
        self.out.write("    uint   instance_custom_data;\n")
        self.out.write("    uint   instance_index;\n")
        self.out.write("    uint   sbt_record_offset;\n")
        self.out.write("    uint   geometry_index;\n")
        self.out.write("    uint   primitive_index;\n")
        self.out.write("    float3x4 object_to_world;\n")
        self.out.write("    float3x4 world_to_object;\n")
        self.out.write("};\n\n")

    def _get_type_name(self, type_name: str) -> str:
        """Get the name of a special type from the module.

        Args:
            type_name: Name of the special type

        Returns:
            Type name string
        """
        if hasattr(self.module, 'special_types'):
            special_type = getattr(self.module.special_types, type_name, None)
            if special_type is not None:
                return f"Naga{type_name.capitalize()}"
        return type_name.capitalize()

    def _has_ray_query_tracking(self) -> bool:
        """Check if ray query initialization tracking is enabled.

        Returns:
            True if tracking is enabled
        """
        return (
            hasattr(self.options, 'ray_query_initialization_tracking')
            and self.options.ray_query_initialization_tracking
        )


__all__ = [
    "RayWriter",
]
