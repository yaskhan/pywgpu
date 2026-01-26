"""
Shader I/O deductions for validation.

This module provides deduction logic for shader input/output variable limits,
taking into account built-in variables and topology.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, TypeVar

try:
    import naga
except ImportError:
    naga = None  # type: ignore

T = TypeVar("T")


class InterStageBuiltIn(Enum):
    """
    A naga.BuiltIn that counts towards inter-stage shader I/O limits.

    See also: https://www.w3.org/TR/webgpu/#inter-stage-builtins

    Attributes:
        POSITION: Position built-in.
        FRONT_FACING: Front-facing built-in.
        SAMPLE_INDEX: Sample index built-in.
        SAMPLE_MASK: Sample mask built-in.
        PRIMITIVE_INDEX: Primitive index built-in.
        SUBGROUP_INVOCATION_ID: Subgroup invocation ID built-in.
        SUBGROUP_SIZE: Subgroup size built-in.
        POINT_COORD: Point coordinate built-in (non-standard).
        BARYCENTRIC: Barycentric coordinate built-in (non-standard).
        VIEW_INDEX: View index built-in (non-standard).
    """

    POSITION = "position"
    FRONT_FACING = "front_facing"
    SAMPLE_INDEX = "sample_index"
    SAMPLE_MASK = "sample_mask"
    PRIMITIVE_INDEX = "primitive_index"
    SUBGROUP_INVOCATION_ID = "subgroup_invocation_id"
    SUBGROUP_SIZE = "subgroup_size"
    POINT_COORD = "point_coord"
    BARYCENTRIC = "barycentric"
    VIEW_INDEX = "view_index"


class MaxVertexShaderOutputDeduction(Enum):
    """
    Max shader I/O variable deductions for vertex shader output.

    Used to calculate effective limits on vertex shader outputs based on
    pipeline configuration.

    Attributes:
        POINT_LIST_PRIMITIVE_TOPOLOGY: Deduction when primitive topology
                                        is PointList.
    """

    POINT_LIST_PRIMITIVE_TOPOLOGY = "point_list"

    def for_variables(self) -> int:
        """
        Get the deduction amount for variable count.

        Returns:
            The number of variables to deduct.
        """
        if self == MaxVertexShaderOutputDeduction.POINT_LIST_PRIMITIVE_TOPOLOGY:
            return 1
        return 0

    def for_location(self) -> int:
        """
        Get the deduction amount for location indexing.

        Returns:
            The location offset to deduct.
        """
        if self == MaxVertexShaderOutputDeduction.POINT_LIST_PRIMITIVE_TOPOLOGY:
            return 0
        return 0


class MaxFragmentShaderInputDeduction:
    """
    Max shader I/O variable deductions for fragment shader input.

    Used to calculate effective limits on fragment shader inputs based on
    the presence of inter-stage built-ins.

    Attributes:
        builtin: The inter-stage built-in causing the deduction.
    """

    def __init__(self, builtin: InterStageBuiltIn) -> None:
        """
        Create a new fragment shader input deduction.

        Args:
            builtin: The inter-stage built-in.
        """
        self.builtin = builtin

    def for_variables(self) -> int:
        """
        Get the deduction amount for variable count.

        Returns:
            The number of variables to deduct.
        """
        mapping = {
            InterStageBuiltIn.FRONT_FACING: 1,
            InterStageBuiltIn.SAMPLE_INDEX: 1,
            InterStageBuiltIn.SAMPLE_MASK: 1,
            InterStageBuiltIn.PRIMITIVE_INDEX: 1,
            InterStageBuiltIn.SUBGROUP_INVOCATION_ID: 1,
            InterStageBuiltIn.SUBGROUP_SIZE: 1,
            InterStageBuiltIn.VIEW_INDEX: 1,
            InterStageBuiltIn.POINT_COORD: 1,
            InterStageBuiltIn.BARYCENTRIC: 3,
            InterStageBuiltIn.POSITION: 0,
        }
        return mapping.get(self.builtin, 0)

    @staticmethod
    def from_inter_stage_builtin(builtin: Any) -> Optional[MaxFragmentShaderInputDeduction]:
        """
        Create a deduction from a naga built-in.

        Args:
            builtin: The naga BuiltIn value.

        Returns:
            A MaxFragmentShaderInputDeduction if the built-in is an inter-stage
            built-in, None otherwise.
        """
        if naga is None:
            return None

        # Map naga built-ins to inter-stage built-ins
        builtin_name = getattr(builtin, "name", str(builtin)).lower()

        mapping = {
            "position": InterStageBuiltIn.POSITION,
            "front_facing": InterStageBuiltIn.FRONT_FACING,
            "sample_index": InterStageBuiltIn.SAMPLE_INDEX,
            "sample_mask": InterStageBuiltIn.SAMPLE_MASK,
            "primitive_index": InterStageBuiltIn.PRIMITIVE_INDEX,
            "subgroup_size": InterStageBuiltIn.SUBGROUP_SIZE,
            "subgroup_invocation_id": InterStageBuiltIn.SUBGROUP_INVOCATION_ID,
            "point_coord": InterStageBuiltIn.POINT_COORD,
            "barycentric": InterStageBuiltIn.BARYCENTRIC,
            "view_index": InterStageBuiltIn.VIEW_INDEX,
        }

        inter_stage = mapping.get(builtin_name)
        if inter_stage is not None:
            return MaxFragmentShaderInputDeduction(inter_stage)

        return None

    def __repr__(self) -> str:
        """Get a debug representation."""
        return f"MaxFragmentShaderInputDeduction({self.builtin})"


def display_deductions_as_optional_list(
    deductions: List[T],
    accessor: Callable[[T], int],
) -> str:
    """
    Display deductions as an optional list for error messages.

    Args:
        deductions: List of deduction objects.
        accessor: Function to get the deduction amount from an object.

    Returns:
        A formatted string describing the deductions.
    """
    relevant_deductions = [
        (deduction, accessor(deduction))
        for deduction in deductions
        if accessor(deduction) > 0
    ]

    if not relevant_deductions:
        return ""

    lines = ["; note that some deductions apply during validation:"]
    for deduction, amount in relevant_deductions:
        lines.append(f"\n- {deduction!r}: {amount}")

    return "".join(lines)


__all__ = [
    "InterStageBuiltIn",
    "MaxVertexShaderOutputDeduction",
    "MaxFragmentShaderInputDeduction",
    "display_deductions_as_optional_list",
]
