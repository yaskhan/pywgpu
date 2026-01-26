"""
Sampler configuration for MSL (Metal Shading Language) backend.

Defines enums and structs for inline sampler configuration in Metal.
"""

from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Coord(Enum):
    """Sampler coordinate mode."""

    NORMALIZED = 0
    PIXEL = 1

    def as_str(self) -> str:
        """Convert to MSL string representation.

        Returns:
            MSL string for the coordinate mode
        """
        match self:
            case Coord.NORMALIZED:
                return "normalized"
            case Coord.PIXEL:
                return "pixel"


class Address(Enum):
    """Sampler address mode (wrapping behavior)."""

    REPEAT = 0
    MIRRORED_REPEAT = 1
    CLAMP_TO_EDGE = 2
    CLAMP_TO_ZERO = 3
    CLAMP_TO_BORDER = 4

    def as_str(self) -> str:
        """Convert to MSL string representation.

        Returns:
            MSL string for the address mode
        """
        match self:
            case Address.REPEAT:
                return "repeat"
            case Address.MIRRORED_REPEAT:
                return "mirrored_repeat"
            case Address.CLAMP_TO_EDGE:
                return "clamp_to_edge"
            case Address.CLAMP_TO_ZERO:
                return "clamp_to_zero"
            case Address.CLAMP_TO_BORDER:
                return "clamp_to_border"


class BorderColor(Enum):
    """Sampler border color."""

    TRANSPARENT_BLACK = 0
    OPAQUE_BLACK = 1
    OPAQUE_WHITE = 2

    def as_str(self) -> str:
        """Convert to MSL string representation.

        Returns:
            MSL string for the border color
        """
        match self:
            case BorderColor.TRANSPARENT_BLACK:
                return "transparent_black"
            case BorderColor.OPAQUE_BLACK:
                return "opaque_black"
            case BorderColor.OPAQUE_WHITE:
                return "opaque_white"


class Filter(Enum):
    """Sampler filter mode."""

    NEAREST = 0
    LINEAR = 1

    def as_str(self) -> str:
        """Convert to MSL string representation.

        Returns:
            MSL string for the filter mode
        """
        match self:
            case Filter.NEAREST:
                return "nearest"
            case Filter.LINEAR:
                return "linear"


class CompareFunc(Enum):
    """Sampler comparison function."""

    NEVER = 0
    LESS = 1
    LESS_EQUAL = 2
    GREATER = 3
    GREATER_EQUAL = 4
    EQUAL = 5
    NOT_EQUAL = 6
    ALWAYS = 7

    def as_str(self) -> str:
        """Convert to MSL string representation.

        Returns:
            MSL string for the comparison function
        """
        match self:
            case CompareFunc.NEVER:
                return "never"
            case CompareFunc.LESS:
                return "less"
            case CompareFunc.LESS_EQUAL:
                return "less_equal"
            case CompareFunc.GREATER:
                return "greater"
            case CompareFunc.GREATER_EQUAL:
                return "greater_equal"
            case CompareFunc.EQUAL:
                return "equal"
            case CompareFunc.NOT_EQUAL:
                return "not_equal"
            case CompareFunc.ALWAYS:
                return "always"


@dataclass
class InlineSampler:
    """Inline sampler configuration for Metal.

    Metal supports inline sampler declaration that includes all sampler
    parameters in a single declaration.
    """

    coord: Coord = Coord.NORMALIZED
    """Coordinate mode: normalized or pixel coordinates."""

    address: Tuple[Address, Address, Address] = (
        Address.CLAMP_TO_EDGE,
        Address.CLAMP_TO_EDGE,
        Address.CLAMP_TO_EDGE,
    )
    """Address mode for (s, t, r) coordinates."""

    border_color: BorderColor = BorderColor.TRANSPARENT_BLACK
    """Border color for clamp_to_border address mode."""

    mag_filter: Filter = Filter.NEAREST
    """Magnification filter."""

    min_filter: Filter = Filter.NEAREST
    """Minification filter."""

    mip_filter: Optional[Filter] = None
    """Mipmap filter (if applicable)."""

    lod_clamp: Optional[Tuple[float, float]] = None
    """Optional LOD clamp range (min, max)."""

    max_anisotropy: Optional[int] = None
    """Maximum anisotropy (1-16)."""

    compare_func: CompareFunc = CompareFunc.NEVER
    """Comparison function for comparison samplers."""

    def __eq__(self, other: object) -> bool:
        """Check equality (custom implementation for hashing)."""
        if not isinstance(other, InlineSampler):
            return False

        return (
            self.coord == other.coord
            and self.address == other.address
            and self.border_color == other.border_color
            and self.mag_filter == other.mag_filter
            and self.min_filter == other.min_filter
            and self.mip_filter == other.mip_filter
            and self.lod_clamp == other.lod_clamp
            and self.max_anisotropy == other.max_anisotropy
            and self.compare_func == other.compare_func
        )

    def __hash__(self) -> int:
        """Hash implementation (for use in sets/maps)."""
        # Combine all fields into a hash
        h = hash(self.coord)
        h = (h * 31) ^ hash(self.address)
        h = (h * 31) ^ hash(self.border_color)
        h = (h * 31) ^ hash(self.mag_filter)
        h = (h * 31) ^ hash(self.min_filter)
        h = (h * 31) ^ hash(self.mip_filter)
        h = (h * 31) ^ hash(self.lod_clamp)
        h = (h * 31) ^ hash(self.max_anisotropy)
        h = (h * 31) ^ hash(self.compare_func)
        return h

    def to_msl_string(self, name: str) -> str:
        """Generate MSL sampler declaration string.

        Args:
            name: Name of the sampler variable

        Returns:
            MSL sampler declaration
        """
        # Build attribute list
        attrs = []

        # Address modes
        attrs.append(f"address({self.address[0].as_str()}, {self.address[1].as_str()}, {self.address[2].as_str()})")

        # Filter modes
        attrs.append(f"filter::{self.mag_filter.as_str()}")
        attrs.append(f"filter::{self.min_filter.as_str()}")

        # Mipmap filter
        if self.mip_filter:
            attrs.append(f"filter::{self.mip_filter.as_str()}")

        # Border color
        if any(addr == Address.CLAMP_TO_BORDER for addr in self.address):
            attrs.append(f"border_color({self.border_color.as_str()})")

        # LOD clamp
        if self.lod_clamp:
            min_lod, max_lod = self.lod_clamp
            attrs.append(f"lod_clamp({min_lod}, {max_lod})")

        # Max anisotropy
        if self.max_anisotropy:
            attrs.append(f"max_anisotropy({self.max_anisotropy})")

        # Compare function
        if self.compare_func != CompareFunc.NEVER:
            attrs.append(f"compare_func({self.compare_func.as_str()})")

        # Combine attributes
        attr_str = ", ".join(attrs)

        # Coordinate mode
        coord_str = self.coord.as_str()

        return f"constexpr sampler {name} {coord_str}({attr_str});"


__all__ = [
    # Enums
    "Coord",
    "Address",
    "BorderColor",
    "Filter",
    "CompareFunc",
    # Classes
    "InlineSampler",
]
