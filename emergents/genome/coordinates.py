from enum import Enum
from typing import Protocol, runtime_checkable


class CoordinateSystem(Enum):
    """Coordinate system types for genome operations."""

    BASE = "base"  # Points to actual bases: 0 = first base, valid range [0, length)
    GAP = "gap"  # Points between bases: 0 = before first base, valid range [0, length] if genome is linear else [0, length)


@runtime_checkable
class CoordinateValidator(Protocol):
    """Protocol for coordinate validation."""

    def validate_position(
        self, pos: int, genome_length: int, coord_sys: CoordinateSystem
    ) -> None:
        """Validate a single position."""
        ...

    def validate_base_range(self, start: int, end: int, genome_length: int) -> None:
        """Validate a range of base positions [start, end)."""
        ...


class DefaultCoordinateValidator:
    """Default implementation of coordinate validation."""

    def validate_position(
        self, pos: int, genome_length: int, coord_sys: CoordinateSystem
    ) -> None:
        """Validate a single position based on coordinate system."""
        if coord_sys == CoordinateSystem.BASE:
            if pos < 0 or pos >= genome_length:
                raise IndexError(
                    f"Base position {pos} out of bounds [0, {genome_length})"
                )
        elif coord_sys == CoordinateSystem.GAP:
            if pos < 0 or pos > genome_length:
                raise IndexError(
                    f"Gap position {pos} out of bounds [0, {genome_length}]"
                )
        else:
            raise ValueError(f"Unknown coordinate system: {coord_sys}")

    def validate_base_range(self, start: int, end: int, genome_length: int) -> None:
        """Validate a range of base positions [start, end)."""
        if start < 0 or end > genome_length:
            raise IndexError(
                f"Base range [{start}, {end}) out of bounds [0, {genome_length}]"
            )
        if start >= end:
            raise ValueError(f"Invalid range: start ({start}) must be < end ({end})")
