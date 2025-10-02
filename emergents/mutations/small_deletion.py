from typing import Optional

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.mutations.base import Mutation


class SmallDeletion(Mutation):
    def __init__(self, position: int, length: int, rng_state: Optional[int] = None):
        super().__init__(rng_state)
        if position < 0:
            raise ValueError("Position must be non-negative")
        if length <= 0:
            raise ValueError("Length must be positive")
        self.position = position
        self.length = length

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the deletion is neutral (i.e., does not affect the organism's fitness)"""
        segment_at_start, *_ = genome.find_segment_at_position(
            self.position, CoordinateSystem.BASE
        )
        if not segment_at_start.is_noncoding():
            return False
        segment_at_end, *_ = genome.find_segment_at_position(
            self.position + self.length - 1, CoordinateSystem.GAP
        )
        if segment_at_start.sid is segment_at_end.sid:
            return True
        return False

    def apply(self, genome: Genome):
        """Apply the deletion to the genome."""
        genome.delete_range(self.position, self.position + self.length)

    def describe(self) -> str:
        return f"SmallDeletion(position={self.position}, length={self.length})"
