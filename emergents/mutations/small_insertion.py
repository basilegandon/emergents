from typing import Optional

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.genome.segments import NonCodingSegment
from emergents.mutations.base import Mutation


class SmallInsertion(Mutation):
    def __init__(self, position: int, length: int, rng_state: Optional[int] = None):
        super().__init__(rng_state)
        if position < 0:
            raise ValueError("Position must be non-negative")
        if length <= 0:
            raise ValueError("Length must be positive")
        self.position = position
        self.length = length

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the insertion is neutral (i.e., does not affect the organism's fitness).
        Here position means in between the bases with 0 being before the first base."""
        if self.position == genome.length:
            # Only possible when genome is not circular.
            return True
        segment, offset, *_ = genome.find_segment_at_position(
            self.position, CoordinateSystem.GAP
        )
        return (
            segment.is_noncoding() or offset == 0
        )  # If inserting at the start of a segment, it's neutral.

    def apply(self, genome: Genome):
        """Apply the insertion to the genome."""
        genome.insert_at_gap(self.position, NonCodingSegment(self.length))

    def describe(self) -> str:
        return f"SmallInsertion(position={self.position}, length={self.length})"
