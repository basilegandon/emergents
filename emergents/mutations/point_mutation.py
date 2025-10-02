from typing import Optional

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.mutations.base import Mutation


class PointMutation(Mutation):
    def __init__(self, position: int, rng_state: Optional[int] = None):
        """Point mutation replace one base per another. If it is a non-coding
        base (i.e., a base outside a coding sequence), the mutation is neutral.
        """
        super().__init__(rng_state)
        if position < 0:
            raise ValueError("Position must be non-negative")
        self.position = position

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the mutation is neutral.

        Position is base number.
        """
        segment, *_ = genome.find_segment_at_position(
            self.position, CoordinateSystem.BASE
        )
        return segment.is_noncoding()

    def apply(self, genome: Genome):
        """Apply the point mutation to the genome."""
        pass

    def describe(self) -> str:
        """Return a human-readable description of the mutation."""
        return f"Point mutation at {self.position}"
