from typing import Optional

from emergents.genome.genome import Genome
from emergents.mutations.base import Mutation


class SmallDeletion(Mutation):
    def __init__(self, position: int, length: int, rng_state: Optional[int] = None):
        super().__init__(rng_state)
        self.position = position
        self.length = length

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the deletion is neutral (i.e., does not affect the organism's fitness)"""
        segment_at_start, *_ = genome[self.position]
        if not segment_at_start.is_noncoding():
            return False
        segment_at_end, *_ = genome[self.position + self.length - 1]
        if segment_at_start.sid is segment_at_end.sid:
            return True
        return False

    def apply(self, genome: Genome):
        """Apply the deletion to the genome."""
        genome.delete_range(self.position, self.position + self.length)

    def describe(self) -> str:
        return f"SmallDeletion(position={self.position}, length={self.length})"
