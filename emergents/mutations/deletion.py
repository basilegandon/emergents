from typing import Optional

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.mutations.base import Mutation


class Deletion(Mutation):
    def __init__(self, start_pos: int, end_pos: int, rng_state: Optional[int] = None):
        super().__init__(rng_state)
        self.start_pos: int = start_pos
        self.end_pos: int = end_pos

    def _intervals_for_del(self, genome: Genome) -> list[tuple[int, int]]:
        """Return list of non-wrapping intervals (start, end) representing the duplicated region(s).
        Handles circular genomes by splitting into at most two intervals when necessary.
        """
        if self.start_pos > self.end_pos:
            # Only possible if genome.circular
            # allow wrap-around duplication
            return [(self.start_pos, genome.length), (0, self.end_pos)]
        if self.start_pos <= self.end_pos:
            return [(self.start_pos, self.end_pos)]
        else:
            raise ValueError(
                "Non-circular genome: duplication start must be < end (no wrap allowed)"
            )

    def _is_deleted_seg_neutral(
        self, genome: Genome, start_pos: int, end_pos: int
    ) -> bool:
        """Check if the deleted segment is neutral (i.e., does not affect the organism's fitness)"""
        segment_at_start, *_ = genome.find_segment_at_position(
            start_pos, CoordinateSystem.BASE
        )
        if not segment_at_start.is_noncoding():
            return False
        segment_at_end, *_ = genome.find_segment_at_position(
            end_pos, CoordinateSystem.GAP
        )
        if segment_at_start.sid is segment_at_end.sid:
            return True
        return False

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the deletion is neutral (i.e., does not affect the organism's fitness)"""
        intervals: list[tuple[int, int]] = self._intervals_for_del(genome)
        start_pos, end_pos = intervals[0]
        if not self._is_deleted_seg_neutral(genome, start_pos, end_pos):
            return False
        if len(intervals) == 2:
            start_pos, end_pos = intervals[1]
            if not self._is_deleted_seg_neutral(genome, start_pos, end_pos):
                return False
        return True

    def apply(self, genome: Genome):
        """Apply the deletion to the genome."""
        genome.delete_range(self.start_pos, self.end_pos + 1)

    def describe(self) -> str:
        return f"Deletion(start_pos={self.start_pos}, end_pos={self.end_pos})"
