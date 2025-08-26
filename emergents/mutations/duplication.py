from typing import Optional

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection
from emergents.mutations.base import Mutation


class Duplication(Mutation):
    def __init__(
        self,
        start_pos: int,
        end_pos: int,
        insertion_pos: int,
        rng_state: Optional[int] = None,
    ):
        super().__init__(rng_state)
        self.start_pos: int = start_pos
        self.end_pos: int = end_pos
        self.insertion_pos: int = insertion_pos
        self.length: None | int = None  # computed on demand

    def get_length(self, genome: Optional[Genome] = None) -> int:
        if self.length is None:
            if genome is None:
                raise AttributeError(
                    "Genome is required to compute length on first call"
                )
            if self.start_pos <= self.end_pos:
                self.length = self.end_pos - self.start_pos + 1
            else:
                self.length = genome.length - (self.start_pos - self.end_pos + 1)
        return self.length

    def _intervals_for_dup(self, genome: Genome) -> list[tuple[int, int]]:
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

    def _is_dupplicated_seg_neutral(
        self, genome: Genome, start_pos: int, end_pos: int
    ) -> bool:
        # iterate through segments in order until we've passed the last relevant end
        for seg, seg_start, seg_end in genome.iter_segments():
            # compute overlap between this segment and the current interval
            overlap_start = max(seg_start, start_pos)
            overlap_end = min(seg_end, end_pos)
            if overlap_start >= overlap_end:
                continue  # no overlap

            # ignore noncoding
            if not isinstance(seg, CodingSegment):
                continue

            # For a forward coding segment: a promoter at the start (offset 0) is copied iff the duplication includes seg_start
            # For a reverse coding segment: a promoter at the end (offset seg.length-1) is copied iff duplication includes seg_end
            if seg.promoter_direction == PromoterDirection.FORWARD:
                # duplication copies promoter-at-start iff overlap includes seg_start (i.e. overlap_start == seg_start)
                if overlap_start == seg_start:
                    return False
            else:
                # reverse strand: promoter at end. duplication copies it iff overlap includes seg_end
                # equivalently, duplication includes seg_end if half-open interval contains last base when overlap_end == seg_end
                if overlap_end == seg_end:
                    return False
        return True

    def is_neutral(self, genome: Genome) -> bool:
        """Return True if duplication is allowed under the rule:
        - If duplication copies any promoter (start-of-segment for forward coding segments,
          end-of-segment for reverse coding segments), it's invalid.
        - NonCoding segments are ignored.
        """
        segment, offset, *_ = genome.find_segment_at_position(
            self.insertion_pos, CoordinateSystem.GAP
        )
        if (
            not segment.is_noncoding()
            and offset != 0
            and self.insertion_pos != genome.length
        ):
            # If inserting other place than at the start of a segment, it's neutral.
            return False

        intervals: list[tuple[int, int]] = self._intervals_for_dup(genome)
        start_pos, end_pos = intervals[0]
        if not self._is_dupplicated_seg_neutral(genome, start_pos, end_pos):
            return False
        if len(intervals) == 2:
            start_pos, end_pos = intervals[1]
            if not self._is_dupplicated_seg_neutral(genome, start_pos, end_pos):
                return False

        # if we never found a forbidden promoter, duplication is valid
        return True

    def apply(self, genome: Genome):
        """Apply the insertion to the genome."""
        genome.insert_at_gap(
            self.insertion_pos, NonCodingSegment(self.get_length(genome))
        )

    def describe(self, genome: Optional[Genome] = None) -> str:
        return f"Duplication(duplicated segment start={self.start_pos}, duplicated segment end={self.end_pos}, length={self.get_length(genome)})"
