from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.genome.node import Node, merge, split_by_pos, update_subtree_len
from emergents.genome.segments import CodingSegment, Segment
from emergents.mutations.base import Mutation


class Inversion(Mutation):
    def __init__(self, start_pos: int, end_pos: int, rng_state: int | None = None):
        super().__init__(rng_state)
        if start_pos < 0 or end_pos < 0:
            raise ValueError("Inversion positions must be non-negative")
        if start_pos == end_pos:
            raise ValueError("Inversion start and end positions cannot be the same")
        self.reverted: bool = False
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos
            self.reverted = True
        self.start_pos: int = start_pos
        self.end_pos: int = end_pos

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the deleted segment is neutral (i.e., does not affect the organism's fitness)"""
        segment_at_start, offset, *_ = genome.find_segment_at_position(
            self.start_pos, CoordinateSystem.GAP
        )
        if not segment_at_start.is_noncoding() and offset != 0:
            return False
        segment_at_end, offset, *_ = genome.find_segment_at_position(
            self.end_pos, CoordinateSystem.GAP
        )
        return not (
            not segment_at_end.is_noncoding()
            and offset != 0
            and self.end_pos != genome.length
        )

    def apply(self, genome: Genome) -> None:
        """Apply the inversion to the genome, preserving order and merging noncoding segments at boundaries."""
        left, mid_right = split_by_pos(genome.root, self.start_pos)
        mid, right = split_by_pos(mid_right, self.end_pos - self.start_pos)

        # Collect middle in order
        segments: list[Segment] = []

        def _flatten(node: Node | None) -> None:
            if not node:
                return
            _flatten(node.left)
            segments.append(node.segment)
            _flatten(node.right)

        _flatten(mid)

        # Reverse, invert promoter orientation
        inverted_segments: list[Segment] = []
        for seg in reversed(segments):
            if isinstance(seg, CodingSegment):
                seg.promoter_direction = seg.promoter_direction.switch()
            inverted_segments.append(seg)

        # Merge noncoding at left boundary
        if left and inverted_segments:
            # Find rightmost node in left
            rightmost = left
            while rightmost.right:
                rightmost = rightmost.right
            if rightmost.segment.is_noncoding() and inverted_segments[0].is_noncoding():
                merged_segment = rightmost.segment.clone_with_length(
                    rightmost.segment.length + inverted_segments[0].length
                )
                # Remove rightmost from left
                left = split_by_pos(left, self.start_pos - rightmost.segment.length)[0]
                inverted_segments = inverted_segments[1:]
                left = merge(left, Node(merged_segment))

        # Merge noncoding at right boundary
        if right and inverted_segments:
            leftmost = right
            while leftmost.left:
                leftmost = leftmost.left
            if leftmost.segment.is_noncoding() and inverted_segments[-1].is_noncoding():
                merged_segment = leftmost.segment.clone_with_length(
                    inverted_segments[-1].length + leftmost.segment.length
                )
                # Remove leftmost from right
                right = split_by_pos(right, leftmost.segment.length)[1]
                inverted_segments = inverted_segments[:-1]
                right = merge(Node(merged_segment), right)

        # Rebuild subtree from inverted list
        new_mid = None
        for seg in inverted_segments:
            new_mid = merge(new_mid, Node(seg))

        # Merge in order: left + new_mid + right
        root = merge(left, new_mid) if left else new_mid
        if right:
            root = merge(root, right)
        genome.root = root
        update_subtree_len(genome.root)

    def describe(self) -> str:
        return f"Inversion(start_pos={self.start_pos}, end_pos={self.end_pos})"
