from collections.abc import Iterator

from emergents.genome.coordinates import CoordinateSystem, DefaultCoordinateValidator
from emergents.genome.node import Node, merge, split_by_pos, update_subtree_len
from emergents.genome.segments import NonCodingSegment, Segment

SegmentLike = Segment | list[Segment]


class Genome:
    """
    Genome stored as an implicit treap of Segment nodes.

    Coordinate Systems:
    - BASE: Points to actual bases (0-indexed). Range: [0, length)
    - GAP: Points between bases (0-indexed). Range: [0, length] if genome is linear else [0, length)

    All range operations use base coordinates with half-open intervals [start, end).
    """

    def __init__(self, segments: list[Segment] | None = None, circular: bool = False):
        self.circular: bool = circular
        self._validator = DefaultCoordinateValidator()
        self.root: Node | None = None
        if segments:
            for seg in segments:
                self.root = merge(self.root, Node(seg))

    def __len__(self) -> int:
        return self.length

    def __bool__(self) -> bool:
        return self.length > 0

    @property
    def length(self) -> int:
        return self.root.sub_len if self.root else 0

    def _validate_coord(self, pos: int, coord_sys: CoordinateSystem) -> None:
        """Validate coordinate based on system type."""
        self._validator.validate_position(pos, self.length, coord_sys)

    def _handle_if_last_position(self) -> tuple[Segment, int, int, int]:
        """Handle special case where position is at the end of the genome."""
        if self.root is None:
            raise IndexError("Cannot access position in empty genome")

        current_node: Node = self.root
        while current_node.right:
            current_node = current_node.right
        seg_start = self.length - current_node.segment.length
        return (
            current_node.segment,
            current_node.segment.length,
            seg_start,
            self.length,
        )

    def find_segment_at_position(
        self, pos: int, coord_sys: CoordinateSystem
    ) -> tuple[Segment, int, int, int]:
        """
        Find segment containing the base at position pos.

        Args:
            pos: Base coordinate (0 = first base)

        Returns:
            tuple: (segment, offset_within_segment, segment_start, segment_end)
        """
        self._validate_coord(pos, coord_sys)

        if pos == self.length:
            # Special case: position is at the end of the genome
            # We only get here if pos is valid <-> if genome is linear and coordinates are in GAP
            return self._handle_if_last_position()

        current_node = self.root
        prefix = 0
        while current_node:
            left_len = current_node.left.sub_len if current_node.left else 0
            if pos < prefix + left_len:
                current_node = current_node.left
            elif pos >= prefix + left_len + current_node.segment.length:
                prefix += left_len + current_node.segment.length
                current_node = current_node.right
            else:
                seg_start = prefix + left_len
                offset = pos - seg_start
                seg_end = seg_start + current_node.segment.length
                return current_node.segment, offset, seg_start, seg_end
        raise IndexError("position out of range (internal mismatch)")

    def insert_at_gap(self, pos: int, segment: NonCodingSegment) -> None:
        """Insert a segment or list of segments at genome coordinate `pos` (0..length inclusive)."""
        if not segment.is_noncoding():
            raise TypeError(f"segment must be NonCodingSegment, got {type(segment)}.")
        # Insertion is always in GAP coordinates
        self._validate_coord(pos, CoordinateSystem.GAP)

        left, right = split_by_pos(self.root, pos)
        mid = merge(None, Node(segment))
        assert mid is not None  # as mid is created from a valid segment
        # Check if previous or following segment is non-coding and make it one non coding segment
        if left:
            rightmost = left
            while rightmost.right:
                rightmost = rightmost.right
            if isinstance(rightmost.segment, NonCodingSegment):
                merged_segment = rightmost.segment.clone_with_length(
                    rightmost.segment.length + segment.length
                )
                # remove rightmost
                left = split_by_pos(left, pos - rightmost.segment.length)[0]
                mid = Node(merged_segment)
        if right:
            leftmost = right
            while leftmost.left:
                leftmost = leftmost.left
            if isinstance(leftmost.segment, NonCodingSegment):
                merged_segment = leftmost.segment.clone_with_length(
                    mid.segment.length + leftmost.segment.length
                )
                # remove leftmost
                right = split_by_pos(right, leftmost.segment.length)[1]
                mid = Node(merged_segment)
        self.root = merge(merge(left, mid), right)

    def delete_range(self, start: int, end: int) -> None:
        """Delete bases in interval [start, end). Does nothing if start == end."""
        # Deletion start is always in BASE coordinates
        self._validate_coord(start, CoordinateSystem.BASE)
        # Deletion end is always in GAP coordinates, so that last base can be deleted
        self._validate_coord(end, CoordinateSystem.GAP)

        if start == end:
            return  # nothing to delete

        if start > end:
            if self.circular:
                self.delete_range(start, self.length)
                start = 0
            else:
                raise ValueError(f"start must be < end, got {start} >= {end}")

        left, right = split_by_pos(self.root, start)
        _, right = split_by_pos(right, end - start)
        # discard middle
        self.root = merge(left, right)

    def extend_segment_at(self, pos: int, delta: int) -> None:
        """Extend the noncoding segment that contains pos.
        If pos falls inside coding segment, raises an error.
        If delta is negative, raises an error.
        Complexity: O(log n).
        """
        if delta < 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self._validate_coord(pos, CoordinateSystem.GAP)

        seg, _, seg_start, _ = self.find_segment_at_position(pos, CoordinateSystem.GAP)
        if not seg.is_noncoding():
            raise TypeError("extend_segment_at only allowed on non-coding segments")
        # isolate the segment as a middle tree
        left, bc = split_by_pos(self.root, seg_start)
        mid, right = split_by_pos(bc, seg.length)
        # mid should be a small tree containing the isolated segment(s)
        # find the (single) leftmost node in mid and mutate it
        if mid is None:
            raise RuntimeError("internal error isolating segment")
        # If mid has more than one node (shouldn't happen with our split), flatten and handle
        # We assume split returned the node as a single node treap; access its root node.
        root_node = mid  # expected single node
        # mutate length
        new_len = root_node.segment.length + delta
        root_node.segment = root_node.segment.clone_with_length(new_len)
        update_subtree_len(root_node)
        self.root = merge(merge(left, root_node), right)

    def coalesce_all(self) -> None:
        """Coalesce adjacent non-coding segments by flattening and rebuilding the tree.
        This is an O(n) operation but simple and robust. Use after many edits or when constructing final form.
        """
        segments_info: list[tuple[Segment, int, int]] = list(self.iter_segments())
        if not segments_info:
            self.root = None
            return

        merged: list[Segment] = []
        for segment, _, _ in segments_info:
            if (
                merged
                and isinstance(merged[-1], NonCodingSegment)
                and isinstance(segment, NonCodingSegment)
            ):
                # merge lengths (create new segment object to avoid mutating IDs accidentally)
                merged[-1] = merged[-1].clone_with_length(
                    merged[-1].length + segment.length
                )
            else:
                merged.append(segment)

        # rebuild tree
        root = None
        for s in merged:
            root = merge(root, Node(s))
        self.root = root

    # --- iteration / representation ---
    def iter_segments(self) -> Iterator[tuple[Segment, int, int]]:
        """Yield (segment, start, end) in order."""

        def _recur(node: Node | None, acc: int) -> Iterator[tuple[Segment, int, int]]:
            if node is None:
                return
            yield from _recur(node.left, acc)
            left_len = node.left.sub_len if node.left else 0
            seg_start = acc + left_len
            seg_end = seg_start + node.segment.length - 1
            yield node.segment, seg_start, seg_end
            yield from _recur(node.right, seg_end + 1)

        yield from _recur(self.root, 0)

    def to_list(self) -> list[Segment]:
        return [segment for segment, _, _ in self.iter_segments()]

    def __repr__(self) -> str:
        parts = [
            f"{segment}@[{start},{end}]" for segment, start, end in self.iter_segments()
        ]
        return (
            f"Genome of length {self.length}{' (circular)' if self.circular else ''}:\n\t"
            + "\n\t".join(parts)
            + "\n"
        )
