from typing import Iterator, Optional

from emergents.genome.node import Node, merge, split_by_pos, update_subtree_len
from emergents.genome.segments import NonCodingSegment, Segment

SegmentLike = Segment | list[Segment]


class Genome:
    """Genome stored as an implicit treap of Segment nodes.
    Public API provides coordinate-based insertions, deletions, lookups and coalescing.
    """

    def __init__(
        self, segments: Optional[list[Segment]] = None, circular: bool = False
    ):
        self.root: Optional[Node] = None
        self.circular: bool = circular
        if segments:
            for seg in segments:
                if seg.length <= 0:
                    raise ValueError("Segments must have positive length")
                self.root = merge(self.root, Node(seg))

    @property
    def length(self) -> int:
        return self.root.sub_len if self.root else 0

    def _validate_coord(self, pos: int, allow_end: bool = True):
        if pos < 0 or (pos > self.length if allow_end else pos >= self.length):
            raise IndexError(
                f"position {pos} out of bounds (0..{self.length}{' inclusive' if allow_end else ''})"
            )

    def find_by_coord(self, pos: int) -> tuple[Segment, int, int, int]:
        """Return (segment, offset_inside_segment, seg_start, seg_end) for given 0-based pos.
        Complexity: O(log n).
        """
        self._validate_coord(pos, allow_end=False)
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

    def __getitem__(self, pos: int) -> tuple[Segment, int, int, int]:
        """Return (segment, offset_inside_segment, seg_start, seg_end) for given 0-based pos.
        Complexity: O(log n).
        """
        segment, offset, seg_start, seg_end = self.find_by_coord(pos)
        return segment, offset, seg_start, seg_end

    def insert_at(self, pos: int, segments: SegmentLike):
        """Insert a segment or list of segments at genome coordinate `pos` (0..length inclusive)."""
        self._validate_coord(pos, allow_end=True)
        if isinstance(segments, Segment):
            segments = [segments]
        # ensure lengths positive
        for segment in segments:
            if segment.length <= 0:
                raise ValueError("Inserted segments must have positive length")
        left, right = split_by_pos(self.root, pos)
        mid = None
        for segment in segments:
            mid = merge(mid, Node(segment))
        self.root = merge(merge(left, mid), right)

    def delete_range(self, start: int, end: int):
        """Delete bases in interval [start, end). Does nothing if start == end."""
        if start == end:
            return
        self._validate_coord(start, allow_end=True)
        self._validate_coord(end, allow_end=True)
        if start >= end:
            raise ValueError("start must be < end")
        left, right = split_by_pos(self.root, start)
        _, right = split_by_pos(right, end - start)
        # discard middle
        self.root = merge(left, right)

    def extend_segment_at(self, pos: int, delta: int):
        """Extend (or shrink if delta negative) the noncoding segment that contains pos.
        If pos falls inside coding segment, raises an error.
        If delta causes length <=0, the segment is removed.
        Complexity: O(log n).
        """
        self._validate_coord(pos, allow_end=False)
        seg, _, seg_start, _ = self.find_by_coord(pos)
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
        if new_len <= 0:
            # remove this segment entirely: merge left and right
            self.root = merge(left, right)
        else:
            root_node.segment = root_node.segment.clone_with_length(new_len)
            update_subtree_len(root_node)
            self.root = merge(merge(left, root_node), right)

    def coalesce_all(self):
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

        def _recur(
            node: Optional[Node], acc: int
        ) -> Iterator[tuple[Segment, int, int]]:
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

    def __repr__(self):
        parts = [
            f"{segment}@[{start},{end}]" for segment, start, end in self.iter_segments()
        ]
        return (
            f"Genome of length {self.length}{' (circular)' if self.circular else ''}:\n\t"
            + "\n\t".join(parts)
            + "\n"
        )
