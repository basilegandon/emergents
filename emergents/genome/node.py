import random
from typing import Optional

from emergents.genome.segments import Segment


class Node:
    __slots__ = ("segment", "priority", "left", "right", "sub_len")

    def __init__(self, segment: Segment):
        self.segment: Segment = segment
        self.priority: int = random.randrange(
            1 << 30
        )  # Random priority between 0 and 2^30
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.sub_len: int = segment.length

    def __repr__(self):
        return f"Node({self.segment}, prio={self.priority}, sub_len={self.sub_len})"


def update_subtree_len(node: Optional[Node]) -> None:
    """Update the subtree size information."""
    if node is None:
        return
    node.sub_len = (
        node.segment.length
        + (node.left.sub_len if node.left else 0)
        + (node.right.sub_len if node.right else 0)
    )


def merge(node_a: Optional[Node], node_b: Optional[Node]) -> Optional[Node]:
    """Merge two treaps where all positions in `node_a` are before positions in `node_b`."""
    if node_a is None:
        return node_b
    if node_b is None:
        return node_a
    if node_a.priority < node_b.priority:
        node_a.right = merge(node_a.right, node_b)
        update_subtree_len(node_a)
        return node_a
    else:
        node_b.left = merge(node_a, node_b.left)
        update_subtree_len(node_b)
        return node_b


def split_by_pos(
    root: Optional[Node], pos: int
) -> tuple[Optional[Node], Optional[Node]]:
    """
    Split tree `root` into (left, right) where left contains coordinates [0, pos)
    and right contains [pos, ...). Position is in bases, not node counts.
    This function splits a node if `pos` falls inside one segment.
    """
    if root is None:
        return None, None
    if pos < 0 or pos > root.sub_len:
        raise IndexError(f"Position {pos} out of bounds [0, {root.sub_len}]")
    left_sub: int = root.left.sub_len if root.left else 0
    seg_len: int = root.segment.length

    if pos < left_sub:
        left, right = split_by_pos(root.left, pos)
        root.left = right
        update_subtree_len(root)
        return left, root
    elif pos > left_sub + seg_len:
        left, right = split_by_pos(root.right, pos - left_sub - seg_len)
        root.right = left
        update_subtree_len(root)
        return root, right
    else:
        # split point is inside this node (including boundaries)
        offset = (
            pos - left_sub
        )  # how many bases of this node go to left side (0..seg_len)
        if offset == 0:
            # split just before this node
            left = root.left
            root.left = None
            update_subtree_len(root)
            return left, root
        elif offset == seg_len:
            # split just after this node
            right = root.right
            root.right = None
            update_subtree_len(root)
            return root, right
        else:
            # split the node into two nodes
            left_seg = root.segment.clone_with_length(offset)
            right_seg = root.segment.clone_with_length(seg_len - offset)
            left_node = Node(left_seg)
            right_node = Node(right_seg)
            # assemble left_tree = merge(root.left, left_node); right_tree = merge(right_node, root.right)
            left_tree = merge(root.left, left_node)
            right_tree = merge(right_node, root.right)
            return left_tree, right_tree
