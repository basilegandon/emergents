import pytest

from emergents.genome.node import Node, merge, split_by_pos, update_subtree_len
from emergents.genome.segments import NonCodingSegment


class DummySegment(NonCodingSegment):
    def clone_with_length(self, new_length: int):
        return DummySegment(length=new_length)


def make_node(length):
    return Node(DummySegment(length=length))


def test_node_init_and_repr():
    n = make_node(5)
    assert n.segment.length == 5
    assert n.left is None
    assert n.right is None
    assert n.sub_len == 5
    assert "Node(" in repr(n)


def test_update_subtree_len_none():
    # Should not raise
    update_subtree_len(None)


def test_update_subtree_len_simple():
    n = make_node(3)
    update_subtree_len(n)
    assert n.sub_len == 3
    n_left = make_node(2)
    n.left = n_left
    update_subtree_len(n)
    assert n.sub_len == 5
    n_right = make_node(4)
    n.right = n_right
    update_subtree_len(n)
    assert n.sub_len == 9


def test_merge_none_cases():
    n1 = make_node(2)
    assert merge(None, n1) is n1
    assert merge(n1, None) is n1
    assert merge(None, None) is None


def test_merge_priority():
    n1 = make_node(2)
    n2 = make_node(3)
    n1.priority = 1
    n2.priority = 2
    merged = merge(n1, n2)
    assert merged is n1
    assert merged.right is n2 or merged.left is n2
    n1.priority = 3
    n2.priority = 2
    merged = merge(n1, n2)
    assert merged is n2


def test_split_by_pos_errors():
    n = make_node(5)
    with pytest.raises(IndexError):
        split_by_pos(n, -1)
    with pytest.raises(IndexError):
        split_by_pos(n, 6)
    n = None
    left, right = split_by_pos(n, 0)
    assert left is None
    assert right is None


def test_split_by_pos_left_right():
    def test_split_by_pos_at_segment_end():
        n = make_node(3)
        # Split at the end of the segment (should return (n, None))
        left, right = split_by_pos(n, 3)
        assert left is n
        assert right is None

    def test_split_by_pos_at_segment_start():
        n = make_node(3)
        # Split at the start of the segment (should return (None, n))
        left, right = split_by_pos(n, 0)
        assert left is None
        assert right is n

    def test_split_by_pos_beyond_segment():
        n = make_node(3)
        # Split beyond the segment length should raise
        with pytest.raises(IndexError):
            split_by_pos(n, 4)

    def test_split_by_pos_at_total_length_with_children():
        # Tree: left(3), root(5), right(2) => total 10
        root = make_node(5)
        left_child = make_node(3)
        right_child = make_node(2)
        root.left = left_child
        root.right = right_child
        update_subtree_len(root)
        # Split at total length (should return (root, None))
        left, right = split_by_pos(root, 10)
        assert left is root
        assert right is None

    n = make_node(5)
    left, right = split_by_pos(n, 0)
    assert left is None
    assert right is n
    left, right = split_by_pos(n, 5)
    assert left is n
    assert right is None


def test_split_by_pos_inside_segment():
    n = make_node(10)
    left, right = split_by_pos(n, 4)
    # left and right should be trees with one node each
    assert left is not None
    assert right is not None
    assert left.sub_len == 4
    assert right.sub_len == 6
    # The sum of lengths should be the original
    assert left.sub_len + right.sub_len == 10


def test_split_by_pos_with_children():
    # Split at position inside left child
    root = make_node(5)
    left_child = make_node(3)
    right_child = make_node(2)
    root.left = left_child
    root.right = right_child
    update_subtree_len(root)
    left, right = split_by_pos(root, 2)
    assert left is not None
    assert right is not None

    # Rebuild the tree for the next split to avoid mutation side effects
    root = make_node(5)
    left_child = make_node(3)
    right_child = make_node(2)
    root.left = left_child
    root.right = right_child
    update_subtree_len(root)
    left, right = split_by_pos(root, 9)
    assert left is not None
    assert right is not None
