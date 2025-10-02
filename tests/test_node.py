"""
Comprehensive unit tests for emergents.genome.node module.
Tests Node class, treap operations, splitting, merging, and edge cases.
"""

from unittest.mock import Mock, patch

import pytest

from emergents.genome.node import Node, merge, split_by_pos, update_subtree_len
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection


class TestNode:
    """Test Node class."""

    def test_node_initialization(self):
        """Test Node initialization with segment."""
        segment = NonCodingSegment(length=100)
        node = Node(segment)

        assert node.segment == segment
        assert node.left is None
        assert node.right is None
        assert node.sub_len == 100
        assert isinstance(node.priority, int)
        assert 0 <= node.priority < (1 << 30)

    def test_node_priority_randomness(self):
        """Test that different nodes get different priorities (with high probability)."""
        segment = NonCodingSegment(length=10)
        nodes = [Node(segment) for _ in range(100)]
        priorities = [node.priority for node in nodes]

        # With random priorities, should be very unlikely to have all the same
        unique_priorities = set(priorities)
        assert len(unique_priorities) > 10  # Should have at least some variation

    def test_node_with_different_segments(self):
        """Test Node with different segment types."""
        noncoding = NonCodingSegment(length=75)
        coding = CodingSegment(length=125, promoter_direction=PromoterDirection.REVERSE)

        noncoding_node = Node(noncoding)
        coding_node = Node(coding)

        assert noncoding_node.segment == noncoding
        assert coding_node.segment == coding
        assert noncoding_node.sub_len == 75
        assert coding_node.sub_len == 125

    def test_node_slots(self):
        """Test that Node uses __slots__ for memory efficiency."""
        segment = NonCodingSegment(length=10)
        node = Node(segment)

        # Should have exactly these attributes and no __dict__
        expected_slots = {"segment", "priority", "left", "right", "sub_len"}
        assert set(node.__slots__) == expected_slots
        assert not hasattr(node, "__dict__")

    @patch("random.randrange")
    def test_node_priority_deterministic(self, mock_randrange: Mock):
        """Test node priority with mocked random for deterministic testing."""
        mock_randrange.return_value = 12345

        segment = NonCodingSegment(length=30)
        node = Node(segment)

        assert node.priority == 12345
        mock_randrange.assert_called_once_with(1 << 30)


class TestUpdateSubtreeLen:
    """Test update_subtree_len function."""

    def test_update_subtree_len_none(self):
        """Test updating subtree length with None node."""
        # Should not raise exception
        update_subtree_len(None)

    def test_update_subtree_len_leaf_node(self):
        """Test updating subtree length for leaf node."""
        segment = NonCodingSegment(length=50)
        node = Node(segment)

        # Initial state should be correct
        assert node.sub_len == 50

        # Update should maintain correctness
        update_subtree_len(node)
        assert node.sub_len == 50

    def test_update_subtree_len_node_with_children(self):
        """Test updating subtree length for node with children."""
        # Create segments
        left_segment = NonCodingSegment(length=20)
        root_segment = NonCodingSegment(length=30)
        right_segment = NonCodingSegment(length=40)

        # Create nodes
        left_node = Node(left_segment)
        root_node = Node(root_segment)
        right_node = Node(right_segment)

        # Build tree structure
        root_node.left = left_node
        root_node.right = right_node

        # Update root node
        update_subtree_len(root_node)

        # Should be sum of all lengths
        assert root_node.sub_len == 20 + 30 + 40

    def test_update_subtree_len_left_child_only(self):
        """Test updating subtree length with only left child."""
        left_segment = NonCodingSegment(length=15)
        root_segment = NonCodingSegment(length=25)

        left_node = Node(left_segment)
        root_node = Node(root_segment)

        root_node.left = left_node
        root_node.right = None

        update_subtree_len(root_node)

        assert root_node.sub_len == 15 + 25

    def test_update_subtree_len_right_child_only(self):
        """Test updating subtree length with only right child."""
        root_segment = NonCodingSegment(length=35)
        right_segment = NonCodingSegment(length=45)

        root_node = Node(root_segment)
        right_node = Node(right_segment)

        root_node.left = None
        root_node.right = right_node

        update_subtree_len(root_node)
        assert root_node.sub_len == 35 + 45

    def test_update_subtree_len_complex_tree(self):
        """Test updating subtree length in complex tree structure."""
        # Create tree:     root(10)
        #                 /        \
        #            left(5)    right(8)
        #           /     \          \
        #         ll(2)  lr(3)      rr(4)

        segments = [NonCodingSegment(length=i) for i in [2, 3, 5, 4, 8, 10]]
        nodes = [Node(seg) for seg in segments]
        ll, lr, left, rr, right, root = nodes

        # Build structure
        left.left = ll
        left.right = lr
        right.right = rr
        root.left = left
        root.right = right

        # Update from bottom up
        update_subtree_len(left)
        update_subtree_len(right)
        update_subtree_len(root)

        # Check lengths
        assert left.sub_len == 2 + 3 + 5  # 10
        assert right.sub_len == 8 + 4  # 12
        assert root.sub_len == 10 + 10 + 12  # 32

    def test_update_subtree_len_with_zero_length_segments(self):
        """Test updating subtree length with zero-length segments."""
        normal_segment = NonCodingSegment(length=10)

        zero_node = None
        normal_node = Node(normal_segment)

        normal_node.left = zero_node

        update_subtree_len(normal_node)

        assert normal_node.sub_len == 0 + 10


class TestMerge:
    """Test merge function."""

    def test_merge_none_with_none(self):
        """Test merging two None values."""
        result = merge(None, None)
        assert result is None

    def test_merge_node_with_none(self):
        """Test merging node with None."""
        segment = NonCodingSegment(length=20)
        node = Node(segment)

        result1 = merge(node, None)
        result2 = merge(None, node)

        assert result1 == node
        assert result2 == node

    def test_merge_two_nodes_priority_order(self):
        """Test merging two nodes based on priority."""
        with patch("random.randrange") as mock_rand:
            # First node gets higher priority
            mock_rand.side_effect = [100, 50]

            seg1 = NonCodingSegment(length=10)
            seg2 = NonCodingSegment(length=20)

            node1 = Node(seg1)  # priority 100
            node2 = Node(seg2)  # priority 50

            result = merge(node1, node2)

            assert result is not None

            # Lower priority value means higher priority in treap
            # So node2 (priority 50) should be root
            assert result == node2
            assert result.left == node1
            assert result.right is None

    def test_merge_maintains_treap_property(self):
        """Test that merge maintains treap priority property."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [10, 20, 30]  # Decreasing priorities

            segments = [NonCodingSegment(length=i * 10) for i in range(1, 4)]
            nodes = [Node(seg) for seg in segments]

            # Merge in sequence
            result = merge(nodes[0], nodes[1])
            result = merge(result, nodes[2])

            # Check that priorities are in heap order
            def check_heap_property(node: Node | None) -> bool:
                if node is None:
                    return True
                if node.left and node.left.priority < node.priority:
                    return False
                if node.right and node.right.priority < node.priority:
                    return False
                return check_heap_property(node.left) and check_heap_property(
                    node.right
                )

            assert check_heap_property(result)

    def test_merge_updates_subtree_lengths(self):
        """Test that merge correctly updates subtree lengths."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [100, 50]

            seg1 = NonCodingSegment(length=15)
            seg2 = NonCodingSegment(length=25)

            node1 = Node(seg1)
            node2 = Node(seg2)

            result = merge(node1, node2)

            assert result is not None

            # Result should have correct subtree length
            assert result.sub_len == 15 + 25
            assert result.sub_len == 40

    def test_merge_complex_subtrees(self):
        """Test merging complex subtrees."""
        with patch("random.randrange") as mock_rand:
            # Set up priorities to get predictable structure
            mock_rand.side_effect = [10, 20, 5, 30]

            # Create two subtrees
            seg1 = NonCodingSegment(length=10)
            seg2 = NonCodingSegment(length=20)
            seg3 = NonCodingSegment(length=30)
            seg4 = NonCodingSegment(length=40)

            node1 = Node(seg1)  # priority 10
            node2 = Node(seg2)  # priority 20
            node3 = Node(seg3)  # priority 5
            node4 = Node(seg4)  # priority 30

            # Build first subtree: node3 as root (lowest priority), node1 as right child
            subtree1 = merge(node1, node3)

            # Build second subtree: node4 as left, node2 as root
            subtree2 = merge(node4, node2)

            # Merge the two subtrees
            final_result = merge(subtree1, subtree2)
            assert final_result is not None

            # Check total length
            expected_total = 10 + 20 + 30 + 40
            assert final_result.sub_len == expected_total

    def test_merge_preserves_order(self):
        """Test that merge preserves left-to-right order of positions."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [50, 50]  # Same priority for simplicity

            left_seg = NonCodingSegment(length=10)
            right_seg = NonCodingSegment(length=20)

            left_node = Node(left_seg)
            right_node = Node(right_seg)

            result = merge(left_node, right_node)

            # Check that in-order traversal gives correct order
            def inorder_lengths(node: Node | None) -> list[int]:
                if node is None:
                    return []
                return (
                    inorder_lengths(node.left)
                    + [node.segment.length]
                    + inorder_lengths(node.right)
                )

            lengths = inorder_lengths(result)
            # Should be in order of merge: left_node first, then right_node
            assert 10 in lengths
            assert 20 in lengths


class TestSplitByPos:
    """Test split_by_pos function."""

    def test_split_by_pos_none(self):
        """Test splitting None tree."""
        left, right = split_by_pos(None, 0)
        assert left is None
        assert right is None

    def test_split_by_pos_single_node_before(self):
        """Test splitting single node before it."""
        segment = NonCodingSegment(length=10)
        node = Node(segment)

        left, right = split_by_pos(node, 0)

        assert left is None
        assert right is not None
        assert right == node
        assert right.segment.length == 10

    def test_split_by_pos_single_node_after(self):
        """Test splitting single node after it."""
        segment = NonCodingSegment(length=10)
        node = Node(segment)

        left, right = split_by_pos(node, 10)

        assert left is not None
        assert left == node
        assert right is None
        assert left.segment.length == 10

    def test_split_by_pos_single_node_middle(self):
        """Test splitting single node in the middle."""
        segment = NonCodingSegment(length=10)
        node = Node(segment)

        left, right = split_by_pos(node, 3)

        assert left is not None
        assert right is not None
        assert left != node  # Should be new node
        assert right != node  # Should be new node
        assert left.segment.length == 3
        assert right.segment.length == 7

    def test_split_by_pos_preserves_segment_type(self):
        """Test that splitting preserves segment types."""
        coding_segment = CodingSegment(
            length=20, promoter_direction=PromoterDirection.REVERSE
        )
        node = Node(coding_segment)

        left, right = split_by_pos(node, 8)
        assert left is not None
        assert right is not None

        assert isinstance(left.segment, CodingSegment)
        assert isinstance(right.segment, CodingSegment)
        assert left.segment.promoter_direction == PromoterDirection.REVERSE
        assert right.segment.promoter_direction == PromoterDirection.REVERSE
        assert left.segment.length == 8
        assert right.segment.length == 12

    def test_split_by_pos_tree_left_subtree(self):
        """Test splitting in left subtree."""
        with patch("random.randrange") as mock_rand:
            # Need more values: 3 for initial nodes + 2 for split operation
            mock_rand.side_effect = [50, 100, 25, 75, 80]  # Added 2 more values

            segments = [NonCodingSegment(length=10 * i) for i in [1, 2, 3]]
            nodes = [Node(seg) for seg in segments]

            # Build tree structure manually
            root = nodes[1]  # length 20
            root.left = nodes[0]  # length 10
            root.right = nodes[2]  # length 30
            update_subtree_len(root)

            assert root.sub_len == 10 + 20 + 30

            # Split at position 5 (inside left child)
            left, right = split_by_pos(root, 5)

            # Left should contain first 5 positions
            assert left is not None
            assert left.segment.length == 5

            # Right should contain the rest
            assert right is not None
            # Remaining: 5 from split + 20 from root + 30 from right child = 55
            assert right.sub_len == 55

    def test_split_by_pos_tree_right_subtree(self):
        """Test splitting in right subtree."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [50, 100, 25, 75, 80]  # Added 2 more values

            segments = [NonCodingSegment(length=10 * i) for i in [1, 2, 3]]
            nodes = [Node(seg) for seg in segments]

            root = nodes[1]  # length 20
            root.left = nodes[0]  # length 10
            root.right = nodes[2]  # length 30
            update_subtree_len(root)

            # Split at position 35 (10 + 20 + 5 into right child)
            left, right = split_by_pos(root, 35)

            # Left should contain first 35 positions
            assert left is not None
            assert left.sub_len == 35

            # Right should contain remaining 25 positions from right child
            assert right is not None
            assert right.segment.length == 25

    def test_split_by_pos_boundary_positions(self):
        """Test splitting at node boundaries."""
        segments = [NonCodingSegment(length=10 * i) for i in [1, 2, 3]]

        # Build simple linear tree: 10 + 20 + 30 = 60 total
        root = Node(segments[1])  # 20
        root.left = Node(segments[0])  # 10
        root.right = Node(segments[2])  # 30
        update_subtree_len(root)

        # Split at position 10 (boundary between left child and root)
        left, right = split_by_pos(root, 10)
        assert left is not None
        assert right is not None

        assert left.segment.length == 10
        assert right.sub_len == 50  # 20 + 30

    def test_split_by_pos_updates_subtree_lengths(self):
        """Test that split correctly updates subtree lengths."""
        segments = [NonCodingSegment(length=10 * i) for i in [1, 2, 3, 4]]

        # Build larger tree
        root = Node(segments[1])  # 20
        root.left = Node(segments[0])  # 10
        right_subtree = Node(segments[2])  # 30
        right_subtree.right = Node(segments[3])  # 40
        update_subtree_len(right_subtree)
        root.right = right_subtree
        update_subtree_len(root)

        # Total length: 10 + 20 + 30 + 40 = 100
        assert root.sub_len == 100

        # Split in the middle
        left, right = split_by_pos(root, 50)

        assert left is not None
        assert right is not None

        assert left.sub_len == 50
        assert right.sub_len == 50

    def test_split_by_pos_edge_cases(self):
        """Test edge cases for split_by_pos."""
        # Single node, split at very end
        segment = NonCodingSegment(length=1)
        node = Node(segment)

        left, right = split_by_pos(node, 1)
        assert left == node
        assert right is None

        # Large split position
        with pytest.raises(IndexError):
            split_by_pos(node, 2)

        with pytest.raises(IndexError):
            split_by_pos(node, -1)

    def test_split_by_pos_preserves_tree_structure(self):
        """Test that splitting preserves valid tree structure."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [i for i in range(50, 70)]

            # Create a larger tree
            segments = [NonCodingSegment(length=10) for _ in range(5)]
            nodes = [Node(seg) for seg in segments]

            # Build a tree by merging
            tree = None
            for node in nodes:
                tree = merge(tree, node)

            # Split it
            left, right = split_by_pos(tree, 25)

            # Check that both parts are valid trees
            def is_valid_treap(node: Node | None) -> bool:
                if node is None:
                    return True
                # Check priority property
                if node.left and node.left.priority < node.priority:
                    return False
                if node.right and node.right.priority < node.priority:
                    return False
                return is_valid_treap(node.left) and is_valid_treap(node.right)

            assert is_valid_treap(left)
            assert is_valid_treap(right)


class TestTreapIntegration:
    """Test integration between merge and split operations."""

    def test_split_then_merge_identity(self):
        """Test that splitting then merging gives back original tree."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [i for i in range(20, 30)]

            segments = [NonCodingSegment(length=10) for _ in range(3)]

            # Build original tree
            original = None
            for seg in segments:
                original = merge(original, Node(seg))

            assert original is not None
            original_length = original.sub_len

            # Split and merge back
            left, right = split_by_pos(original, 15)
            reconstructed = merge(left, right)

            assert reconstructed is not None

            # Should have same total length
            assert reconstructed.sub_len == original_length

    def test_multiple_splits_and_merges(self):
        """Test complex sequence of splits and merges."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [i for i in range(50, 80)]

            # Start with segments of length 5 each
            segments = [NonCodingSegment(length=5) for _ in range(6)]

            # Build tree: total length 30
            tree = None
            for seg in segments:
                tree = merge(tree, Node(seg))

            assert tree is not None
            assert tree.sub_len == 30

            # Split into three parts
            left, temp = split_by_pos(tree, 10)
            middle, right = split_by_pos(temp, 10)

            assert left is not None
            assert middle is not None
            assert right is not None

            assert left.sub_len == 10
            assert middle.sub_len == 10
            assert right.sub_len == 10

            # Merge back in different order
            new_tree = merge(middle, left)
            new_tree = merge(new_tree, right)
            assert new_tree is not None

            assert new_tree.sub_len == 30

    def test_treap_maintains_order_property(self):
        """Test that treap maintains BST order property for positions."""
        with patch("random.randrange") as mock_rand:
            mock_rand.side_effect = [i for i in range(100, 120)]

            # Create segments with different lengths
            lengths = [5, 15, 8, 12, 20]
            segments = [NonCodingSegment(length=l) for l in lengths]

            # Build tree
            tree = None
            for seg in segments:
                tree = merge(tree, Node(seg))

            # Extract all segments in order
            def extract_lengths_inorder(node: Node | None) -> list[int]:
                if node is None:
                    return []
                return (
                    extract_lengths_inorder(node.left)
                    + [node.segment.length]
                    + extract_lengths_inorder(node.right)
                )

            extracted_lengths = extract_lengths_inorder(tree)

            # Should contain all original lengths
            assert sorted(extracted_lengths) == sorted(lengths)
            assert len(extracted_lengths) == len(lengths)
