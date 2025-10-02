"""
Comprehensive unit tests for emergents.genome.segments module.
Tests all segment types, cloning, promoter directions, and edge cases.
"""

import uuid

import pytest

from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)


class TestPromoterDirection:
    """Test PromoterDirection enum."""

    def test_promoter_direction_switch(self) -> None:
        """Test switching promoter directions."""
        forward = PromoterDirection.FORWARD
        reverse = PromoterDirection.REVERSE

        assert forward.switch() == reverse
        assert reverse.switch() == forward

        # Test that switch is symmetric
        assert forward.switch().switch() == forward
        assert reverse.switch().switch() == reverse

    def test_promoter_direction_equality(self) -> None:
        """Test promoter direction equality."""
        assert PromoterDirection.FORWARD == PromoterDirection.FORWARD
        assert PromoterDirection.REVERSE == PromoterDirection.REVERSE
        assert not (PromoterDirection.FORWARD is PromoterDirection.REVERSE)


class TestSegment:
    """Test base Segment class."""

    def test_segment_initialization(self) -> None:
        """Test segment initialization with different parameters."""
        # Basic initialization
        seg = Segment(length=10)
        assert seg.length == 10
        assert isinstance(seg.sid, uuid.UUID)

        # Initialization with custom UUID
        custom_uuid = uuid.uuid4()
        seg = Segment(length=5, sid=custom_uuid)
        assert seg.length == 5
        assert seg.sid == custom_uuid

    def test_segment_length_validation(self) -> None:
        """Test segment creation with various length values."""
        # Zero and negative lengths
        with pytest.raises(ValueError):
            Segment(length=0)

        with pytest.raises(ValueError):
            Segment(length=-5)

    def test_segment_unique_ids(self) -> None:
        """Test that segments get unique IDs by default."""
        seg1 = Segment(length=10)
        seg2 = Segment(length=10)

        assert seg1.sid != seg2.sid

    def test_segment_equality_and_identity(self) -> None:
        """Test segment equality and identity."""
        seg1 = Segment(length=10)
        seg2 = Segment(length=10)

        # Different segments should not be equal (different UUIDs)
        assert seg1 != seg2
        assert seg1 is not seg2

        # Same segment should be equal to itself
        assert seg1 == seg1
        assert seg1 is seg1


class TestNonCodingSegment:
    """Test NonCodingSegment class."""

    def test_noncoding_segment_is_noncoding(self) -> None:
        """Test that NonCodingSegment correctly identifies as noncoding."""
        seg = NonCodingSegment(length=10)
        assert seg.is_noncoding()

    def test_noncoding_segment_clone_with_length(self) -> None:
        """Test cloning NonCodingSegment with new length."""
        original = NonCodingSegment(length=100)
        cloned = original.clone_with_length(200)

        assert isinstance(cloned, NonCodingSegment)
        assert cloned.length == 200
        assert cloned.sid != original.sid  # Should get new UUID
        assert original.length == 100  # Original unchanged

    def test_noncoding_segment_clone_edge_cases(self) -> None:
        """Test cloning with edge case lengths."""
        original = NonCodingSegment(length=100)

        # Clone with zero length
        with pytest.raises(ValueError):
            original.clone_with_length(0)

        # Clone with negative length
        with pytest.raises(ValueError):
            original.clone_with_length(-1)


class TestCodingSegment:
    """Test CodingSegment class."""

    def test_coding_segment_initialization_defaults(self) -> None:
        """Test CodingSegment initialization with defaults."""
        seg = CodingSegment(length=100)
        assert seg.promoter_direction == PromoterDirection.FORWARD

    def test_coding_segment_initialization_with_direction(self) -> None:
        """Test CodingSegment initialization with specific promoter direction."""
        forward_seg = CodingSegment(
            length=100, promoter_direction=PromoterDirection.FORWARD
        )
        reverse_seg = CodingSegment(
            length=200, promoter_direction=PromoterDirection.REVERSE
        )

        assert forward_seg.promoter_direction == PromoterDirection.FORWARD
        assert reverse_seg.promoter_direction == PromoterDirection.REVERSE

    def test_coding_segment_is_not_noncoding(self) -> None:
        """Test that CodingSegment correctly identifies as coding."""
        seg = CodingSegment(length=10)
        assert not seg.is_noncoding()

    def test_coding_segment_clone_with_length(self) -> None:
        """Test cloning CodingSegment with new length."""
        original = CodingSegment(
            length=100, promoter_direction=PromoterDirection.REVERSE
        )
        cloned = original.clone_with_length(300)

        assert isinstance(cloned, CodingSegment)
        assert cloned.length == 300
        assert (
            cloned.promoter_direction == PromoterDirection.REVERSE
        )  # Should preserve direction
        assert cloned.sid != original.sid  # Should get new UUID
        assert original.length == 100  # Original unchanged

    def test_coding_segment_clone_preserves_direction(self) -> None:
        """Test that cloning preserves promoter direction."""
        forward_original = CodingSegment(
            length=50, promoter_direction=PromoterDirection.FORWARD
        )
        reverse_original = CodingSegment(
            length=50, promoter_direction=PromoterDirection.REVERSE
        )

        forward_cloned = forward_original.clone_with_length(100)
        reverse_cloned = reverse_original.clone_with_length(100)

        assert forward_cloned.promoter_direction == PromoterDirection.FORWARD
        assert reverse_cloned.promoter_direction == PromoterDirection.REVERSE

    def test_coding_segment_clone_edge_cases(self) -> None:
        """Test cloning with edge case lengths."""
        original = CodingSegment(
            length=100, promoter_direction=PromoterDirection.REVERSE
        )

        # Clone with zero length
        with pytest.raises(ValueError):
            original.clone_with_length(0)

        # Clone with negative length
        with pytest.raises(ValueError):
            original.clone_with_length(-1)


class TestSegmentPromoterDirectionSwitches:
    """Test promoter direction switching cases for coding segments."""

    def test_promoter_direction_switching(self) -> None:
        """Test promoter direction switching."""
        # Test multiple switches
        direction = PromoterDirection.FORWARD
        for _ in range(10):
            direction = direction.switch()
        # Should alternate, so after even number of switches, back to original
        assert direction == PromoterDirection.FORWARD

        # Test with coding segments
        coding = CodingSegment(length=10, promoter_direction=PromoterDirection.FORWARD)
        assert coding.promoter_direction == PromoterDirection.FORWARD

        # Direction switching should not affect the segment object itself
        switched = coding.promoter_direction.switch()
        assert switched == PromoterDirection.REVERSE
        assert (
            coding.promoter_direction == PromoterDirection.FORWARD
        )  # Original unchanged
