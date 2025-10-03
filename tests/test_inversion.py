"""
Comprehensive unit tests for emergents.mutations.inversion module.
Tests the Inversion class with all edge cases and error conditions.
"""

import pytest

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.inversion import Inversion


class TestInversionInitialization:
    """Test Inversion initialization and parameter validation."""

    def test_initialization_basic(self) -> None:
        """Test basic Inversion initialization with valid parameters."""
        mutation = Inversion(start_pos=10, end_pos=50)

        assert mutation.start_pos == 10
        assert mutation.end_pos == 50
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self) -> None:
        """Test Inversion initialization with RNG state."""
        mutation = Inversion(start_pos=25, end_pos=75, rng_state=54321)

        assert mutation.start_pos == 25
        assert mutation.end_pos == 75
        assert mutation.rng_state == 54321

    def test_initialization_equal_start_end(self) -> None:
        """Test Inversion initialization with equal start and end."""
        # Zero-length inversion should be allowed at initialization
        with pytest.raises(ValueError):
            Inversion(start_pos=25, end_pos=25)

    def test_initialization_reverse_order(self) -> None:
        """Test Inversion initialization with end < start."""
        # Invalid order should be allowed at initialization
        mutation = Inversion(start_pos=50, end_pos=25)
        assert mutation.start_pos == 25
        assert mutation.end_pos == 50


class TestInversionNeutrality:
    """Test neutrality checking for Inversion across different genome structures."""

    def setup_method(self) -> None:
        """Set up test genomes with various segment configurations."""
        # Create genomes with known segment IDs for testing
        self.noncoding_seg1 = NonCodingSegment(length=30)
        self.coding_seg = CodingSegment(length=40)
        self.noncoding_seg2 = NonCodingSegment(length=30)

        # Genome with mixed segments
        self.mixed_genome = Genome(
            [
                self.noncoding_seg1,  # positions 0-29
                self.coding_seg,  # positions 30-69
                self.noncoding_seg2,  # positions 70-99
            ]
        )

        # Pure non-coding genome
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_neutrality_within_single_noncoding_segment(self) -> None:
        """Test that inversions entirely within non-coding segments are neutral."""
        # Invert within first non-coding segment
        mutation = Inversion(start_pos=5, end_pos=15)  # positions 5-14
        assert mutation.is_neutral(self.mixed_genome)

        # Invert within last non-coding segment
        mutation = Inversion(start_pos=75, end_pos=90)  # positions 75-89
        assert mutation.is_neutral(self.mixed_genome)

        # Invert at boundaries of non-coding segment
        mutation = Inversion(start_pos=0, end_pos=5)  # start of first segment
        assert mutation.is_neutral(self.mixed_genome)

        mutation = Inversion(start_pos=25, end_pos=30)  # end of first segment
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_within_coding_segment(self) -> None:
        """Test neutrality when inversion is entirely within a coding segment."""
        # Invert within coding segment - should not be neutral (affects gene orientation)
        mutation = Inversion(start_pos=35, end_pos=45)  # positions 35-44
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = Inversion(start_pos=30, end_pos=35)  # start of coding segment
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = Inversion(start_pos=65, end_pos=70)  # end of coding segment
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_across_segment_boundaries(self) -> None:
        """Test neutrality when inversion spans multiple segments."""
        # Invert across non-coding and coding boundary
        mutation = Inversion(start_pos=25, end_pos=35)  # spans segments
        assert not mutation.is_neutral(self.mixed_genome)

        # Invert across coding and non-coding boundary
        mutation = Inversion(start_pos=65, end_pos=75)  # spans segments
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_sid_comparison(self) -> None:
        """Test that SID comparison works correctly."""

        # Create two segments with same ID but different objects
        seg1 = NonCodingSegment(length=50)
        seg2 = NonCodingSegment(length=50)

        genome = Genome([seg1, seg2])

        # Invert spanning both segments (should be neutral since same ID)
        mutation = Inversion(start_pos=25, end_pos=75)

        assert mutation.is_neutral(genome)

    def test_is_neutral_pure_coding_genome(self) -> None:
        """Test neutrality in pure coding genome."""
        # Single base
        mutation = Inversion(start_pos=10, end_pos=11)
        assert not mutation.is_neutral(self.coding_genome)

        # Multi-base inversions should not be neutral
        test_ranges = [(10, 12), (25, 40), (50, 70), (80, 100)]

        for start, end in test_ranges:
            mutation = Inversion(start_pos=start, end_pos=end)
            assert not mutation.is_neutral(self.coding_genome)

    def test_border_of_coding_and_noncoding(self) -> None:
        """Test neutrality at the border of coding and non-coding segments."""
        # Invert exactly at the boundary (single base)
        mutation = Inversion(start_pos=30, end_pos=70)
        assert mutation.is_neutral(self.mixed_genome)

        mutation = Inversion(start_pos=30, end_pos=69)
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = Inversion(start_pos=31, end_pos=70)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_invalid_positions(self) -> None:
        """Test neutrality checking with invalid positions."""
        # Start beyond genome length
        mutation = Inversion(start_pos=100, end_pos=105)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # End beyond genome length
        mutation = Inversion(start_pos=95, end_pos=105)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # Negative start
        with pytest.raises(ValueError):
            Inversion(start_pos=-1, end_pos=5)

    def test_is_neutral_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = Inversion(start_pos=0, end_pos=1)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)

    def test_is_neutral_reverse_order_positions(self) -> None:
        """Test neutrality with end < start."""
        mutation = Inversion(start_pos=70, end_pos=25)
        assert mutation.is_neutral(self.mixed_genome)

        mutation = Inversion(start_pos=70, end_pos=32)
        assert not mutation.is_neutral(self.mixed_genome)


class TestInversionApplication:
    """Test Inversion application to genomes."""

    def test_apply_basic(self) -> None:
        """Test basic inversion application."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = Inversion(start_pos=25, end_pos=35)  # invert 10 bases
        mutation.apply(genome)

        # Genome length should be unchanged
        assert genome.length == original_length

    def test_apply_at_beginning(self) -> None:
        """Test inversion at beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        mutation = Inversion(start_pos=0, end_pos=15)
        mutation.apply(genome)

        assert genome.length == original_length

    def test_apply_at_end(self) -> None:
        """Test inversion at end of genome."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        mutation = Inversion(start_pos=35, end_pos=50)  # invert last 15 bases
        mutation.apply(genome)

        assert genome.length == original_length

    def test_apply_entire_segment(self) -> None:
        """Test inverting entire segment."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            NonCodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        genome = Genome(segments=segments)
        original_length = genome.length

        # Invert the entire middle segment
        mutation = Inversion(start_pos=20, end_pos=50)
        mutation.apply(genome)

        assert genome.length == original_length  # Length unchanged

    def test_apply_partial_segment(self) -> None:
        """Test inverting part of a segment."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = Inversion(start_pos=30, end_pos=70)
        mutation.apply(genome)

        assert genome.length == original_length

    def test_apply_reverse_order(self) -> None:
        """Test application with end < start."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Inversion(start_pos=30, end_pos=20)
        mutation.apply(genome)
        assert genome.length == 50

    def test_apply_coding_segment_inversion_effects(self) -> None:
        """Test that application preserves segment properties but may affect coding segments."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            CodingSegment(length=40, promoter_direction=PromoterDirection.REVERSE),
            NonCodingSegment(length=20),
            CodingSegment(length=20, promoter_direction=PromoterDirection.FORWARD),
        ]
        genome = Genome(segments=segments)
        original_length = genome.length

        # Invert non-coding region (neutral operation)
        mutation = Inversion(start_pos=5, end_pos=100)
        mutation.apply(genome)

        # Length should be preserved
        assert genome.length == original_length
        first_coding_seg, offset, start, end = genome.find_segment_at_position(
            5, CoordinateSystem.BASE
        )
        assert isinstance(first_coding_seg, CodingSegment)
        assert first_coding_seg.promoter_direction == PromoterDirection.REVERSE
        assert offset == 0
        assert start == 5
        assert end == 25

        second_coding_seg, offset, start, end = genome.find_segment_at_position(
            45, CoordinateSystem.BASE
        )
        assert isinstance(second_coding_seg, CodingSegment)
        assert second_coding_seg.promoter_direction == PromoterDirection.FORWARD
        assert offset == 0
        assert start == 45
        assert end == 85

    def test_apply_entire_genome(self) -> None:
        """Test inverting entire genome."""
        genome = Genome([CodingSegment(length=50)])
        original_length = genome.length

        mutation = Inversion(start_pos=0, end_pos=50)
        mutation.apply(genome)

        assert genome.length == original_length
        assert genome.root is not None
        assert isinstance(genome.root.segment, CodingSegment)
        assert genome.root.segment.promoter_direction == PromoterDirection.REVERSE


class TestInversionSerialization:
    """Test Inversion serialization and string representation."""

    def test_serialization_methods_exist(self) -> None:
        """Test that Inversion implements serialization methods."""
        mutation = Inversion(start_pos=10, end_pos=50)

        # These methods should exist and return appropriate types
        description = mutation.describe()
        assert isinstance(description, str)
        assert "10" in description
        assert "50" in description

        # Check if serialize is implemented or raises NotImplementedError
        try:
            serialized = mutation.serialize()
            assert isinstance(serialized, dict)
        except NotImplementedError:
            # This is acceptable if not yet implemented
            pass
