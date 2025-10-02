"""
Comprehensive unit tests for emergents.mutations.deletion module.
Tests Deletion class with all edge cases and error conditions.
"""

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.deletion import Deletion


class TestDeletionInitialization:
    """Test Deletion initialization."""

    def test_basic_initialization(self):
        """Test basic Deletion initialization."""
        mutation = Deletion(start_pos=10, end_pos=50)

        assert mutation.start_pos == 10
        assert mutation.end_pos == 50
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self):
        """Test Deletion initialization with RNG state."""
        mutation = Deletion(start_pos=25, end_pos=75, rng_state=67890)

        assert mutation.start_pos == 25
        assert mutation.end_pos == 75
        assert mutation.rng_state == 67890

    def test_initialization_edge_cases(self):
        """Test Deletion initialization edge cases."""
        # Zero start
        mutation = Deletion(start_pos=0, end_pos=20)
        assert mutation.start_pos == 0
        assert mutation.end_pos == 20

        # Adjacent positions (length 2)
        mutation = Deletion(start_pos=10, end_pos=11)
        assert mutation.start_pos == 10
        assert mutation.end_pos == 11

    def test_initialization_equal_start_end(self):
        """Test Deletion initialization with equal start and end."""
        # One-length deletion should be allowed at initialization
        mutation = Deletion(start_pos=25, end_pos=25)
        assert mutation.start_pos == 25
        assert mutation.end_pos == 25

    def test_initialization_reverse_order(self):
        """Test Deletion initialization with end < start."""
        # Invalid order should be allowed at initialization
        mutation = Deletion(start_pos=50, end_pos=25)
        assert mutation.start_pos == 50
        assert mutation.end_pos == 25


class TestDeletionNeutrality:
    """Test Deletion neutrality checking."""

    def setup_method(self):
        """Set up test genomes."""
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

        self.circular_mixed_genome = Genome(
            [
                self.noncoding_seg1,  # positions 0-29
                self.coding_seg,  # positions 30-69
                self.noncoding_seg2,  # positions 70-99
            ],
            circular=True,
        )

        # Pure non-coding genome
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_is_neutral_within_single_noncoding_segment(self):
        """Test neutrality when deletion is entirely within a single non-coding segment."""
        # Delete within first non-coding segment
        mutation = Deletion(start_pos=5, end_pos=14)  # positions 5-14
        assert mutation.is_neutral(self.mixed_genome)

        # Delete within last non-coding segment
        mutation = Deletion(start_pos=75, end_pos=89)  # positions 75-89
        assert mutation.is_neutral(self.mixed_genome)

        # Delete at boundaries of non-coding segment
        mutation = Deletion(start_pos=0, end_pos=5)  # start of first segment
        assert mutation.is_neutral(self.mixed_genome)

        mutation = Deletion(start_pos=0, end_pos=29)  # end of first segment
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_within_coding_segment(self):
        """Test neutrality when deletion is entirely within a coding segment."""
        # Delete within coding segment - should not be neutral
        mutation = Deletion(start_pos=35, end_pos=44)  # positions 35-44
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = Deletion(start_pos=30, end_pos=35)  # start of coding segment
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = Deletion(start_pos=20, end_pos=69)  # end of coding segment
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_across_segment_boundaries(self):
        """Test neutrality when deletion spans multiple segments."""
        # Delete across non-coding and coding boundary
        mutation = Deletion(start_pos=0, end_pos=35)  # spans segments
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete across coding and non-coding boundary
        mutation = Deletion(start_pos=0, end_pos=75)  # spans segments
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_different_noncoding_segments(self):
        """Test neutrality when deletion spans different non-coding segments."""
        # Create genome with two different non-coding segments
        seg1 = NonCodingSegment(length=50)
        seg2 = NonCodingSegment(length=50)  # Different ID
        genome = Genome([seg1, seg2])

        # Delete spanning both segments (different IDs) - should not be neutral
        mutation = Deletion(start_pos=0, end_pos=75)
        assert not mutation.is_neutral(genome)

        # but deletion should be use after coalescing
        genome.coalesce_all()
        assert mutation.is_neutral(genome)

    def test_is_neutral_pure_noncoding_genome(self):
        """Test neutrality in pure non-coding genome."""
        # All deletions should be neutral (single segment)
        test_ranges = [(0, 10), (25, 40), (50, 70), (80, 99)]

        for start, end in test_ranges:
            mutation = Deletion(start_pos=start, end_pos=end)
            assert mutation.is_neutral(self.noncoding_genome)

    def test_is_neutral_pure_coding_genome(self):
        """Test neutrality in pure coding genome."""
        # No deletions should be neutral (single coding segment)
        test_ranges = [(0, 10), (25, 40), (50, 70), (80, 99)]

        for start, end in test_ranges:
            mutation = Deletion(start_pos=start, end_pos=end)
            assert not mutation.is_neutral(self.coding_genome)

    def test_is_neutral_invalid_positions(self):
        """Test neutrality checking with invalid positions."""
        # Start beyond genome length
        mutation = Deletion(start_pos=100, end_pos=105)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # End beyond genome length
        mutation = Deletion(start_pos=98, end_pos=105)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # Negative start
        mutation = Deletion(start_pos=-1, end_pos=5)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_empty_genome(self):
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = Deletion(start_pos=0, end_pos=0)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)

    def test_is_neutral_one_length_deletion(self):
        """Test neutrality with one-length deletion."""
        # One-length deletion (start == end)
        mutation = Deletion(start_pos=29, end_pos=29)

        # Should not raise error for one-length
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_reverse_order_positions(self):
        """Test neutrality with end < start."""
        # Should raise error for invalid order if genome is non-circular
        mutation = Deletion(start_pos=90, end_pos=10)

        with pytest.raises(ValueError):
            mutation.is_neutral(self.mixed_genome)

        assert mutation.is_neutral(self.circular_mixed_genome)

        mutation = Deletion(start_pos=90, end_pos=30)
        assert not mutation.is_neutral(self.circular_mixed_genome)

        mutation = Deletion(start_pos=94, end_pos=93)
        assert not mutation.is_neutral(self.circular_mixed_genome)


class TestDeletionApplication:
    """Test Deletion application."""

    def test_apply_basic(self):
        """Test basic deletion application."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = Deletion(
            start_pos=20, end_pos=35
        )  # delete 16 bases (inclusive range)
        mutation.apply(genome)

        # Genome should be shorter
        assert genome.length == original_length - 16

    def test_apply_at_beginning(self):
        """Test deletion at beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Deletion(start_pos=0, end_pos=15)
        mutation.apply(genome)

        assert genome.length == 34  # 50 - 16 (inclusive range 0-15)

    def test_apply_at_end(self):
        """Test deletion at end of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Deletion(start_pos=35, end_pos=49)  # delete last 15 bases
        mutation.apply(genome)

        assert genome.length == 35

    def test_apply_entire_segment(self):
        """Test deleting entire segment."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            NonCodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        genome = Genome(segments=segments)

        # Delete the entire middle segment
        mutation = Deletion(start_pos=20, end_pos=49)
        mutation.apply(genome)

        assert genome.length == 40  # 20 + 20

    def test_apply_reverse_order(self):
        """Test application with end < start."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            CodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        circular_genome = Genome(segments=segments, circular=True)

        mutation = Deletion(start_pos=50, end_pos=19)
        mutation.apply(circular_genome)

        assert circular_genome.length == 30

    def test_apply_entire_genome(self):
        """Test deleting entire genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Deletion(start_pos=0, end_pos=49)
        mutation.apply(genome)

        assert genome.length == 0
        assert genome.root is None
