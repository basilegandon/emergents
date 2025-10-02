"""
Comprehensive unit tests for emergents.mutations.small_deletion module.
Tests SmallDeletion class with all edge cases and error conditions.
"""

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.small_deletion import SmallDeletion


class TestSmallDeletionInitialization:
    """Test SmallDeletion initialization."""

    def test_basic_initialization(self):
        """Test basic SmallDeletion initialization."""
        mutation = SmallDeletion(position=50, length=10)

        assert mutation.position == 50
        assert mutation.length == 10
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self):
        """Test SmallDeletion initialization with RNG state."""
        mutation = SmallDeletion(position=25, length=5, rng_state=12345)

        assert mutation.position == 25
        assert mutation.length == 5
        assert mutation.rng_state == 12345

    def test_initialization_edge_cases(self):
        """Test SmallDeletion initialization edge cases."""
        # Zero position
        mutation = SmallDeletion(position=0, length=1)
        assert mutation.position == 0
        assert mutation.length == 1

    def test_initialization_invalid_length(self):
        """Test SmallDeletion initialization with invalid length."""
        with pytest.raises(ValueError):
            SmallDeletion(position=10, length=0)

        with pytest.raises(ValueError):
            SmallDeletion(position=10, length=-5)


class TestSmallDeletionNeutrality:
    """Test SmallDeletion neutrality checking."""

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

        # Pure non-coding genome
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_is_neutral_within_single_noncoding_segment(self):
        """Test neutrality when deletion is entirely within a single non-coding segment."""
        # Delete within first non-coding segment
        mutation = SmallDeletion(position=5, length=10)  # positions 5-14
        assert mutation.is_neutral(self.mixed_genome)

        # Delete within last non-coding segment
        mutation = SmallDeletion(position=75, length=15)  # positions 75-89
        assert mutation.is_neutral(self.mixed_genome)

        # Delete at boundaries of non-coding segment
        mutation = SmallDeletion(position=0, length=5)  # start of first segment
        assert mutation.is_neutral(self.mixed_genome)

        mutation = SmallDeletion(position=25, length=5)  # end of first segment
        assert mutation.is_neutral(self.mixed_genome)

        mutation = SmallDeletion(
            position=26, length=5
        )  # end of first segment + first base of coding
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_within_coding_segment(self):
        """Test neutrality when deletion is entirely within a coding segment."""
        # Delete within coding segment - should not be neutral
        mutation = SmallDeletion(position=35, length=10)  # positions 35-44
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = SmallDeletion(position=30, length=5)  # start of coding segment
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = SmallDeletion(position=67, length=5)  # end of coding segment
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_across_segment_boundaries(self):
        """Test neutrality when deletion spans multiple segments."""
        # Delete across non-coding and coding boundary
        mutation = SmallDeletion(position=25, length=10)  # spans segments
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete across coding and non-coding boundary
        mutation = SmallDeletion(position=65, length=10)  # spans segments
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_single_base_deletions(self):
        """Test neutrality for single base deletions."""
        # Single base in non-coding
        mutation = SmallDeletion(position=10, length=1)
        assert mutation.is_neutral(self.mixed_genome)

        # Single base in coding
        mutation = SmallDeletion(position=40, length=1)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_pure_noncoding_genome(self):
        """Test neutrality in pure non-coding genome."""
        # All deletions should be neutral (single segment)
        positions_and_lengths = [(0, 10), (25, 15), (50, 20), (90, 10)]

        for pos, length in positions_and_lengths:
            mutation = SmallDeletion(position=pos, length=length)
            assert mutation.is_neutral(self.noncoding_genome)

        mutation = SmallDeletion(position=95, length=10)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.noncoding_genome)

    def test_is_neutral_pure_coding_genome(self):
        """Test neutrality in pure coding genome."""
        # No deletions should be neutral (single coding segment)
        positions_and_lengths = [(0, 10), (25, 15), (50, 20), (90, 10)]

        for pos, length in positions_and_lengths:
            mutation = SmallDeletion(position=pos, length=length)
            assert not mutation.is_neutral(self.coding_genome)

    def test_is_neutral_empty_genome(self):
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = SmallDeletion(position=0, length=1)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)


class TestSmallDeletionApplication:
    """Test SmallDeletion application."""

    def test_apply_basic(self):
        """Test basic deletion application."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = SmallDeletion(position=25, length=10)
        mutation.apply(genome)

        # Genome should be shorter
        assert genome.length == original_length - 10

    def test_apply_at_beginning(self):
        """Test deletion at beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallDeletion(position=0, length=15)
        mutation.apply(genome)

        assert genome.length == 35

    def test_apply_at_end(self):
        """Test deletion at end of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallDeletion(position=35, length=15)  # delete last 15 bases
        mutation.apply(genome)

        assert genome.length == 35

    def test_apply_entire_segment(self):
        """Test deleting entire segment."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            CodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        genome = Genome(segments=segments)

        # Delete the entire middle segment
        mutation = SmallDeletion(position=0, length=20)
        mutation.apply(genome)

        assert genome.length == 50  # 30 + 20

    def test_apply_across_segments(self):
        """Test deletion across multiple segments."""
        segments: list[Segment] = [
            NonCodingSegment(length=30),
            NonCodingSegment(length=40),
            NonCodingSegment(length=30),
        ]
        genome = Genome(segments=segments)

        # Delete across all three segments
        mutation = SmallDeletion(position=10, length=80)
        mutation.apply(genome)

        assert genome.length == 20  # 10 bases from start + 10 bases from end

    def test_apply_entire_genome(self):
        """Test deleting entire genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallDeletion(position=0, length=50)
        mutation.apply(genome)

        assert genome.length == 0
        assert genome.root is None


class TestSmallDeletionEdgeCases:
    """Test edge cases for SmallDeletion."""

    def test_deletion_single_base_genome(self):
        """Test deletion in single-base genome."""
        genome = Genome([NonCodingSegment(length=1)])

        # Delete the only base
        mutation = SmallDeletion(position=0, length=1)
        assert mutation.is_neutral(genome)

        mutation.apply(genome)
        assert genome.length == 0
