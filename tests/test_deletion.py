"""
Comprehensive unit tests for emergents.mutations.deletion module.
Tests the Deletion class with all edge cases and error conditions.
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
    """Test Deletion initialization and parameter validation."""

    def test_initialization_basic(self) -> None:
        """Test basic Deletion initialization with valid parameters."""
        mutation = Deletion(start_pos=10, end_pos=50)

        assert mutation.start_pos == 10
        assert mutation.end_pos == 50
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self) -> None:
        """Test Deletion initialization with RNG state parameter."""
        mutation = Deletion(start_pos=25, end_pos=75, rng_state=67890)

        assert mutation.start_pos == 25
        assert mutation.end_pos == 75
        assert mutation.rng_state == 67890

    def test_initialization_edge_cases(self) -> None:
        """Test Deletion initialization with edge case parameters."""
        # Zero start position should be valid
        mutation = Deletion(start_pos=0, end_pos=20)
        assert mutation.start_pos == 0
        assert mutation.end_pos == 20

        # Adjacent positions (length 2) should be valid
        mutation = Deletion(start_pos=10, end_pos=11)
        assert mutation.start_pos == 10
        assert mutation.end_pos == 11

    def test_initialization_equal_start_end(self) -> None:
        """Test Deletion initialization with equal start and end positions."""
        # Single-base deletion should be allowed at initialization
        mutation = Deletion(start_pos=25, end_pos=25)
        assert mutation.start_pos == 25
        assert mutation.end_pos == 25

    def test_initialization_reverse_order(self) -> None:
        """Test Deletion initialization with end position before start."""
        # Reverse order should be allowed at initialization
        mutation = Deletion(start_pos=50, end_pos=25)
        assert mutation.start_pos == 50
        assert mutation.end_pos == 25


class TestDeletionNeutrality:
    """Test neutrality checking for Deletion across different genome structures."""

    def setup_method(self) -> None:
        """Set up test genomes with various segment configurations."""
        # Create distinct segments for testing
        self.noncoding_seg1 = NonCodingSegment(length=30)
        self.coding_seg = CodingSegment(length=40)
        self.noncoding_seg2 = NonCodingSegment(length=30)

        # Mixed genome with alternating segment types
        self.mixed_genome = Genome(
            [
                self.noncoding_seg1,  # positions 0-29
                self.coding_seg,  # positions 30-69
                self.noncoding_seg2,  # positions 70-99
            ]
        )

        # Circular genome for testing circular deletions
        self.circular_mixed_genome = Genome(
            [
                self.noncoding_seg1,  # positions 0-29
                self.coding_seg,  # positions 30-69
                self.noncoding_seg2,  # positions 70-99
            ],
            circular=True,
        )

        # Pure non-coding genome for baseline testing
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome for contrast testing
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_neutrality_within_single_noncoding_segment(self) -> None:
        """Test that deletions entirely within non-coding segments are neutral."""
        # Delete within first non-coding segment
        mutation = Deletion(start_pos=5, end_pos=14)  # positions 5-14
        assert mutation.is_neutral(self.mixed_genome)

        # Delete within last non-coding segment
        mutation = Deletion(start_pos=75, end_pos=89)  # positions 75-89
        assert mutation.is_neutral(self.mixed_genome)

        # Delete at boundaries of non-coding segment
        mutation = Deletion(start_pos=0, end_pos=5)  # start of first segment
        assert mutation.is_neutral(self.mixed_genome)

        mutation = Deletion(start_pos=0, end_pos=29)  # entire first segment
        assert mutation.is_neutral(self.mixed_genome)

    def test_neutrality_within_coding_segment(self) -> None:
        """Test that deletions entirely within coding segments are not neutral."""
        # Delete within middle of coding segment
        mutation = Deletion(start_pos=35, end_pos=44)  # positions 35-44
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete at start of coding segment
        mutation = Deletion(start_pos=30, end_pos=35)
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete at end of coding segment
        mutation = Deletion(start_pos=65, end_pos=69)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_across_segment_boundaries(self) -> None:
        """Test that deletions spanning segment boundaries are not neutral when touching coding."""
        # Delete across non-coding to coding boundary
        mutation = Deletion(start_pos=25, end_pos=35)  # spans boundary
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete across coding to non-coding boundary
        mutation = Deletion(start_pos=65, end_pos=75)  # spans boundary
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_different_noncoding_segments(self) -> None:
        """Test neutrality when deletion spans different non-coding segments."""
        # Create genome with two distinct non-coding segments
        seg1 = NonCodingSegment(length=50)
        seg2 = NonCodingSegment(length=50)  # Different segment instance
        genome = Genome([seg1, seg2])

        # Delete spanning both segments (different IDs) - should not be neutral
        mutation = Deletion(start_pos=25, end_pos=75)
        assert not mutation.is_neutral(genome)

        # After coalescing, deletion should be neutral
        genome.coalesce_all()
        assert mutation.is_neutral(genome)

    def test_neutrality_pure_noncoding_genome(self) -> None:
        """Test neutrality in genome with only non-coding segments."""
        # All deletions should be neutral in pure non-coding genome
        test_ranges = [(0, 10), (25, 40), (50, 70), (80, 99)]

        for start_pos, end_pos in test_ranges:
            mutation = Deletion(start_pos=start_pos, end_pos=end_pos)
            assert mutation.is_neutral(self.noncoding_genome)

    def test_neutrality_pure_coding_genome(self) -> None:
        """Test neutrality in genome with only coding segments."""
        # No deletions should be neutral in pure coding genome
        test_ranges = [(0, 10), (25, 40), (50, 70), (80, 99)]

        for start_pos, end_pos in test_ranges:
            mutation = Deletion(start_pos=start_pos, end_pos=end_pos)
            assert not mutation.is_neutral(self.coding_genome)

    def test_neutrality_invalid_positions(self) -> None:
        """Test neutrality checking with positions outside genome bounds."""
        # Start position beyond genome length
        mutation = Deletion(start_pos=100, end_pos=105)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # End position beyond genome length
        mutation = Deletion(start_pos=98, end_pos=105)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # Negative start position
        mutation = Deletion(start_pos=-1, end_pos=5)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

    def test_neutrality_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = Deletion(start_pos=0, end_pos=0)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)

    def test_neutrality_single_base_deletion(self) -> None:
        """Test neutrality with single-base deletion."""
        # Single-base deletion (start == end) in non-coding
        mutation = Deletion(start_pos=10, end_pos=10)
        assert mutation.is_neutral(self.mixed_genome)

        # Single-base deletion in coding segment
        mutation = Deletion(start_pos=40, end_pos=40)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_reverse_order_positions(self) -> None:
        """Test neutrality checking with reversed position order."""
        # Should raise error for invalid order in non-circular genome
        mutation = Deletion(start_pos=90, end_pos=10)

        with pytest.raises(ValueError):
            mutation.is_neutral(self.mixed_genome)

        # Should work for circular genome - spans non-coding segments
        assert mutation.is_neutral(self.circular_mixed_genome)

        # Should not be neutral if spanning coding in circular genome
        mutation = Deletion(start_pos=90, end_pos=40)  # spans coding
        assert not mutation.is_neutral(self.circular_mixed_genome)

        mutation = Deletion(start_pos=94, end_pos=93)
        assert not mutation.is_neutral(self.circular_mixed_genome)


class TestDeletionApplication:
    """Test Deletion application to genomes."""

    def test_apply_basic_deletion(self) -> None:
        """Test basic deletion application reduces genome length correctly."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = Deletion(
            start_pos=20, end_pos=35
        )  # delete 16 bases (inclusive range)
        mutation.apply(genome)

        # Genome should be shortened by deletion length
        assert genome.length == original_length - 16

    def test_apply_at_genome_start(self) -> None:
        """Test deletion at the beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Deletion(start_pos=0, end_pos=15)
        mutation.apply(genome)

        assert genome.length == 34  # 50 - 16 (inclusive range 0-15)

    def test_apply_at_genome_end(self) -> None:
        """Test deletion at the end of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Deletion(start_pos=35, end_pos=49)  # delete last 15 bases
        mutation.apply(genome)

        assert genome.length == 35

    def test_apply_entire_segment_deletion(self) -> None:
        """Test deletion of an entire segment."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            NonCodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        genome = Genome(segments=segments)

        # Delete the entire middle segment
        mutation = Deletion(start_pos=20, end_pos=49)
        mutation.apply(genome)

        assert genome.length == 40  # 20 + 20 remaining

    def test_apply_reverse_order_positions(self) -> None:
        """Test deletion application with reversed position order."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            CodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        circular_genome = Genome(segments=segments, circular=True)

        mutation = Deletion(start_pos=50, end_pos=19)
        mutation.apply(circular_genome)

        assert circular_genome.length == 30

    def test_apply_entire_genome_deletion(self) -> None:
        """Test deletion of the entire genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Deletion(start_pos=0, end_pos=49)
        mutation.apply(genome)

        assert genome.length == 0
        assert genome.root is None


class TestDeletionSerialization:
    """Test Deletion serialization and string representation."""

    def test_serialization_methods_exist(self) -> None:
        """Test that Deletion implements serialization methods."""
        mutation = Deletion(start_pos=10, end_pos=50)

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
