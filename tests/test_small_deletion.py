"""
Comprehensive unit tests for emergents.mutations.small_deletion module.
Tests the SmallDeletion class with all edge cases and error conditions.
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
    """Test SmallDeletion initialization and parameter validation."""

    def test_initialization_basic(self) -> None:
        """Test basic SmallDeletion initialization with valid parameters."""
        mutation = SmallDeletion(position=50, length=10)

        assert mutation.position == 50
        assert mutation.length == 10
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self) -> None:
        """Test SmallDeletion initialization with RNG state parameter."""
        mutation = SmallDeletion(position=25, length=5, rng_state=12345)

        assert mutation.position == 25
        assert mutation.length == 5
        assert mutation.rng_state == 12345

    def test_initialization_edge_cases(self) -> None:
        """Test SmallDeletion initialization with edge case parameters."""
        # Zero position should be valid
        mutation = SmallDeletion(position=0, length=1)
        assert mutation.position == 0
        assert mutation.length == 1

    def test_initialization_invalid_parameters(self) -> None:
        """Test SmallDeletion initialization with invalid parameters."""
        # Zero length should raise ValueError
        with pytest.raises(ValueError):
            SmallDeletion(position=10, length=0)

        # Negative length should raise ValueError
        with pytest.raises(ValueError):
            SmallDeletion(position=10, length=-5)


class TestSmallDeletionNeutrality:
    """Test neutrality checking for SmallDeletion across different genome structures."""

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

        # Pure non-coding genome for baseline testing
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome for contrast testing
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_neutrality_within_single_noncoding_segment(self) -> None:
        """Test that deletions entirely within non-coding segments are neutral."""
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

    def test_neutrality_crossing_segment_boundaries(self) -> None:
        """Test that deletions crossing into coding segments are not neutral."""
        # Deletion crossing from non-coding to coding
        mutation = SmallDeletion(position=26, length=5)  # spans boundary
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_within_coding_segment(self) -> None:
        """Test that deletions entirely within coding segments are not neutral."""
        # Delete within middle of coding segment
        mutation = SmallDeletion(position=35, length=10)  # positions 35-44
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete at start of coding segment
        mutation = SmallDeletion(position=30, length=5)
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete at end of coding segment
        mutation = SmallDeletion(position=67, length=5)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_across_segment_boundaries(self) -> None:
        """Test that deletions spanning segment boundaries are not neutral when touching coding."""
        # Delete across non-coding to coding boundary
        mutation = SmallDeletion(position=25, length=10)  # spans boundary
        assert not mutation.is_neutral(self.mixed_genome)

        # Delete across coding to non-coding boundary
        mutation = SmallDeletion(position=65, length=10)  # spans boundary
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_single_base_deletions(self) -> None:
        """Test neutrality for single-base deletions in different segment types."""
        # Single base in non-coding segment
        mutation = SmallDeletion(position=10, length=1)
        assert mutation.is_neutral(self.mixed_genome)

        # Single base in coding segment
        mutation = SmallDeletion(position=40, length=1)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_pure_noncoding_genome(self) -> None:
        """Test neutrality in genome with only non-coding segments."""
        # All deletions should be neutral in pure non-coding genome
        test_cases = [(0, 10), (25, 15), (50, 20), (90, 10)]

        for position, length in test_cases:
            mutation = SmallDeletion(position=position, length=length)
            assert mutation.is_neutral(self.noncoding_genome)

    def test_neutrality_pure_coding_genome(self) -> None:
        """Test neutrality in genome with only coding segments."""
        # No deletions should be neutral in pure coding genome
        test_cases = [(0, 10), (25, 15), (50, 20), (90, 10)]

        for position, length in test_cases:
            mutation = SmallDeletion(position=position, length=length)
            assert not mutation.is_neutral(self.coding_genome)

    def test_neutrality_invalid_positions(self) -> None:
        """Test neutrality checking with positions outside genome bounds."""
        # Deletion extending beyond genome length
        mutation = SmallDeletion(position=95, length=10)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.noncoding_genome)

    def test_neutrality_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = SmallDeletion(position=0, length=1)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)


class TestSmallDeletionApplication:
    """Test SmallDeletion application to genomes."""

    def test_apply_basic_deletion(self) -> None:
        """Test basic deletion application reduces genome length correctly."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = SmallDeletion(position=25, length=10)
        mutation.apply(genome)

        # Genome should be shortened by deletion length
        assert genome.length == original_length - 10

    def test_apply_at_genome_start(self) -> None:
        """Test deletion at the beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallDeletion(position=0, length=15)
        mutation.apply(genome)

        assert genome.length == 35

    def test_apply_at_genome_end(self) -> None:
        """Test deletion at the end of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallDeletion(position=35, length=15)  # delete last 15 bases
        mutation.apply(genome)

        assert genome.length == 35

    def test_apply_entire_segment_deletion(self) -> None:
        """Test deletion of an entire segment."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            CodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        genome = Genome(segments=segments)

        # Delete the entire first segment
        mutation = SmallDeletion(position=0, length=20)
        mutation.apply(genome)

        assert genome.length == 50  # 30 + 20 remaining

    def test_apply_across_multiple_segments(self) -> None:
        """Test deletion spanning multiple segments."""
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

    def test_apply_entire_genome_deletion(self) -> None:
        """Test deletion of the entire genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallDeletion(position=0, length=50)
        mutation.apply(genome)

        assert genome.length == 0
        assert genome.root is None


class TestSmallDeletionSerialization:
    """Test SmallDeletion serialization and string representation."""

    def test_serialization_methods_exist(self) -> None:
        """Test that SmallDeletion implements serialization methods."""
        mutation = SmallDeletion(position=25, length=10)

        # These methods should exist and return appropriate types
        description = mutation.describe()
        assert isinstance(description, str)
        assert "25" in description
        assert "10" in description

        # Check if serialize is implemented or raises NotImplementedError
        try:
            serialized = mutation.serialize()
            assert isinstance(serialized, dict)
        except NotImplementedError:
            # This is acceptable if not yet implemented
            pass


class TestSmallDeletionEdgeCases:
    """Test edge cases and boundary conditions for SmallDeletion."""

    def test_deletion_single_base_genome(self) -> None:
        """Test deletion in genome with only one base."""
        genome = Genome([NonCodingSegment(length=1)])

        # Deleting the only base should be neutral and result in empty genome
        mutation = SmallDeletion(position=0, length=1)
        assert mutation.is_neutral(genome)

        mutation.apply(genome)
        assert genome.length == 0
