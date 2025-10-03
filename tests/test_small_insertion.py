"""
Comprehensive unit tests for emergents.mutations.small_insertion module.
Tests the SmallInsertion class with all edge cases and error conditions.
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
from emergents.mutations.small_insertion import SmallInsertion


class TestSmallInsertionInitialization:
    """Test SmallInsertion initialization and parameter validation."""

    def test_initialization_basic(self) -> None:
        """Test basic SmallInsertion initialization with valid parameters."""
        mutation = SmallInsertion(position=50, length=10)

        assert mutation.position == 50
        assert mutation.length == 10
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self) -> None:
        """Test SmallInsertion initialization with RNG state parameter."""
        mutation = SmallInsertion(position=25, length=5, rng_state=12345)

        assert mutation.position == 25
        assert mutation.length == 5
        assert mutation.rng_state == 12345

    def test_initialization_edge_cases(self) -> None:
        """Test SmallInsertion initialization with edge case parameters."""
        # Zero position should be valid (gap coordinate)
        mutation = SmallInsertion(position=0, length=1)
        assert mutation.position == 0
        assert mutation.length == 1

    def test_initialization_invalid_parameters(self) -> None:
        """Test SmallInsertion initialization with invalid parameters."""
        # Zero length should raise ValueError
        with pytest.raises(ValueError):
            SmallInsertion(position=10, length=0)

        # Negative position should raise ValueError
        with pytest.raises(ValueError):
            SmallInsertion(position=-1, length=2)


class TestSmallInsertionNeutrality:
    """Test neutrality checking for SmallInsertion across different genome structures."""

    def setup_method(self) -> None:
        """Set up test genomes with various segment configurations."""
        # Mixed genome with alternating segment types
        self.mixed_genome = Genome(
            [
                NonCodingSegment(length=20),  # positions 0-19 (gaps 0-20)
                CodingSegment(length=30),  # positions 20-49 (gaps 20-50)
                NonCodingSegment(length=50),  # positions 50-99 (gaps 50-100)
            ]
        )

        # Pure non-coding genome for baseline testing
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome for contrast testing
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_neutrality_at_genome_end(self) -> None:
        """Test that insertions at genome end are always neutral."""
        # Insertion at end of mixed genome should be neutral
        mutation = SmallInsertion(position=100, length=10)  # End of mixed_genome
        assert mutation.is_neutral(self.mixed_genome)

        # Insertion at end of non-coding genome should be neutral
        mutation = SmallInsertion(position=100, length=1)  # End of noncoding_genome
        assert mutation.is_neutral(self.noncoding_genome)

        # Insertion at end of coding genome should be neutral
        mutation = SmallInsertion(position=100, length=50)  # End of coding_genome
        assert mutation.is_neutral(self.coding_genome)

    def test_neutrality_in_noncoding_segments(self) -> None:
        """Test that insertions in non-coding segments are neutral."""
        # Various gap positions in first non-coding segment
        test_positions = [0, 5, 10, 15, 20]  # Gap positions in first segment

        for position in test_positions:
            mutation = SmallInsertion(position=position, length=5)
            assert mutation.is_neutral(self.mixed_genome)

        # Various gap positions in last non-coding segment
        test_positions = [50, 60, 75, 90, 100]  # Gap positions in last segment

        for position in test_positions:
            mutation = SmallInsertion(position=position, length=3)
            assert mutation.is_neutral(self.mixed_genome)

    def test_neutrality_in_coding_segments_interior(self) -> None:
        """Test that insertions inside coding segments are not neutral."""
        # Interior gap positions in coding segment should not be neutral
        test_positions = [21, 25, 30, 35, 40, 45, 49]  # Interior gap positions

        for position in test_positions:
            mutation = SmallInsertion(position=position, length=5)
            assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_pure_coding_genome(self) -> None:
        """Test neutrality in genome with only coding segments."""
        # Start position should be neutral (before any coding starts)
        mutation = SmallInsertion(position=0, length=10)
        assert mutation.is_neutral(self.coding_genome)

        # End position should be neutral (after all coding ends)
        mutation = SmallInsertion(position=100, length=10)
        assert mutation.is_neutral(self.coding_genome)

        # Interior positions should not be neutral
        test_positions = [1, 25, 50, 75, 99]
        for position in test_positions:
            mutation = SmallInsertion(position=position, length=5)
            assert not mutation.is_neutral(self.coding_genome)

    def test_neutrality_invalid_positions(self) -> None:
        """Test neutrality checking with positions outside genome bounds."""
        # Position beyond genome length + 1 (invalid for GAP coordinates)
        mutation = SmallInsertion(position=101, length=5)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

    def test_neutrality_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()

        # Position 0 should be valid for empty genome (only valid GAP position)
        mutation = SmallInsertion(position=0, length=5)
        assert mutation.is_neutral(empty_genome)

        # Position > 0 should be invalid for empty genome
        mutation = SmallInsertion(position=1, length=5)
        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)


class TestSmallInsertionApplication:
    """Test SmallInsertion application to genomes."""

    def test_apply_basic_insertion(self) -> None:
        """Test basic insertion application increases genome length correctly."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        mutation = SmallInsertion(position=25, length=10)
        mutation.apply(genome)

        # Genome should be lengthened by insertion length
        assert genome.length == original_length + 10

        # Should have inserted NonCodingSegment material
        seg, offset, start, end = genome.find_segment_at_position(
            25, CoordinateSystem.BASE
        )
        assert isinstance(seg, NonCodingSegment)
        assert seg.length == 60
        assert offset == 25
        assert start == 0
        assert end == 60

    def test_apply_at_genome_start(self) -> None:
        """Test insertion at the beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallInsertion(position=0, length=15)
        mutation.apply(genome)

        assert genome.length == 65

        # Segment should be expanded by insertion
        seg, offset, start, end = genome.find_segment_at_position(
            0, CoordinateSystem.BASE
        )
        assert seg.length == 65
        assert offset == 0
        assert start == 0
        assert end == 65

    def test_apply_at_genome_end(self) -> None:
        """Test insertion at the end of genome."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = SmallInsertion(position=50, length=20)
        mutation.apply(genome)

        assert genome.length == 70

        # Segment should be expanded by insertion
        seg, offset, start, end = genome.find_segment_at_position(
            50, CoordinateSystem.BASE
        )
        assert seg.length == 70
        assert offset == 50
        assert start == 0
        assert end == 70

    def test_apply_creates_noncoding_material(self) -> None:
        """Test that insertions always create non-coding material."""
        genome = Genome([CodingSegment(length=50)])

        mutation = SmallInsertion(position=50, length=15)
        mutation.apply(genome)

        # Should have inserted a NonCodingSegment
        inserted_seg, _, _, _ = genome.find_segment_at_position(
            50, CoordinateSystem.BASE
        )
        assert isinstance(inserted_seg, NonCodingSegment)
        assert inserted_seg.length == 15


class TestSmallInsertionSerialization:
    """Test SmallInsertion serialization and string representation."""

    def test_serialization_methods_exist(self) -> None:
        """Test that SmallInsertion implements serialization methods."""
        mutation = SmallInsertion(position=25, length=10)

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


class TestSmallInsertionEdgeCases:
    """Test edge cases and boundary conditions for SmallInsertion."""

    def test_neutrality_with_zero_offset_logic(self) -> None:
        """Test the neutrality logic for insertions at segment boundaries."""
        # Create genome where we can test the offset logic
        segments: list[Segment] = [
            CodingSegment(length=20),  # positions 0-19
            CodingSegment(length=20),  # positions 20-39
        ]
        genome = Genome(segments=segments)

        # Position 0 is start of first coding segment (offset 0) - should be neutral
        mutation = SmallInsertion(position=0, length=5)
        assert mutation.is_neutral(genome)

        # Position 20 is start of second coding segment (offset 0) - should be neutral
        mutation = SmallInsertion(position=20, length=5)
        assert mutation.is_neutral(genome)

        # Position 10 is interior of first coding segment - should not be neutral
        mutation = SmallInsertion(position=10, length=5)
        assert not mutation.is_neutral(genome)
