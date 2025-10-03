"""
Comprehensive unit tests for emergents.mutations.point_mutation module.
Tests the PointMutation class with all edge cases and error conditions.
"""

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection
from emergents.mutations.point_mutation import PointMutation


class TestPointMutationInitialization:
    """Test PointMutation initialization and parameter validation."""

    def test_initialization_basic(self) -> None:
        """Test basic PointMutation initialization with valid parameters."""
        mutation = PointMutation(position=50)

        assert mutation.position == 50
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self) -> None:
        """Test PointMutation initialization with RNG state parameter."""
        mutation = PointMutation(position=25, rng_state=12345)

        assert mutation.position == 25
        assert mutation.rng_state == 12345

    def test_initialization_edge_cases(self) -> None:
        """Test PointMutation initialization with edge case positions."""
        # Zero position should be valid
        mutation = PointMutation(position=0)
        assert mutation.position == 0

    def test_initialization_invalid_parameters(self) -> None:
        """Test PointMutation initialization with invalid parameters."""
        # Negative position should raise ValueError
        with pytest.raises(ValueError):
            PointMutation(position=-1)


class TestPointMutationNeutrality:
    """Test neutrality checking for PointMutation across different genome structures."""

    def setup_method(self) -> None:
        """Set up test genomes with various segment configurations."""
        # Mixed genome with alternating segment types
        self.mixed_genome = Genome(
            [
                NonCodingSegment(length=20),  # positions 0-19
                CodingSegment(length=30),  # positions 20-49
                NonCodingSegment(length=50),  # positions 50-99
            ]
        )

        # Pure non-coding genome for baseline testing
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome for contrast testing
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_neutrality_in_noncoding_segments(self) -> None:
        """Test that mutations in non-coding segments are neutral."""
        # Test various positions within first non-coding segment
        mutation = PointMutation(position=10)
        assert mutation.is_neutral(self.mixed_genome)

        # Test boundaries of non-coding segments
        mutation = PointMutation(position=0)  # Start of genome
        assert mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=19)  # End of first non-coding
        assert mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=50)  # Start of last non-coding
        assert mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=99)  # End of genome
        assert mutation.is_neutral(self.mixed_genome)

    def test_neutrality_in_coding_segments(self) -> None:
        """Test that mutations in coding segments are not neutral."""
        # Test boundaries of coding segment
        mutation = PointMutation(position=20)
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=49)
        assert not mutation.is_neutral(self.mixed_genome)

        # Test middle of coding segment
        mutation = PointMutation(position=35)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_neutrality_pure_noncoding_genome(self) -> None:
        """Test neutrality in genome with only non-coding segments."""
        test_positions = [0, 25, 50, 75, 99]

        for position in test_positions:
            mutation = PointMutation(position=position)
            assert mutation.is_neutral(self.noncoding_genome)

    def test_neutrality_pure_coding_genome(self) -> None:
        """Test neutrality in genome with only coding segments."""
        test_positions = [0, 25, 50, 75, 99]

        for position in test_positions:
            mutation = PointMutation(position=position)
            assert not mutation.is_neutral(self.coding_genome)

    def test_neutrality_invalid_positions(self) -> None:
        """Test neutrality checking with positions outside genome bounds."""
        # Position beyond genome length should raise IndexError
        mutation = PointMutation(position=100)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

    def test_neutrality_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = PointMutation(position=0)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)


class TestPointMutationApplication:
    """Test PointMutation application to genomes."""

    def test_apply_implementation(self) -> None:
        """Test that apply method preserves genome structure (placeholder implementation)."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length
        original_segments = genome.to_list()

        mutation = PointMutation(position=50)
        mutation.apply(genome)

        # Genome should remain unchanged with current implementation
        assert genome.length == original_length
        assert genome.to_list() == original_segments


class TestPointMutationSerialization:
    """Test PointMutation serialization and string representation."""

    def test_serialize_not_implemented(self) -> None:
        """Test that serialize method raises NotImplementedError."""
        mutation = PointMutation(position=25)

        with pytest.raises(NotImplementedError):
            mutation.serialize()

    def test_describe_method(self) -> None:
        """Test that describe method returns proper string representation."""
        mutation = PointMutation(position=25)
        description = mutation.describe()

        assert isinstance(description, str)
        assert "25" in description
        assert "Point mutation" in description
