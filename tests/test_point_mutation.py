"""
Comprehensive unit tests for emergents.mutations.point_mutation module.
Tests PointMutation class with all edge cases and error conditions.
"""

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection
from emergents.mutations.point_mutation import PointMutation


class TestPointMutationInitialization:
    """Test PointMutation initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic PointMutation initialization."""
        mutation = PointMutation(position=50)

        assert mutation.position == 50
        assert mutation.rng_state is None

    def test_initialization_with_rng_state(self) -> None:
        """Test PointMutation initialization with RNG state."""
        mutation = PointMutation(position=25, rng_state=12345)

        assert mutation.position == 25
        assert mutation.rng_state == 12345

    def test_initialization_edge_cases(self) -> None:
        """Test PointMutation initialization edge cases."""
        # Zero position
        mutation = PointMutation(position=0)
        assert mutation.position == 0

        # Negative position (should be allowed at initialization)
        with pytest.raises(ValueError):
            PointMutation(position=-1)


class TestPointMutationNeutrality:
    """Test PointMutation neutrality checking."""

    def setup_method(self) -> None:
        """Set up test genomes."""
        # Genome with mixed segments
        self.mixed_genome = Genome(
            [
                NonCodingSegment(length=20),  # positions 0-19
                CodingSegment(length=30),  # positions 20-49
                NonCodingSegment(length=50),  # positions 50-99
            ]
        )

        # Pure non-coding genome
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_is_neutral_noncoding_segments(self) -> None:
        """Test neutrality in non-coding segments."""
        # First non-coding segment
        mutation = PointMutation(position=10)
        assert mutation.is_neutral(self.mixed_genome)

        # Boundaries of non-coding segments
        mutation = PointMutation(position=0)
        assert mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=19)
        assert mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=50)
        assert mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=99)
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_coding_segments(self) -> None:
        """Test neutrality in coding segments."""
        # Coding segment boundaries
        mutation = PointMutation(position=20)
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = PointMutation(position=49)
        assert not mutation.is_neutral(self.mixed_genome)

        # Middle of coding segment
        mutation = PointMutation(position=35)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_pure_noncoding_genome(self) -> None:
        """Test neutrality in pure non-coding genome."""
        positions = [0, 25, 50, 75, 99]

        for pos in positions:
            mutation = PointMutation(position=pos)
            assert mutation.is_neutral(self.noncoding_genome)

    def test_is_neutral_pure_coding_genome(self) -> None:
        """Test neutrality in pure coding genome."""
        positions = [0, 25, 50, 75, 99]

        for pos in positions:
            mutation = PointMutation(position=pos)
            assert not mutation.is_neutral(self.coding_genome)

    def test_is_neutral_invalid_positions(self) -> None:
        """Test neutrality checking with invalid positions."""
        # Position beyond genome length
        mutation = PointMutation(position=100)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = PointMutation(position=0)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)


class TestPointMutationApplication:
    """Test PointMutation application."""

    def test_apply_does_nothing(self) -> None:
        """Test that apply method does nothing (current implementation)."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length
        original_segments = genome.to_list()

        mutation = PointMutation(position=50)
        mutation.apply(genome)

        # Genome should be unchanged
        assert genome.length == original_length
        assert genome.to_list() == original_segments
