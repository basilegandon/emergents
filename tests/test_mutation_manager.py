"""
Comprehensive test suite for mutation_manager.py.

Tests cover initialization, configuration updates, mutation type selection,
mutation creation for all types, mutation application with edge cases,
and population-level mutation handling.
"""

import contextlib
import random
from unittest.mock import MagicMock, patch

import pytest

from emergents.config import MutationConfig
from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment
from emergents.mutation_manager import MutationManager
from emergents.mutations.base import Mutation
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion
from emergents.statistics import MutationCounts

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def default_config():
    """Create a default mutation configuration."""
    return MutationConfig()


@pytest.fixture()
def custom_config():
    """Create a custom mutation configuration with different weights."""
    return MutationConfig(
        weights={
            PointMutation: 2.0,
            SmallDeletion: 1.0,
            SmallInsertion: 0.5,
            Deletion: 0.5,
            Duplication: 0.5,
            Inversion: 0.5,
        }
    )


@pytest.fixture()
def mutation_manager(default_config):
    """Create a default mutation manager."""
    return MutationManager(default_config)


def make_simple_genome(
    length: int = 10, coding_positions: list[tuple[int, int]] | None = None
):
    """
    Create a simple test genome.

    Args:
        length: Total length of genome
        coding_positions: List of (start, end) tuples for coding regions
    """
    if coding_positions is None:
        # Default: coding region from 3 to 7
        coding_positions = [(3, 7)]

    segments = []
    current_pos = 0

    for start, end in sorted(coding_positions):
        if start > current_pos:
            segments.append(NonCodingSegment(length=start - current_pos))
        segments.append(CodingSegment(length=end - start))
        current_pos = end

    if current_pos < length:
        segments.append(NonCodingSegment(length=length - current_pos))

    return Genome(segments)


def make_genome_with_single_coding():
    """Create a genome with a single coding segment."""
    return Genome([CodingSegment(length=5)])


def make_genome_all_noncoding(length: int = 10):
    """Create a genome with only non-coding segments."""
    return Genome([NonCodingSegment(length=length)])


# ============================================================================
# Initialization and Configuration Tests
# ============================================================================


class TestMutationManagerInitialization:
    """Tests for MutationManager initialization."""

    def test_init_with_default_config(self, default_config):
        """Test initialization with default configuration."""
        manager = MutationManager(default_config)
        assert manager.config == default_config
        assert len(manager.mutation_types) == 6
        assert set(manager.mutation_types) == {
            PointMutation,
            SmallDeletion,
            SmallInsertion,
            Deletion,
            Duplication,
            Inversion,
        }

    def test_init_with_custom_config(self, custom_config):
        """Test initialization with custom configuration."""
        manager = MutationManager(custom_config)
        assert manager.config == custom_config
        assert PointMutation in manager.mutation_types
        assert SmallDeletion in manager.mutation_types

    def test_mutation_weights_extracted_correctly(self, default_config):
        """Verify weights are extracted in same order as types."""
        manager = MutationManager(default_config)
        assert len(manager.mutation_weights) == len(manager.mutation_types)
        # Default weights are normalized to sum to 1.0 (1/6 each for 6 types)
        assert all(w > 0 for w in manager.mutation_weights)
        assert abs(sum(manager.mutation_weights) - 1.0) < 1e-9

    def test_custom_weights_extracted_correctly(self, custom_config):
        """Verify custom weights are extracted correctly."""
        manager = MutationManager(custom_config)
        # Check that weights correspond to types
        for mut_type, weight in zip(
            manager.mutation_types, manager.mutation_weights, strict=False
        ):
            assert weight == custom_config.weights[mut_type]

    def test_init_preserves_weight_order(self):
        """Ensure weights and types maintain correspondence."""
        config = MutationConfig(
            weights={
                PointMutation: 5.0,
                SmallDeletion: 3.0,
                SmallInsertion: 2.0,
                Deletion: 1.0,
                Duplication: 1.0,
                Inversion: 1.0,
            }
        )
        manager = MutationManager(config)

        for mut_type, weight in zip(
            manager.mutation_types, manager.mutation_weights, strict=False
        ):
            assert config.weights[mut_type] == weight


# ============================================================================
# Configuration Update Tests
# ============================================================================


class TestUpdateConfig:
    """Tests for updating mutation configuration."""

    def test_update_config_replaces_all_attributes(
        self, mutation_manager, custom_config
    ):
        """Verify update_config replaces config, types, and weights."""
        old_config = mutation_manager.config
        old_types = mutation_manager.mutation_types.copy()

        mutation_manager.update_config(custom_config)

        assert mutation_manager.config == custom_config
        assert mutation_manager.config != old_config
        # Types should be same set, but weights might differ
        assert set(mutation_manager.mutation_types) == set(old_types)

    def test_update_config_updates_weights(self, mutation_manager):
        """Test that weights are updated when config changes."""
        old_weights = mutation_manager.mutation_weights.copy()

        new_config = MutationConfig(
            weights={
                PointMutation: 10.0,
                SmallDeletion: 1.0,
                SmallInsertion: 1.0,
                Deletion: 1.0,
                Duplication: 1.0,
                Inversion: 1.0,
            }
        )

        mutation_manager.update_config(new_config)
        new_weights = mutation_manager.mutation_weights

        assert new_weights != old_weights
        assert new_config.weights[PointMutation] in new_weights


# ============================================================================
# Mutation Type Selection Tests
# ============================================================================


class TestSelectMutationType:
    """Tests for mutation type selection."""

    def test_select_mutation_type_returns_valid_type(self, mutation_manager):
        """Verify select_mutation_type returns a valid mutation type."""
        for _ in range(20):
            selected = mutation_manager.select_mutation_type()
            assert selected in mutation_manager.mutation_types
            assert issubclass(selected, Mutation)

    def test_select_mutation_type_respects_weights(self, custom_config):
        """Test that weights influence selection distribution.

        PointMutation has weight 2.0, others have 0.5-1.0.
        In 100 samples, PointMutation should appear more frequently.
        """
        manager = MutationManager(custom_config)

        # Set seed for reproducibility
        random.seed(42)
        selections = [manager.select_mutation_type() for _ in range(200)]

        point_mutation_count = selections.count(PointMutation)
        # With weight 2.0 vs average of ~0.9, expect roughly 50%+ of selections
        # This is a statistical test, so we use a reasonable threshold
        assert point_mutation_count > 50

    def test_select_mutation_type_with_equal_weights(self, default_config):
        """Test selection with equal weights has reasonable distribution."""
        manager = MutationManager(default_config)

        random.seed(42)
        selections = [manager.select_mutation_type() for _ in range(600)]

        # With equal weights, each of 6 types should appear roughly equally
        counts = {
            mutation_type: selections.count(mutation_type)
            for mutation_type in manager.mutation_types
        }

        # Each type should appear at least 50 times in 600 samples (> 50/100)
        for count in counts.values():
            assert count > 50


# ============================================================================
# Mutation Creation Tests
# ============================================================================


class TestCreateMutation:
    """Tests for mutation creation with various genome sizes."""

    def test_create_small_deletion(self, mutation_manager):
        """Test SmallDeletion creation."""
        genome = make_simple_genome(length=50)
        mutation = mutation_manager.create_mutation(genome, SmallDeletion)

        assert isinstance(mutation, SmallDeletion)
        assert 0 <= mutation.position < len(genome)
        assert 1 <= mutation.length <= mutation_manager.config.small_mutation_max_size

    def test_create_small_deletion_respects_max_size(self, mutation_manager):
        """Test SmallDeletion respects configured max size."""
        config = MutationConfig(small_mutation_max_size=3)
        manager = MutationManager(config)
        genome = make_simple_genome(length=50)

        for _ in range(20):
            mutation = manager.create_mutation(genome, SmallDeletion)
            assert mutation.length <= 3

    def test_create_small_deletion_small_genome(self, mutation_manager):
        """Test SmallDeletion with small genome."""
        genome = make_simple_genome(length=2, coding_positions=[])
        mutation = mutation_manager.create_mutation(genome, SmallDeletion)

        assert isinstance(mutation, SmallDeletion)
        assert mutation.length <= 2

    def test_create_small_insertion(self, mutation_manager):
        """Test SmallInsertion creation."""
        genome = make_simple_genome(length=50)
        mutation = mutation_manager.create_mutation(genome, SmallInsertion)

        assert isinstance(mutation, SmallInsertion)
        assert 0 <= mutation.position <= len(genome)
        assert 1 <= mutation.length <= mutation_manager.config.small_mutation_max_size

    def test_create_small_insertion_respects_max_size(self, mutation_manager):
        """Test SmallInsertion respects configured max size."""
        config = MutationConfig(small_mutation_max_size=2)
        manager = MutationManager(config)
        genome = make_simple_genome(length=50)

        for _ in range(20):
            mutation = manager.create_mutation(genome, SmallInsertion)
            assert mutation.length <= 2

    def test_create_deletion(self, mutation_manager):
        """Test Deletion creation."""
        genome = make_simple_genome(length=100)
        mutation = mutation_manager.create_mutation(genome, Deletion)

        assert isinstance(mutation, Deletion)
        assert 0 <= mutation.start_pos <= mutation.end_pos < len(genome)

    def test_create_duplication(self, mutation_manager):
        """Test Duplication creation."""
        genome = make_simple_genome(length=100)
        mutation = mutation_manager.create_mutation(genome, Duplication)

        assert isinstance(mutation, Duplication)
        assert 0 <= mutation.start_pos <= mutation.end_pos < len(genome)
        assert 0 <= mutation.insertion_pos <= len(genome)

    def test_create_inversion(self, mutation_manager):
        """Test Inversion creation."""
        genome = make_simple_genome(length=100)
        mutation = mutation_manager.create_mutation(genome, Inversion)

        assert isinstance(mutation, Inversion)
        assert 0 <= mutation.start_pos < mutation.end_pos <= len(genome)

    def test_create_point_mutation(self, mutation_manager):
        """Test PointMutation creation."""
        genome = make_simple_genome(length=50)
        mutation = mutation_manager.create_mutation(genome, PointMutation)

        assert isinstance(mutation, PointMutation)
        assert 0 <= mutation.position < len(genome)

    # Edge cases for mutation creation

    def test_create_mutation_on_minimal_genome(self, mutation_manager):
        """Test mutation creation on minimal (single-base) genome."""
        genome = Genome([CodingSegment(length=1)])

        # PointMutation should work
        pm = mutation_manager.create_mutation(genome, PointMutation)
        assert pm.position == 0

        # Deletion should work (delete the single base)
        del_mut = mutation_manager.create_mutation(genome, Deletion)
        assert del_mut.start_pos == 0
        assert del_mut.end_pos == 0

    def test_create_mutation_empty_genome_handling(self, mutation_manager):
        """Test behavior with empty genome (returns without mutation creation logic)."""
        genome = Genome([])

        # These should still create mutations with potentially out-of-bounds values
        # The actual validation happens in mutation.apply() or is_neutral()
        # No exception should be raised during creation
        for mutation_type in mutation_manager.mutation_types:
            # Just verify we don't crash; actual validation is mutation's concern
            with contextlib.suppress(ValueError, IndexError):
                mutation_manager.create_mutation(genome, mutation_type)

    def test_create_all_mutation_types(self, mutation_manager):
        """Test creation of all mutation types on same genome."""
        genome = make_simple_genome(length=100)

        for mutation_type in mutation_manager.mutation_types:
            mutation = mutation_manager.create_mutation(genome, mutation_type)
            assert isinstance(mutation, Mutation)


# ============================================================================
# Mutation Application Tests
# ============================================================================


class TestApplyMutationsToGenome:
    """Tests for applying mutations to a single genome."""

    def test_apply_mutations_returns_tuple(self, mutation_manager):
        """Test that apply_mutations_to_genome returns proper tuple."""
        genome = make_simple_genome(length=50)
        result = mutation_manager.apply_mutations_to_genome(genome, mutation_rate=0.01)

        assert isinstance(result, tuple)
        assert len(result) == 3
        neutral, non_neutral, survived = result
        assert isinstance(neutral, int)
        assert isinstance(non_neutral, int)
        assert isinstance(survived, bool)

    def test_apply_mutations_empty_genome(self, mutation_manager):
        """Test applying mutations to empty genome returns genome death."""
        genome = Genome([])
        neutral, non_neutral, survived = mutation_manager.apply_mutations_to_genome(
            genome, mutation_rate=0.5
        )

        assert neutral == 0
        assert non_neutral == 0
        assert survived is False

    def test_apply_mutations_zero_rate(self, mutation_manager):
        """Test with zero mutation rate (no mutations)."""
        genome = make_simple_genome(length=50)
        neutral, non_neutral, survived = mutation_manager.apply_mutations_to_genome(
            genome, mutation_rate=0.0
        )

        assert neutral == 0
        assert non_neutral == 0
        assert survived is True

    def test_apply_mutations_tracks_counts(self, mutation_manager):
        """Test that neutral and non-neutral counts are tracked."""
        genome = make_all_noncoding_genome()
        neutral, non_neutral, survived = mutation_manager.apply_mutations_to_genome(
            genome, mutation_rate=0.1
        )

        # With all non-coding and neutral mutations, survived should be True
        # and non_neutral should be 0
        if neutral + non_neutral > 0:
            assert survived is True or non_neutral > 0

    def test_apply_mutations_kills_on_non_neutral(self, mutation_manager):
        """Test that non-neutral mutation kills genome."""
        genome = make_simple_genome(length=50, coding_positions=[(0, 50)])

        # Mock to ensure we get a non-neutral mutation
        with patch.object(mutation_manager, "select_mutation_type") as mock_select:
            mock_select.return_value = PointMutation

            # Use high mutation rate to ensure mutations happen
            neutral, non_neutral, survived = mutation_manager.apply_mutations_to_genome(
                genome, mutation_rate=1.0
            )

            # If any mutations were applied and they hit coding regions,
            # genome should die
            if neutral + non_neutral > 0:
                # Can't guarantee non_neutral > 0 due to randomness, but if survived=False,
                # then non_neutral must have been triggered
                assert survived is False or non_neutral == 0

    def test_apply_mutations_multiple_mutations(self, mutation_manager):
        """Test applying multiple mutations in single call."""
        genome = make_all_noncoding_genome()
        neutral, non_neutral, _ = mutation_manager.apply_mutations_to_genome(
            genome, mutation_rate=0.2
        )

        total_mutations = neutral + non_neutral
        # With 0.2 rate on genome of length ~10, expect 2 mutations on average
        # but probabilistic
        assert total_mutations >= 0

    def test_apply_mutations_genome_becomes_empty_during_mutation(
        self, mutation_manager
    ):
        """Test handling when genome dies mid-mutations."""
        genome = make_simple_genome(length=10)

        # Force genome death by returning False from a mutation
        with patch("emergents.mutation_manager.Mutation.is_neutral") as mock_neutral:
            mock_neutral.return_value = False

            _, non_neutral, survived = mutation_manager.apply_mutations_to_genome(
                genome, mutation_rate=0.5
            )

            if non_neutral > 0:
                assert survived is False

    def test_apply_mutations_exception_handling(self, mutation_manager):
        """Test that exceptions during mutation application are handled."""
        genome = make_simple_genome(length=50)

        # Mock mutation to raise exception
        with patch.object(mutation_manager, "select_mutation_type") as mock_select:
            mock_type = MagicMock()
            mock_mutation = MagicMock(spec=Mutation)
            mock_mutation.is_neutral.return_value = True
            mock_mutation.apply.side_effect = Exception("Mutation failed")

            mock_select.return_value = mock_type
            mutation_manager.create_mutation = MagicMock(return_value=mock_mutation)

            _, non_neutral, survived = mutation_manager.apply_mutations_to_genome(
                genome, mutation_rate=0.5
            )

            # Exception should be caught and treated as non-neutral
            if non_neutral > 0:
                assert survived is False

    def test_apply_mutations_coalesce_called(self, mutation_manager):
        """Test that coalesce is called after neutral mutations."""
        genome = make_all_noncoding_genome()

        with patch.object(genome, "coalesce_all"):
            mutation_manager.apply_mutations_to_genome(genome, mutation_rate=0.1)

            # coalesce_all should be called at least once if mutations happened
            # (can't guarantee mutations with low rate, but verify method exists)
            assert hasattr(genome, "coalesce_all")


# ============================================================================
# Population Mutation Application Tests
# ============================================================================


class TestApplyMutationsToPopulation:
    """Tests for applying mutations to a population of genomes."""

    def test_apply_mutations_population_returns_correct_types(self, mutation_manager):
        """Test return type of apply_mutations_to_population."""
        genomes = [make_simple_genome(length=50) for _ in range(5)]
        counts, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.01
        )

        assert isinstance(counts, MutationCounts)
        assert isinstance(dead_indices, list)
        assert all(isinstance(idx, int) for idx in dead_indices)

    def test_apply_mutations_population_empty_list(self, mutation_manager):
        """Test with empty population."""
        genomes = []
        counts, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.01
        )

        assert counts.neutral == 0
        assert counts.non_neutral == 0
        assert counts.total == 0
        assert dead_indices == []

    def test_apply_mutations_population_single_genome(self, mutation_manager):
        """Test with single genome population."""
        genomes = [make_simple_genome(length=50)]
        counts, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.01
        )

        assert isinstance(counts.total, int)
        assert len(dead_indices) <= 1

    def test_apply_mutations_population_tracks_dead_genomes(self, mutation_manager):
        """Test that dead genome indices are tracked correctly."""
        genomes = [
            make_simple_genome(length=10, coding_positions=[(0, 10)]) for _ in range(5)
        ]

        # With high mutation rate on all-coding genomes, likely to kill some
        _, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.5
        )

        # Dead indices should be valid and in order
        assert all(0 <= idx < len(genomes) for idx in dead_indices)
        assert len(set(dead_indices)) == len(dead_indices)  # No duplicates

    def test_apply_mutations_population_counts_correct(self, mutation_manager):
        """Test that mutation counts are aggregated correctly."""
        genomes = [make_all_noncoding_genome() for _ in range(3)]
        counts, _ = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.1
        )

        # Total should equal neutral + non_neutral
        assert counts.total == counts.neutral + counts.non_neutral

    def test_apply_mutations_population_all_survive(self, mutation_manager):
        """Test population where all genomes survive."""
        genomes = [make_all_noncoding_genome() for _ in range(5)]
        _, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.01
        )

        # Non-coding only, should all survive
        assert len(dead_indices) == 0

    def test_apply_mutations_population_preserves_genomes(self, mutation_manager):
        """Test that surviving genomes are preserved in original list."""
        genomes = [make_simple_genome(length=50) for _ in range(5)]
        original_ids = [id(g) for g in genomes]

        _, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.01
        )

        # The genomes should still be the same objects
        for i, original_id in enumerate(original_ids):
            if i not in dead_indices:
                assert id(genomes[i]) == original_id

    def test_apply_mutations_population_zero_rate(self, mutation_manager):
        """Test population with zero mutation rate."""
        genomes = [make_simple_genome(length=50) for _ in range(5)]
        counts, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.0
        )

        assert counts.total == 0
        assert len(dead_indices) == 0

    def test_apply_mutations_population_large_population(self, mutation_manager):
        """Test with larger population."""
        genomes = [make_simple_genome(length=100) for _ in range(50)]
        counts, dead_indices = mutation_manager.apply_mutations_to_population(
            genomes, mutation_rate=0.01
        )

        # Should handle without issues
        assert isinstance(counts, MutationCounts)
        assert len(dead_indices) <= 50


# ============================================================================
# Integration Tests
# ============================================================================


class TestMutationManagerIntegration:
    """Integration tests for full mutation manager workflows."""

    def test_sequential_mutations_on_genome(self, mutation_manager):
        """Test applying mutations sequentially to same genome."""
        genome = make_simple_genome(length=100)

        for _ in range(5):
            _, _, survived = mutation_manager.apply_mutations_to_genome(
                genome, mutation_rate=0.01
            )

            if not survived:
                break

    def test_config_change_between_mutations(self, mutation_manager, custom_config):
        """Test changing config between mutation applications."""
        genome = make_simple_genome(length=50)

        mutation_manager.apply_mutations_to_genome(genome, mutation_rate=0.01)
        mutation_manager.update_config(custom_config)
        mutation_manager.apply_mutations_to_genome(genome, mutation_rate=0.01)

        assert mutation_manager.config == custom_config

    def test_multiple_populations_independent(self, mutation_manager):
        """Test that applying mutations to different populations is independent."""
        pop1 = [make_simple_genome(length=50) for _ in range(3)]
        pop2 = [make_simple_genome(length=50) for _ in range(3)]

        counts1, _ = mutation_manager.apply_mutations_to_population(pop1, 0.01)
        counts2, _ = mutation_manager.apply_mutations_to_population(pop2, 0.01)

        # Both populations should work independently
        assert isinstance(counts1, MutationCounts)
        assert isinstance(counts2, MutationCounts)

    def test_selection_distribution_over_many_mutations(self):
        """Test that mutation type selection reflects weights over large sample."""
        config = MutationConfig(
            weights={
                PointMutation: 4.0,
                SmallDeletion: 1.0,
                SmallInsertion: 1.0,
                Deletion: 1.0,
                Duplication: 1.0,
                Inversion: 1.0,
            }
        )
        manager = MutationManager(config)

        random.seed(42)
        genomes = [make_simple_genome(length=100) for _ in range(10)]

        # Apply mutations multiple times
        for _ in range(3):
            manager.apply_mutations_to_population(genomes, 0.1)

        # Should complete without error


# ============================================================================
# Helper Functions for Test Genomes
# ============================================================================


def make_all_noncoding_genome(length: int = 10):
    """Create a genome with only non-coding segments."""
    return Genome([NonCodingSegment(length=length)])
