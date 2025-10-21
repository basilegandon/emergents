"""
Comprehensive unit tests for emergents.mutation_manager module.
Tests MutationManager class and mutation application logic.
"""

import random
from unittest.mock import Mock, patch

import pytest

from emergents.config import MutationConfig
from emergents.genome.genome import Genome
from emergents.genome.segments import NonCodingSegment, Segment
from emergents.mutation_manager import MutationManager
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion
from emergents.statistics import MutationCounts


class TestMutationManager:
    """Test MutationManager class."""

    def create_test_genome(self, length: int = 1000) -> Genome:
        """Helper to create a test genome with specified length."""
        if length <= 0:
            return Genome()
        segments: list[Segment] = [NonCodingSegment(length=length)]
        return Genome(segments=segments)

    def test_initialization(self) -> None:
        """Test MutationManager initialization."""
        config = MutationConfig()
        manager = MutationManager(config)

        assert manager.config == config
        assert len(manager.mutation_types) == 6  # All mutation types
        assert len(manager.mutation_weights) == 6
        assert sum(manager.mutation_weights) == pytest.approx(1.0)

    def test_initialization_with_custom_config(self) -> None:
        """Test MutationManager with custom configuration."""
        custom_config = MutationConfig(
            weights={
                PointMutation: 0.5,
                SmallDeletion: 0.3,
                SmallInsertion: 0.2,
            },
            small_mutation_max_size=10,
        )

        manager = MutationManager(custom_config)

        assert len(manager.mutation_types) == 3
        assert PointMutation in manager.mutation_types
        assert SmallDeletion in manager.mutation_types
        assert SmallInsertion in manager.mutation_types

    def test_update_config(self) -> None:
        """Test updating mutation configuration."""
        initial_config = MutationConfig()
        manager = MutationManager(initial_config)

        new_config = MutationConfig(
            weights={PointMutation: 1.0}, small_mutation_max_size=8
        )

        manager.update_config(new_config)

        assert manager.config == new_config
        assert len(manager.mutation_types) == 1
        assert manager.mutation_types[0] == PointMutation
        assert manager.mutation_weights[0] == 1.0

    @patch("emergents.mutation_manager.random.choices")
    def test_select_mutation_type(self, mock_choices: Mock) -> None:
        """Test mutation type selection."""
        config = MutationConfig()
        manager = MutationManager(config)

        mock_choices.return_value = [PointMutation]

        result = manager.select_mutation_type()

        assert result == PointMutation
        mock_choices.assert_called_once_with(
            manager.mutation_types, weights=manager.mutation_weights
        )

    def test_create_point_mutation(self) -> None:
        """Test creating a point mutation."""
        config = MutationConfig()
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        with patch("emergents.mutation_manager.random.randint", return_value=50):
            mutation = manager.create_mutation(genome, PointMutation)

        assert isinstance(mutation, PointMutation)
        assert mutation.position == 50

    def test_create_small_deletion(self) -> None:
        """Test creating a small deletion."""
        config = MutationConfig(small_mutation_max_size=5)
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        with patch("emergents.mutation_manager.random.randint") as mock_randint:
            mock_randint.side_effect = [3, 20]  # length=3, position=20
            mutation = manager.create_mutation(genome, SmallDeletion)

        assert isinstance(mutation, SmallDeletion)
        assert mutation.length == 3
        assert mutation.position == 20

    def test_create_small_insertion(self) -> None:
        """Test creating a small insertion."""
        config = MutationConfig(small_mutation_max_size=4)
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        with patch("emergents.mutation_manager.random.randint") as mock_randint:
            mock_randint.side_effect = [30, 2]  # position=30, length=2
            mutation = manager.create_mutation(genome, SmallInsertion)

        assert isinstance(mutation, SmallInsertion)
        assert mutation.position == 30
        assert mutation.length == 2

    def test_create_deletion(self) -> None:
        """Test creating a deletion."""
        config = MutationConfig()
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        with patch("emergents.mutation_manager.random.randint") as mock_randint:
            mock_randint.side_effect = [20, 10]  # del_size=20, start_pos=10
            mutation = manager.create_mutation(genome, Deletion)

        assert isinstance(mutation, Deletion)
        assert mutation.start_pos == 10
        assert mutation.end_pos == 29  # start_pos + del_size - 1

    def test_create_duplication(self) -> None:
        """Test creating a duplication."""
        config = MutationConfig()
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        with patch("emergents.mutation_manager.random.randint") as mock_randint:
            mock_randint.side_effect = [
                15,
                20,
                50,
            ]  # dup_size=15, start_pos=20, insertion_pos=50
            mutation = manager.create_mutation(genome, Duplication)

        assert isinstance(mutation, Duplication)
        assert mutation.start_pos == 20
        assert mutation.end_pos == 34  # start_pos + dup_size - 1
        assert mutation.insertion_pos == 50

    def test_create_inversion(self) -> None:
        """Test creating an inversion."""
        config = MutationConfig()
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        with patch("emergents.mutation_manager.random.randint") as mock_randint:
            mock_randint.side_effect = [25, 30]  # inv_size=25, start_pos=30
            mutation = manager.create_mutation(genome, Inversion)

        assert isinstance(mutation, Inversion)
        assert mutation.start_pos == 30
        assert mutation.end_pos == 55  # start_pos + inv_size

    def test_create_small_deletion_respects_max_size(self) -> None:
        """Test that small deletion respects genome length limit."""
        config = MutationConfig(small_mutation_max_size=50)
        manager = MutationManager(config)
        genome = self.create_test_genome(10)  # Small genome

        # Mock random.randint to simulate the min() constraint
        with patch("emergents.mutation_manager.random.randint") as mock_randint:
            mock_randint.side_effect = [
                5,
                3,
            ]  # del_size=5 (limited to min(50, 10)=10), start_pos=3
            mutation = manager.create_mutation(genome, SmallDeletion)

        assert isinstance(mutation, SmallDeletion)
        # Should have called randint(1, 10) for deletion size, not randint(1, 50)
        mock_randint.assert_any_call(1, 10)

    @patch("emergents.mutation_manager.np.random.binomial")
    def test_apply_mutations_to_genome_no_mutations(self, mock_binomial: Mock) -> None:
        """Test applying mutations when no mutations occur."""
        config = MutationConfig()
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        mock_binomial.return_value = 0  # No mutations

        neutral, non_neutral, survived = manager.apply_mutations_to_genome(
            genome, 0.001
        )

        assert neutral == 0
        assert non_neutral == 0
        assert survived is True

    @patch("emergents.mutation_manager.np.random.binomial")
    def test_apply_mutations_to_genome_empty_genome(self, mock_binomial: Mock) -> None:
        """Test applying mutations to empty genome."""
        config = MutationConfig()
        manager = MutationManager(config)
        genome = self.create_test_genome(0)  # Empty genome

        neutral, non_neutral, survived = manager.apply_mutations_to_genome(
            genome, 0.001
        )

        assert neutral == 0
        assert non_neutral == 0
        assert survived is False
        # binomial should not be called for empty genome
        mock_binomial.assert_not_called()

    @patch("emergents.mutation_manager.np.random.binomial")
    def test_apply_mutations_to_genome_neutral_mutations(
        self, mock_binomial: Mock
    ) -> None:
        """Test applying neutral mutations to genome."""
        config = MutationConfig(weights={PointMutation: 1.0})
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        mock_binomial.return_value = 2  # 2 mutations

        # Mock the mutations to be neutral
        with (
            patch.object(manager, "create_mutation") as mock_create,
            patch.object(genome, "coalesce_all") as mock_coalesce,
        ):

            mock_mutation = Mock()
            mock_mutation.is_neutral.return_value = True
            mock_create.return_value = mock_mutation

            neutral, non_neutral, survived = manager.apply_mutations_to_genome(
                genome, 0.001
            )

        assert neutral == 2
        assert non_neutral == 0
        assert survived is True
        assert mock_mutation.apply.call_count == 2
        assert mock_coalesce.call_count == 2

    @patch("emergents.mutation_manager.np.random.binomial")
    def test_apply_mutations_to_genome_non_neutral_mutation(
        self, mock_binomial: Mock
    ) -> None:
        """Test applying non-neutral mutation that kills genome."""
        config = MutationConfig(weights={PointMutation: 1.0})
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        mock_binomial.return_value = 1  # 1 mutation

        # Mock the mutation to be non-neutral
        with patch.object(manager, "create_mutation") as mock_create:
            mock_mutation = Mock()
            mock_mutation.is_neutral.return_value = False
            mock_create.return_value = mock_mutation

            neutral, non_neutral, survived = manager.apply_mutations_to_genome(
                genome, 0.001
            )

        assert neutral == 0
        assert non_neutral == 1
        assert survived is False
        assert (
            mock_mutation.apply.call_count == 0
        )  # Should not apply non-neutral mutations

    @patch("emergents.mutation_manager.np.random.binomial")
    def test_apply_mutations_to_genome_exception_handling(
        self, mock_binomial: Mock
    ) -> None:
        """Test handling of exceptions during mutation application."""
        config = MutationConfig(weights={PointMutation: 1.0})
        manager = MutationManager(config)
        genome = self.create_test_genome(100)

        mock_binomial.return_value = 1  # 1 mutation

        # Mock mutation that raises exception
        with patch.object(manager, "create_mutation") as mock_create:
            mock_mutation = Mock()
            mock_mutation.is_neutral.side_effect = Exception("Mutation failed")
            mock_create.return_value = mock_mutation

            neutral, non_neutral, survived = manager.apply_mutations_to_genome(
                genome, 0.001
            )

        assert neutral == 0
        assert non_neutral == 1
        assert survived is False

    def test_apply_mutations_to_population_empty(self) -> None:
        """Test applying mutations to empty population."""
        config = MutationConfig()
        manager = MutationManager(config)

        mutation_counts, dead_indices = manager.apply_mutations_to_population([], 0.001)

        assert mutation_counts.total == 0
        assert mutation_counts.neutral == 0
        assert mutation_counts.non_neutral == 0
        assert dead_indices == []

    def test_apply_mutations_to_population_all_survive(self) -> None:
        """Test applying mutations to population where all genomes survive."""
        config = MutationConfig()
        manager = MutationManager(config)
        genomes = [self.create_test_genome(100) for _ in range(3)]

        # Mock apply_mutations_to_genome to return survival
        with patch.object(manager, "apply_mutations_to_genome") as mock_apply:
            mock_apply.return_value = (2, 0, True)  # 2 neutral, 0 non-neutral, survived

            mutation_counts, dead_indices = manager.apply_mutations_to_population(
                genomes, 0.001
            )

        assert mutation_counts.total == 6  # 3 genomes * 2 mutations each
        assert mutation_counts.neutral == 6
        assert mutation_counts.non_neutral == 0
        assert dead_indices == []
        assert mock_apply.call_count == 3

    def test_apply_mutations_to_population_some_die(self) -> None:
        """Test applying mutations to population where some genomes die."""
        config = MutationConfig()
        manager = MutationManager(config)
        genomes = [self.create_test_genome(100) for _ in range(4)]

        # Mock apply_mutations_to_genome with different outcomes
        with patch.object(manager, "apply_mutations_to_genome") as mock_apply:
            mock_apply.side_effect = [
                (1, 0, True),  # genome 0: survives
                (0, 1, False),  # genome 1: dies
                (2, 0, True),  # genome 2: survives
                (1, 1, False),  # genome 3: dies
            ]

            mutation_counts, dead_indices = manager.apply_mutations_to_population(
                genomes, 0.001
            )

        assert mutation_counts.total == 6  # 1+1+2+2 total mutations
        assert mutation_counts.neutral == 4  # 1+0+2+1 neutral mutations
        assert mutation_counts.non_neutral == 2  # 0+1+0+1 non-neutral mutations
        assert dead_indices == [1, 3]  # genomes 1 and 3 died


class TestMutationManagerIntegration:
    """Integration tests for MutationManager with real mutations."""

    def create_test_genome(self, length: int = 100) -> Genome:
        """Helper to create a test genome."""
        if length <= 0:
            return Genome()
        segments: list[Segment] = [NonCodingSegment(length=length)]
        return Genome(segments=segments)

    def test_mutation_selection_distribution(self) -> None:
        """Test that mutation selection follows configured weights."""
        # Create config with specific weights
        config = MutationConfig(
            weights={
                PointMutation: 0.8,
                SmallDeletion: 0.2,
            }
        )
        manager = MutationManager(config)

        # Select many mutations and check distribution
        selections = []
        random.seed(42)  # For reproducible test
        for _ in range(1000):
            selections.append(manager.select_mutation_type())

        point_mutations = sum(1 for sel in selections if sel == PointMutation)
        small_deletions = sum(1 for sel in selections if sel == SmallDeletion)

        # Should be roughly 80% point mutations, 20% small deletions
        assert abs(point_mutations / 1000 - 0.8) < 0.05
        assert abs(small_deletions / 1000 - 0.2) < 0.05

    def test_real_mutation_application(self) -> None:
        """Test applying real mutations to real genomes."""
        config = MutationConfig(
            weights={PointMutation: 1.0},  # Only point mutations for predictability
            small_mutation_max_size=1,
        )
        manager = MutationManager(config)
        genome = self.create_test_genome(50)
        original_length = len(genome)

        # Apply mutations with high rate to ensure some occur
        random.seed(123)  # For reproducible test
        neutral, non_neutral, survived = manager.apply_mutations_to_genome(genome, 0.1)

        # Point mutations should be neutral and not change length
        assert len(genome) == original_length
        assert survived is True
        # Should have had some mutations with 0.1 rate on 50bp genome
        assert neutral + non_neutral > 0

    def test_config_update_affects_behavior(self) -> None:
        """Test that updating config changes manager behavior."""
        # Start with one config
        initial_config = MutationConfig(weights={PointMutation: 1.0})
        manager = MutationManager(initial_config)

        # All selections should be PointMutation
        for _ in range(10):
            assert manager.select_mutation_type() == PointMutation

        # Update to different config
        new_config = MutationConfig(weights={SmallDeletion: 1.0})
        manager.update_config(new_config)

        # All selections should now be SmallDeletion
        for _ in range(10):
            assert manager.select_mutation_type() == SmallDeletion
