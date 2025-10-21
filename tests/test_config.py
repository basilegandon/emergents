"""
Comprehensive unit tests for emergents.config module.
Tests all configuration classes, validation, and edge cases.
"""

import pytest

from emergents.config import (
    EvolutionConfig,
    GenomeConfig,
    MutationConfig,
    PopulationConfig,
    SimulationConfig,
)
from emergents.genome.segments import PromoterDirection
from emergents.mutations.base import Mutation
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion


class TestGenomeConfig:
    """Test GenomeConfig class."""

    def test_default_initialization(self) -> None:
        """Test default GenomeConfig creation."""
        config = GenomeConfig()

        assert config.initial_length == 10000
        assert config.num_coding_segments == 10
        assert config.coding_segment_length == 100
        assert config.non_coding_segment_length == 900
        assert config.promoter_direction == PromoterDirection.FORWARD
        assert config.is_circular is True
        assert config.nature_of_extremities == "NC--NC"

    def test_custom_initialization(self) -> None:
        """Test GenomeConfig with custom values."""
        config = GenomeConfig(
            initial_length=5000,
            num_coding_segments=5,
            coding_segment_length=50,
            non_coding_segment_length=450,
            promoter_direction=PromoterDirection.REVERSE,
            is_circular=False,
            nature_of_extremities="C--C",
        )

        assert config.initial_length == 5000
        assert config.num_coding_segments == 5
        assert config.coding_segment_length == 50
        assert config.non_coding_segment_length == 450
        assert config.promoter_direction == PromoterDirection.REVERSE
        assert config.is_circular is False
        assert config.nature_of_extremities == "C--C"

    def test_invalid_initial_length(self) -> None:
        """Test validation of initial_length."""
        with pytest.raises(ValueError, match="Initial length must be positive"):
            GenomeConfig(initial_length=0)

        with pytest.raises(ValueError, match="Initial length must be positive"):
            GenomeConfig(initial_length=-100)

    def test_invalid_num_coding_segments(self) -> None:
        """Test validation of num_coding_segments."""
        with pytest.raises(
            ValueError, match="Number of coding segments cannot be negative"
        ):
            GenomeConfig(num_coding_segments=-1)


class TestPopulationConfig:
    """Test PopulationConfig class."""

    def test_default_initialization(self) -> None:
        """Test default PopulationConfig creation."""
        config = PopulationConfig()

        assert config.size == 10000
        assert config.mutation_rate == 1e-4
        assert config.random_seed is None

    def test_custom_initialization(self) -> None:
        """Test PopulationConfig with custom values."""
        config = PopulationConfig(size=1000, mutation_rate=0.01, random_seed=42)

        assert config.size == 1000
        assert config.mutation_rate == 0.01
        assert config.random_seed == 42

    def test_invalid_population_size(self) -> None:
        """Test validation of population size."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            PopulationConfig(size=0)

        with pytest.raises(ValueError, match="Population size must be positive"):
            PopulationConfig(size=-100)

    def test_invalid_mutation_rate(self) -> None:
        """Test validation of mutation rate."""
        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            PopulationConfig(mutation_rate=-0.1)

        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            PopulationConfig(mutation_rate=1.5)

    def test_valid_mutation_rate_boundaries(self) -> None:
        """Test valid boundary values for mutation rate."""
        config1 = PopulationConfig(mutation_rate=0.0)
        assert config1.mutation_rate == 0.0

        config2 = PopulationConfig(mutation_rate=1.0)
        assert config2.mutation_rate == 1.0


class TestEvolutionConfig:
    """Test EvolutionConfig class."""

    def test_default_initialization(self) -> None:
        """Test default EvolutionConfig creation."""
        config = EvolutionConfig()

        assert config.num_generations == 1000
        assert config.report_interval == 10
        assert config.enable_progress_bar is True
        assert config.enable_plotting is True
        assert config.plot_filename == "evolution_progress.png"

    def test_custom_initialization(self) -> None:
        """Test EvolutionConfig with custom values."""
        config = EvolutionConfig(
            num_generations=500,
            report_interval=5,
            enable_progress_bar=False,
            enable_plotting=False,
            plot_filename="custom_plot.png",
        )

        assert config.num_generations == 500
        assert config.report_interval == 5
        assert config.enable_progress_bar is False
        assert config.enable_plotting is False
        assert config.plot_filename == "custom_plot.png"

    def test_invalid_num_generations(self) -> None:
        """Test validation of num_generations."""
        with pytest.raises(ValueError, match="Number of generations must be positive"):
            EvolutionConfig(num_generations=0)

        with pytest.raises(ValueError, match="Number of generations must be positive"):
            EvolutionConfig(num_generations=-10)

    def test_invalid_report_interval(self) -> None:
        """Test validation of report_interval."""
        with pytest.raises(ValueError, match="Report interval must be positive"):
            EvolutionConfig(report_interval=0)

        with pytest.raises(ValueError, match="Report interval must be positive"):
            EvolutionConfig(report_interval=-5)


class TestMutationConfig:
    """Test MutationConfig class."""

    def test_default_initialization(self) -> None:
        """Test default MutationConfig creation."""
        config = MutationConfig()

        # Check that all mutation types are present
        expected_mutations = {
            PointMutation,
            SmallDeletion,
            SmallInsertion,
            Deletion,
            Duplication,
            Inversion,
        }
        assert set(config.weights.keys()) == expected_mutations

        # Check that weights are normalized (sum to 1)
        total_weight = sum(config.weights.values())
        assert abs(total_weight - 1.0) < 1e-10

        # Check individual weights are equal (since default is 1.0 for all)
        expected_weight = 1.0 / 6  # 6 mutation types
        for weight in config.weights.values():
            assert abs(weight - expected_weight) < 1e-10

        assert config.small_mutation_max_size == 6

    def test_custom_weights(self) -> None:
        """Test MutationConfig with custom weights."""
        custom_weights = {
            PointMutation: 2.0,
            SmallDeletion: 1.0,
            SmallInsertion: 1.0,
            Deletion: 0.5,
            Duplication: 0.5,
            Inversion: 1.0,
        }

        config = MutationConfig(weights=custom_weights, small_mutation_max_size=10)

        # Check that weights are normalized
        total_weight = sum(config.weights.values())
        assert abs(total_weight - 1.0) < 1e-10

        # Check specific normalized weights
        expected_total = 6.0  # Sum of custom weights
        assert abs(config.weights[PointMutation] - (2.0 / expected_total)) < 1e-10
        assert abs(config.weights[Deletion] - (0.5 / expected_total)) < 1e-10

        assert config.small_mutation_max_size == 10

    def test_empty_weights(self) -> None:
        """Test validation of empty weights."""
        with pytest.raises(ValueError, match="Mutation weights cannot be empty"):
            MutationConfig(weights={})

    def test_zero_total_weight(self) -> None:
        """Test validation of zero total weight."""
        with pytest.raises(ValueError, match="Total mutation weight must be positive"):
            MutationConfig(weights={PointMutation: 0.0, SmallDeletion: 0.0})

    def test_negative_total_weight(self) -> None:
        """Test validation of negative total weight."""
        with pytest.raises(ValueError, match="Total mutation weight must be positive"):
            MutationConfig(weights={PointMutation: -1.0})

    def test_invalid_small_mutation_max_size(self) -> None:
        """Test validation of small_mutation_max_size."""
        with pytest.raises(
            ValueError, match="Small mutation max size must be positive"
        ):
            MutationConfig(small_mutation_max_size=0)

        with pytest.raises(
            ValueError, match="Small mutation max size must be positive"
        ):
            MutationConfig(small_mutation_max_size=-5)


class TestSimulationConfig:
    """Test SimulationConfig class."""

    def test_default_initialization(self) -> None:
        """Test default SimulationConfig creation."""
        config = SimulationConfig()

        # Check that all sub-configs are present and have correct types
        assert isinstance(config.genome, GenomeConfig)
        assert isinstance(config.population, PopulationConfig)
        assert isinstance(config.evolution, EvolutionConfig)
        assert isinstance(config.mutations, MutationConfig)

        # Spot check some default values
        assert config.genome.initial_length == 10000
        assert config.population.size == 10000
        assert config.evolution.num_generations == 1000
        assert config.mutations.small_mutation_max_size == 6

    def test_custom_initialization(self) -> None:
        """Test SimulationConfig with custom sub-configs."""
        custom_genome = GenomeConfig(initial_length=5000)
        custom_population = PopulationConfig(size=1000)
        custom_evolution = EvolutionConfig(num_generations=500)
        custom_mutations = MutationConfig(small_mutation_max_size=10)

        config = SimulationConfig(
            genome=custom_genome,
            population=custom_population,
            evolution=custom_evolution,
            mutations=custom_mutations,
        )

        assert config.genome.initial_length == 5000
        assert config.population.size == 1000
        assert config.evolution.num_generations == 500
        assert config.mutations.small_mutation_max_size == 10

    def test_create_default_class_method(self) -> None:
        """Test the create_default class method."""
        config = SimulationConfig.create_default()

        assert isinstance(config, SimulationConfig)
        assert isinstance(config.genome, GenomeConfig)
        assert isinstance(config.population, PopulationConfig)
        assert isinstance(config.evolution, EvolutionConfig)
        assert isinstance(config.mutations, MutationConfig)

        # Should be equivalent to default initialization
        default_config = SimulationConfig()
        assert config.genome.initial_length == default_config.genome.initial_length
        assert config.population.size == default_config.population.size

    def test_validate_all(self) -> None:
        """Test the validate_all method."""
        config = SimulationConfig()

        # Should not raise any exception for valid config
        config.validate_all()

        # Test with invalid sub-config
        config.genome.initial_length = -1
        # validate_all doesn't currently do additional validation beyond __post_init__
        # but we test that it exists and is callable
        config.validate_all()


class TestConfigIntegration:
    """Integration tests for configuration classes."""

    def test_full_configuration_workflow(self) -> None:
        """Test creating and using a complete configuration."""
        # Create a custom configuration
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8000, num_coding_segments=8, is_circular=False
            ),
            population=PopulationConfig(
                size=5000, mutation_rate=0.001, random_seed=123
            ),
            evolution=EvolutionConfig(
                num_generations=100, report_interval=5, enable_plotting=False
            ),
            mutations=MutationConfig(
                weights={
                    PointMutation: 3.0,
                    SmallDeletion: 1.0,
                    SmallInsertion: 1.0,
                    Deletion: 0.5,
                    Duplication: 0.5,
                    Inversion: 1.0,
                },
                small_mutation_max_size=8,
            ),
        )

        # Verify all components work together
        assert config.genome.initial_length == 8000
        assert config.population.size == 5000
        assert config.evolution.num_generations == 100
        assert config.mutations.small_mutation_max_size == 8

        # Verify mutation weights are normalized
        total_weight = sum(config.mutations.weights.values())
        assert abs(total_weight - 1.0) < 1e-10

    def test_configuration_immutability_patterns(self) -> None:
        """Test that configurations behave consistently after creation."""
        config1 = SimulationConfig.create_default()
        config2 = SimulationConfig.create_default()

        # Different instances should have same values
        assert config1.genome.initial_length == config2.genome.initial_length
        assert config1.population.size == config2.population.size

        # Modifying one shouldn't affect the other
        config1.genome.initial_length = 5000
        assert config2.genome.initial_length == 10000  # Still default
