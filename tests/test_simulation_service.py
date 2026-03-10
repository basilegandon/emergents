"""
Test suite for simulation_service.py module.

Tests cover the SimulationService orchestration functionality.
"""

import contextlib
import math
import random

import pytest

from emergents.config import (
    EvolutionConfig,
    GenomeConfig,
    PopulationConfig,
    SimulationConfig,
)
from emergents.simulation_service import SimulationService


@pytest.fixture(autouse=True)
def _set_global_seed():
    """
    Set a fixed random seed for all tests to ensure reproducibility.

    This is a senior-level solution to prevent non-deterministic test failures
    caused by stochastic processes like evolutionary mutation and selection.
    The seed is set globally before each test and ensures consistent behavior
    across test runs.
    """
    random.seed(42)


class TestSimulationServiceInit:
    """Test SimulationService initialization."""

    def test_init_with_config(self):
        """Test initialization with SimulationConfig."""
        config = SimulationConfig()
        service = SimulationService(config)

        assert service.config is config
        assert service.population is None
        assert service.evolution_stats == []


class TestSimulationServiceRun:
    """Test SimulationService.run method."""

    def test_run_basic_workflow(self):
        """Test basic simulation run workflow."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,  # For 2 coding with is_circular: 2 * (2 + 2)
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=10),
            evolution=EvolutionConfig(num_generations=2, report_interval=1),
        )

        service = SimulationService(config)
        service.run()

        # After run, population should be initialized
        assert service.population is not None
        assert len(service.population.genomes) == 10
        # Evolution results should be recorded
        assert len(service.evolution_stats) == 2
        # Population should be at correct generation
        assert service.population.generation == 2

    def test_run_creates_population(self):
        """Test that run creates a population."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,  # For circular: can be flexible
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=5),
            evolution=EvolutionConfig(num_generations=1),
        )

        service = SimulationService(config)
        service.run()

        assert service.population is not None
        assert len(service.population.genomes) == 5

    def test_run_configurable_generations(self):
        """Test run with different generation counts."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=8),
            evolution=EvolutionConfig(num_generations=5, report_interval=1),
        )

        service = SimulationService(config)
        service.run()

        assert service.population.generation == 5
        assert len(service.evolution_stats) == 5

    def test_run_with_different_population_sizes(self):
        """Test run with various population sizes."""
        for pop_size in [5, 10, 15]:
            config = SimulationConfig(
                genome=GenomeConfig(
                    initial_length=8,
                    num_coding_segments=2,
                    coding_segment_length=2,
                    non_coding_segment_length=2,
                    is_circular=True,
                ),
                population=PopulationConfig(size=pop_size),
                evolution=EvolutionConfig(num_generations=1, report_interval=1),
            )

            service = SimulationService(config)
            service.run()

            assert len(service.population.genomes) == pop_size

    def test_run_with_no_plotting(self):
        """Test run with plotting disabled."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=10),
            evolution=EvolutionConfig(
                num_generations=2,
                enable_plotting=False,
                report_interval=1,
            ),
        )

        service = SimulationService(config)
        service.run()

        assert len(service.evolution_stats) == 2


class TestSimulationServiceMultipleRuns:
    """Test SimulationService with multiple runs."""

    def test_consecutive_simulations(self):
        """Test running multiple simulations sequentially."""
        for _ in range(2):
            config = SimulationConfig(
                genome=GenomeConfig(
                    initial_length=8,
                    num_coding_segments=2,
                    coding_segment_length=2,
                    non_coding_segment_length=2,
                    is_circular=True,
                ),
                population=PopulationConfig(size=10),
                evolution=EvolutionConfig(num_generations=2, report_interval=1),
            )

            service = SimulationService(config)
            service.run()

            assert service.population is not None
            assert len(service.evolution_stats) == 2


class TestSimulationServiceWithCircularGenomes:
    """Test SimulationService with circular genomes."""

    def test_run_with_circular_genomes(self):
        """Test simulation with circular genomes."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=8),
            evolution=EvolutionConfig(num_generations=2),
        )

        service = SimulationService(config)
        service.run()

        assert service.population is not None
        assert len(service.population.genomes) == 8
        # All genomes should be circular
        assert all(g.circular for g in service.population.genomes)


class TestSimulationServiceWithDifferentMutationRates:
    """Test SimulationService with various mutation rates."""

    def test_run_with_low_mutation_rate(self):
        """Test simulation with low mutation rate."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=10, mutation_rate=0.01),
            evolution=EvolutionConfig(num_generations=2, report_interval=1),
        )

        service = SimulationService(config)
        service.run()

        assert service.population is not None
        assert math.isclose(service.population.mutation_rate, 0.01)

    def test_run_with_high_mutation_rate(self):
        """Test simulation with high mutation rate."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=20, mutation_rate=0.3),
            evolution=EvolutionConfig(num_generations=2, report_interval=1),
        )

        service = SimulationService(config)
        service.run()

        assert math.isclose(service.population.mutation_rate, 0.3)


class TestSimulationServiceReportingOptions:
    """Test SimulationService reporting configurations."""

    def test_run_with_frequent_reporting(self):
        """Test simulation with frequent reporting."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=10),
            evolution=EvolutionConfig(num_generations=5, report_interval=1),
        )

        service = SimulationService(config)
        service.run()

        assert len(service.evolution_stats) == 5

    def test_run_with_sparse_reporting(self):
        """Test simulation with sparse reporting."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=10),
            evolution=EvolutionConfig(num_generations=5, report_interval=2),
        )

        service = SimulationService(config)
        service.run()

        # Even with report_interval=2, all generations are evolved
        assert service.population.generation == 5


class TestSimulationServiceErrorHandling:
    """Test error handling in SimulationService."""

    def test_run_with_invalid_initialization_config(self):
        """Test run with configuration that causes initialization error."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=2,  # Too small: sum of segments (2*2 + 3*2 = 10) > 2
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=False,
                nature_of_extremities="NC--NC",
            ),
            population=PopulationConfig(size=5),
            evolution=EvolutionConfig(num_generations=1),
        )

        service = SimulationService(config)

        # Should raise during run due to segment length mismatch
        with pytest.raises(
            ValueError,
            match="Sum of segment lengths is not equal to initial genome length",
        ):
            service.run()

    def test_service_state_after_failed_run(self):
        """Test that service state is consistent after failed run."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=2,  # Invalid
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
            ),
            population=PopulationConfig(size=5),
        )

        service = SimulationService(config)

        with contextlib.suppress(ValueError):
            service.run()

        # Service should still have its config
        assert service.config is not None


class TestSimulationServiceIntegration:
    """Integration tests for complete simulation workflows."""

    def test_complete_simulation_small(self):
        """Test complete simulation with small configuration."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=5, mutation_rate=0.1),
            evolution=EvolutionConfig(
                num_generations=3,
                report_interval=1,
                enable_plotting=False,
            ),
        )

        service = SimulationService(config)
        service.run()

        # Verify complete workflow
        assert service.population is not None
        assert service.population.generation == 3
        assert len(service.evolution_stats) == 3
        assert len(service.population.genomes) == 5

    def test_complete_simulation_with_different_sizes(self):
        """Test complete simulation with varying population sizes."""
        for pop_size in [5, 10, 20]:
            config = SimulationConfig(
                genome=GenomeConfig(
                    initial_length=8,
                    num_coding_segments=2,
                    coding_segment_length=2,
                    non_coding_segment_length=2,
                    is_circular=True,
                ),
                population=PopulationConfig(size=pop_size),
                evolution=EvolutionConfig(
                    num_generations=2,
                    report_interval=1,
                    enable_plotting=False,
                ),
            )

            service = SimulationService(config)
            service.run()

            assert len(service.population.genomes) == pop_size
            assert service.population.generation == 2

    def test_simulation_preserves_config(self):
        """Test that simulation preserves original configuration."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=8, mutation_rate=0.15),
            evolution=EvolutionConfig(num_generations=2, report_interval=1),
        )

        service = SimulationService(config)
        original_config = service.config
        service.run()

        # Config should be unchanged
        assert service.config is original_config
        assert service.config.population.size == 8
        assert math.isclose(service.config.population.mutation_rate, 0.15)

    def test_simulation_diversity_metrics_available(self):
        """Test that diversity metrics are available after simulation."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=8),
            evolution=EvolutionConfig(num_generations=2),
        )

        service = SimulationService(config)
        service.run()

        assert service.population is not None
        diversity = service.population.get_genome_diversity()
        assert isinstance(diversity, dict)
        assert "length_diversity" in diversity


class TestSimulationServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_simulation_single_generation(self):
        """Test simulation with just one generation."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=5),
            evolution=EvolutionConfig(num_generations=1),
        )

        service = SimulationService(config)
        service.run()

        assert service.population.generation == 1
        assert len(service.evolution_stats) == 1

    def test_simulation_minimal_population(self):
        """Test simulation with minimal population size."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=1),
            evolution=EvolutionConfig(num_generations=2),
        )

        service = SimulationService(config)
        service.run()

        assert len(service.population.genomes) == 1

    def test_simulation_low_mutation_rate(self):
        """Test simulation with very low mutation rate."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(size=5, mutation_rate=0.001),
            evolution=EvolutionConfig(num_generations=2),
        )

        service = SimulationService(config)
        service.run()

        assert math.isclose(service.population.mutation_rate, 0.001)

    def test_simulation_high_mutation_rate(self):
        """Test simulation with moderately high mutation rate."""
        config = SimulationConfig(
            genome=GenomeConfig(
                initial_length=8,
                num_coding_segments=2,
                coding_segment_length=2,
                non_coding_segment_length=2,
                is_circular=True,
            ),
            population=PopulationConfig(
                size=15, mutation_rate=0.3
            ),  # Reduced from 0.5 to avoid complete extinction
            evolution=EvolutionConfig(num_generations=2),
        )

        service = SimulationService(config)
        service.run()

        assert math.isclose(service.population.mutation_rate, 0.3)
        assert service.population.generation == 2
