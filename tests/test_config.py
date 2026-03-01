"""
Exhaustive tests for configuration management module.

Tests cover:
- Initialization with valid/invalid parameters
- Default values and their appropriateness
- Validation logic and edge cases
- Boundary conditions
- Weight normalization in MutationConfig
- Interactions between nested configs
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
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion


class TestGenomeConfigDefaults:
    """Test default values and initialization of GenomeConfig."""

    def test_default_values(self):
        """All default values should be sensible and match docstring."""
        config = GenomeConfig()
        assert config.initial_length == 1000
        assert config.num_coding_segments == 10
        assert config.coding_segment_length == 10
        assert config.non_coding_segment_length == 90
        assert config.promoter_direction == PromoterDirection.FORWARD
        assert config.is_circular is True
        assert config.nature_of_extremities == "NC--NC"

    def test_custom_values(self):
        """All attributes should be settable."""
        config = GenomeConfig(
            initial_length=500,
            num_coding_segments=5,
            coding_segment_length=20,
            non_coding_segment_length=50,
            promoter_direction=PromoterDirection.REVERSE,
            is_circular=False,
            nature_of_extremities="C--C",
        )
        assert config.initial_length == 500
        assert config.num_coding_segments == 5
        assert config.coding_segment_length == 20
        assert config.non_coding_segment_length == 50
        assert config.promoter_direction == PromoterDirection.REVERSE
        assert config.is_circular is False
        assert config.nature_of_extremities == "C--C"


class TestGenomeConfigValidation:
    """Test validation logic for GenomeConfig."""

    def test_negative_initial_length(self):
        """Initial length cannot be negative."""
        with pytest.raises(ValueError, match="Initial length must be positive"):
            GenomeConfig(initial_length=-1)

    def test_zero_initial_length(self):
        """Initial length cannot be zero."""
        with pytest.raises(ValueError, match="Initial length must be positive"):
            GenomeConfig(initial_length=0)

    def test_positive_initial_length_boundary(self):
        """Initial length can be 1 (minimum positive)."""
        config = GenomeConfig(initial_length=1)
        assert config.initial_length == 1

    def test_negative_num_coding_segments(self):
        """Number of coding segments cannot be negative."""
        with pytest.raises(
            ValueError, match="Number of coding segments cannot be negative"
        ):
            GenomeConfig(num_coding_segments=-1)

    def test_zero_num_coding_segments(self):
        """Zero coding segments should be allowed (no genes)."""
        config = GenomeConfig(num_coding_segments=0)
        assert config.num_coding_segments == 0

    def test_large_initial_length(self):
        """Should accept arbitrarily large initial length."""
        config = GenomeConfig(initial_length=1_000_000)
        assert config.initial_length == 1_000_000

    def test_large_num_coding_segments(self):
        """Should accept arbitrarily many coding segments."""
        config = GenomeConfig(num_coding_segments=10000)
        assert config.num_coding_segments == 10000

    def test_both_promoter_directions(self):
        """Both promoter directions should be valid."""
        config_forward = GenomeConfig(promoter_direction=PromoterDirection.FORWARD)
        config_reverse = GenomeConfig(promoter_direction=PromoterDirection.REVERSE)
        assert config_forward.promoter_direction == PromoterDirection.FORWARD
        assert config_reverse.promoter_direction == PromoterDirection.REVERSE

    def test_circular_and_linear_genomes(self):
        """Both circular and linear genomes should be valid."""
        circular = GenomeConfig(is_circular=True)
        linear = GenomeConfig(is_circular=False)
        assert circular.is_circular is True
        assert linear.is_circular is False

    def test_various_nature_of_extremities(self):
        """Different nature_of_extremities strings should be accepted."""
        configs = [
            GenomeConfig(nature_of_extremities="NC--NC"),
            GenomeConfig(nature_of_extremities="C--C"),
            GenomeConfig(nature_of_extremities="N--N"),
            GenomeConfig(nature_of_extremities=""),
        ]
        assert configs[0].nature_of_extremities == "NC--NC"
        assert configs[1].nature_of_extremities == "C--C"
        assert configs[2].nature_of_extremities == "N--N"
        assert configs[3].nature_of_extremities == ""


class TestPopulationConfigDefaults:
    """Test default values and initialization of PopulationConfig."""

    def test_default_values(self):
        """All default values should be sensible."""
        config = PopulationConfig()
        assert config.size == 1000
        assert config.mutation_rate == pytest.approx(1e-4)
        assert config.random_seed is None

    def test_custom_values(self):
        """All attributes should be settable."""
        config = PopulationConfig(size=500, mutation_rate=0.001, random_seed=42)
        assert config.size == 500
        assert config.mutation_rate == pytest.approx(0.001)
        assert config.random_seed == 42


class TestPopulationConfigValidation:
    """Test validation logic for PopulationConfig."""

    def test_negative_population_size(self):
        """Population size cannot be negative."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            PopulationConfig(size=-1)

    def test_zero_population_size(self):
        """Population size cannot be zero."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            PopulationConfig(size=0)

    def test_minimum_population_size(self):
        """Population size of 1 should be allowed."""
        config = PopulationConfig(size=1)
        assert config.size == 1

    def test_large_population_size(self):
        """Should accept very large population sizes."""
        config = PopulationConfig(size=1_000_000)
        assert config.size == 1_000_000

    def test_negative_mutation_rate(self):
        """Mutation rate cannot be negative."""
        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            PopulationConfig(mutation_rate=-0.001)

    def test_mutation_rate_exceeds_one(self):
        """Mutation rate cannot exceed 1."""
        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            PopulationConfig(mutation_rate=1.1)

    def test_zero_mutation_rate(self):
        """Zero mutation rate should be allowed (no mutations)."""
        config = PopulationConfig(mutation_rate=0.0)
        assert config.mutation_rate == pytest.approx(0.0)

    def test_mutation_rate_of_one(self):
        """Mutation rate of 1.0 should be allowed (all individuals mutate)."""
        config = PopulationConfig(mutation_rate=1.0)
        assert config.mutation_rate == pytest.approx(1.0)

    def test_typical_mutation_rates(self):
        """Various realistic mutation rates should work."""
        for rate in [1e-5, 1e-4, 1e-3, 0.01, 0.1]:
            config = PopulationConfig(mutation_rate=rate)
            assert config.mutation_rate == pytest.approx(rate)

    def test_random_seed_none(self):
        """Random seed can be None."""
        config = PopulationConfig(random_seed=None)
        assert config.random_seed is None

    def test_random_seed_zero(self):
        """Random seed can be 0."""
        config = PopulationConfig(random_seed=0)
        assert config.random_seed == 0

    def test_random_seed_positive(self):
        """Random seed can be positive."""
        config = PopulationConfig(random_seed=12345)
        assert config.random_seed == 12345

    def test_random_seed_large(self):
        """Random seed can be very large."""
        config = PopulationConfig(random_seed=2**31 - 1)
        assert config.random_seed == 2**31 - 1


class TestEvolutionConfigDefaults:
    """Test default values and initialization of EvolutionConfig."""

    def test_default_values(self):
        """All default values should be sensible."""
        config = EvolutionConfig()
        assert config.num_generations == 10000
        assert config.report_interval == 50
        assert config.enable_progress_bar is True
        assert config.enable_plotting is True
        assert config.plot_filename == "evolution_progress.png"

    def test_custom_values(self):
        """All attributes should be settable."""
        config = EvolutionConfig(
            num_generations=5000,
            report_interval=100,
            enable_progress_bar=False,
            enable_plotting=False,
            plot_filename="custom.png",
        )
        assert config.num_generations == 5000
        assert config.report_interval == 100
        assert config.enable_progress_bar is False
        assert config.enable_plotting is False
        assert config.plot_filename == "custom.png"


class TestEvolutionConfigValidation:
    """Test validation logic for EvolutionConfig."""

    def test_negative_num_generations(self):
        """Number of generations cannot be negative."""
        with pytest.raises(ValueError, match="Number of generations must be positive"):
            EvolutionConfig(num_generations=-1)

    def test_zero_num_generations(self):
        """Number of generations cannot be zero."""
        with pytest.raises(ValueError, match="Number of generations must be positive"):
            EvolutionConfig(num_generations=0)

    def test_minimum_num_generations(self):
        """One generation should be allowed."""
        config = EvolutionConfig(num_generations=1)
        assert config.num_generations == 1

    def test_large_num_generations(self):
        """Should accept very large generation counts."""
        config = EvolutionConfig(num_generations=1_000_000)
        assert config.num_generations == 1_000_000

    def test_negative_report_interval(self):
        """Report interval cannot be negative."""
        with pytest.raises(ValueError, match="Report interval must be positive"):
            EvolutionConfig(report_interval=-1)

    def test_zero_report_interval(self):
        """Report interval cannot be zero."""
        with pytest.raises(ValueError, match="Report interval must be positive"):
            EvolutionConfig(report_interval=0)

    def test_minimum_report_interval(self):
        """Report interval of 1 should be allowed."""
        config = EvolutionConfig(report_interval=1)
        assert config.report_interval == 1

    def test_report_interval_greater_than_generations(self):
        """Report interval can exceed number of generations (just won't report much)."""
        config = EvolutionConfig(num_generations=100, report_interval=1000)
        assert config.report_interval == 1000

    def test_boolean_flags(self):
        """Both True and False should work for boolean flags."""
        enabled = EvolutionConfig(enable_progress_bar=True, enable_plotting=True)
        disabled = EvolutionConfig(enable_progress_bar=False, enable_plotting=False)
        assert enabled.enable_progress_bar is True
        assert enabled.enable_plotting is True
        assert disabled.enable_progress_bar is False
        assert disabled.enable_plotting is False

    def test_various_plot_filenames(self):
        """Various valid filenames should be accepted."""
        for filename in [
            "output.png",
            "evolution.pdf",
            "simulation_results.jpg",
            "",
            "path/to/file.png",
        ]:
            config = EvolutionConfig(plot_filename=filename)
            assert config.plot_filename == filename


class TestMutationConfigDefaults:
    """Test default values and initialization of MutationConfig."""

    def test_default_values(self):
        """All default values should be sensible."""
        config = MutationConfig()
        assert config.seed == 42
        assert config.small_mutation_max_size == 6

    def test_default_weights_are_normalized(self):
        """Default weights should sum to 1.0."""
        config = MutationConfig()
        total_weight = sum(config.weights.values())
        assert pytest.approx(total_weight) == pytest.approx(1.0)

    def test_default_weights_all_present(self):
        """All mutation types should have non-zero weight."""
        config = MutationConfig()
        expected_types = {
            PointMutation,
            SmallDeletion,
            SmallInsertion,
            Deletion,
            Duplication,
            Inversion,
        }
        assert set(config.weights.keys()) == expected_types
        for weight in config.weights.values():
            assert weight > 0

    def test_custom_weights(self):
        """Custom weights should be accepted and normalized."""
        custom_weights = {
            PointMutation: 2.0,
            SmallDeletion: 1.0,
            SmallInsertion: 1.0,
            Deletion: 1.0,
            Duplication: 1.0,
            Inversion: 1.0,
        }
        config = MutationConfig(weights=custom_weights)
        total = sum(config.weights.values())
        assert pytest.approx(total) == pytest.approx(1.0)
        # Verify normalization: PointMutation should have double the weight of others
        assert pytest.approx(config.weights[PointMutation]) == pytest.approx(2.0 / 7.0)
        assert pytest.approx(config.weights[SmallDeletion]) == pytest.approx(1.0 / 7.0)

    def test_custom_seed(self):
        """Custom seed should be accepted."""
        config = MutationConfig(seed=999)
        assert config.seed == 999

    def test_custom_small_mutation_max_size(self):
        """Custom small mutation max size should be accepted."""
        config = MutationConfig(small_mutation_max_size=10)
        assert config.small_mutation_max_size == 10

    def test_weight_normalization_preserves_ratios(self):
        """Weight normalization should preserve relative ratios."""
        custom_weights = {
            PointMutation: 10.0,
            SmallDeletion: 5.0,
            SmallInsertion: 1.0,
            Deletion: 1.0,
            Duplication: 1.0,
            Inversion: 1.0,
        }
        config = MutationConfig(weights=custom_weights)
        # Ratio should be maintained: 10:5 = 2:1
        ratio = config.weights[PointMutation] / config.weights[SmallDeletion]
        assert pytest.approx(ratio) == pytest.approx(2.0)


class TestMutationConfigValidation:
    """Test validation logic for MutationConfig."""

    def test_empty_weights_raises_error(self):
        """Empty weights dictionary should raise ValueError."""
        with pytest.raises(ValueError, match="Mutation weights cannot be empty"):
            MutationConfig(weights={})

    def test_all_zero_weights_raises_error(self):
        """All-zero weights should raise ValueError."""
        with pytest.raises(ValueError, match="Total mutation weight must be positive"):
            MutationConfig(
                weights={
                    PointMutation: 0,
                    SmallDeletion: 0,
                    SmallInsertion: 0,
                    Deletion: 0,
                    Duplication: 0,
                    Inversion: 0,
                }
            )

    def test_negative_total_weight_raises_error(self):
        """Negative total weight should raise ValueError."""
        with pytest.raises(ValueError, match="Total mutation weight must be positive"):
            MutationConfig(
                weights={
                    PointMutation: -10.0,
                    SmallDeletion: 1.0,
                    SmallInsertion: 1.0,
                    Deletion: 1.0,
                    Duplication: 1.0,
                    Inversion: 1.0,
                }
            )

    def test_negative_small_mutation_max_size(self):
        """Small mutation max size cannot be negative."""
        with pytest.raises(
            ValueError, match="Small mutation max size must be positive"
        ):
            MutationConfig(small_mutation_max_size=-1)

    def test_zero_small_mutation_max_size(self):
        """Small mutation max size cannot be zero."""
        with pytest.raises(
            ValueError, match="Small mutation max size must be positive"
        ):
            MutationConfig(small_mutation_max_size=0)

    def test_minimum_small_mutation_max_size(self):
        """Small mutation max size of 1 should be allowed."""
        config = MutationConfig(small_mutation_max_size=1)
        assert config.small_mutation_max_size == 1

    def test_large_small_mutation_max_size(self):
        """Large small mutation max size should be allowed."""
        config = MutationConfig(small_mutation_max_size=1000)
        assert config.small_mutation_max_size == 1000

    def test_single_mutation_type_weight(self):
        """Can specify weight for only one mutation type."""
        weights = {PointMutation: 1.0}
        config = MutationConfig(weights=weights)
        assert config.weights[PointMutation] == pytest.approx(1.0)
        assert len(config.weights) == 1

    def test_very_small_weights(self):
        """Very small weights should still normalize correctly."""
        weights = {
            PointMutation: 1e-10,
            SmallDeletion: 1e-10,
            SmallInsertion: 1e-10,
            Deletion: 1e-10,
            Duplication: 1e-10,
            Inversion: 1e-10,
        }
        config = MutationConfig(weights=weights)
        total = sum(config.weights.values())
        assert pytest.approx(total) == pytest.approx(1.0)

    def test_very_large_weights(self):
        """Very large weights should still normalize correctly."""
        weights = {
            PointMutation: 1e10,
            SmallDeletion: 1e10,
            SmallInsertion: 1e10,
            Deletion: 1e10,
            Duplication: 1e10,
            Inversion: 1e10,
        }
        config = MutationConfig(weights=weights)
        total = sum(config.weights.values())
        assert pytest.approx(total) == pytest.approx(1.0)


class TestSimulationConfigDefaults:
    """Test default values and initialization of SimulationConfig."""

    def test_default_values(self):
        """All nested configs should have defaults."""
        config = SimulationConfig()
        assert isinstance(config.genome, GenomeConfig)
        assert isinstance(config.population, PopulationConfig)
        assert isinstance(config.evolution, EvolutionConfig)
        assert isinstance(config.mutations, MutationConfig)

    def test_create_default_classmethod(self):
        """create_default() should create identical config to default init."""
        default1 = SimulationConfig()
        default2 = SimulationConfig.create_default()

        assert default1.genome.initial_length == default2.genome.initial_length
        assert default1.population.size == default2.population.size
        assert default1.evolution.num_generations == default2.evolution.num_generations

    def test_custom_nested_configs(self):
        """Can override all nested configs."""
        custom_genome = GenomeConfig(initial_length=2000)
        custom_population = PopulationConfig(size=500)
        custom_evolution = EvolutionConfig(num_generations=5000)
        custom_mutations = MutationConfig(seed=123)

        config = SimulationConfig(
            genome=custom_genome,
            population=custom_population,
            evolution=custom_evolution,
            mutations=custom_mutations,
        )

        assert config.genome.initial_length == 2000
        assert config.population.size == 500
        assert config.evolution.num_generations == 5000
        assert config.mutations.seed == 123

    def test_custom_partial_nested_configs(self):
        """Can override only some nested configs."""
        custom_genome = GenomeConfig(initial_length=500)
        config = SimulationConfig(genome=custom_genome)

        assert config.genome.initial_length == 500
        assert config.population.size == 1000  # default
        assert config.evolution.num_generations == 10000  # default
        assert config.mutations.seed == 42  # default

    def test_nested_config_validation_still_works(self):
        """Invalid nested configs should still raise errors during creation."""
        # The error should be raised during GenomeConfig creation
        with pytest.raises(ValueError, match="Initial length must be positive"):
            GenomeConfig(initial_length=-1)


class TestSimulationConfigIntegration:
    """Test interactions and integration between configs."""

    def test_realistic_small_simulation(self):
        """Create realistic small simulation config."""
        config = SimulationConfig(
            genome=GenomeConfig(initial_length=500, num_coding_segments=5),
            population=PopulationConfig(size=100, mutation_rate=0.001),
            evolution=EvolutionConfig(num_generations=1000, report_interval=100),
            mutations=MutationConfig(seed=42),
        )
        assert config.genome.initial_length == 500
        assert config.population.size == 100
        assert config.evolution.num_generations == 1000

    def test_realistic_large_simulation(self):
        """Create realistic large simulation config."""
        config = SimulationConfig(
            genome=GenomeConfig(initial_length=10000, num_coding_segments=50),
            population=PopulationConfig(size=10000, mutation_rate=1e-5),
            evolution=EvolutionConfig(num_generations=100000, report_interval=1000),
            mutations=MutationConfig(seed=999),
        )
        assert config.genome.initial_length == 10000
        assert config.population.size == 10000
        assert config.evolution.num_generations == 100000

    def test_minimal_simulation(self):
        """Create minimal valid simulation config."""
        config = SimulationConfig(
            genome=GenomeConfig(initial_length=1, num_coding_segments=0),
            population=PopulationConfig(size=1, mutation_rate=0.0),
            evolution=EvolutionConfig(num_generations=1, report_interval=1),
            mutations=MutationConfig(
                weights={PointMutation: 1.0},
                small_mutation_max_size=1,
            ),
        )
        assert config.genome.initial_length == 1
        assert config.population.size == 1

    def test_configs_are_independent(self):
        """Modifying one config instance shouldn't affect another."""
        config1 = SimulationConfig()
        config2 = SimulationConfig()

        config1.genome.initial_length = 2000
        assert config2.genome.initial_length == 1000  # unchanged

    def test_mutation_weights_independent_across_instances(self):
        """Mutation weights in different instances should be independent."""
        config1 = MutationConfig()
        config2 = MutationConfig()

        # Verify they're equal but independent
        assert config1.weights[PointMutation] == pytest.approx(
            config2.weights[PointMutation]
        )

        # Modify config1's weights (if mutating dict)
        config1.weights[PointMutation] = 0.5
        assert config2.weights[PointMutation] != pytest.approx(0.5)  # config2 unchanged


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions across all configs."""

    def test_all_false_boolean_flags(self):
        """Disabling all features should be valid."""
        config = EvolutionConfig(
            enable_progress_bar=False,
            enable_plotting=False,
        )
        assert config.enable_progress_bar is False
        assert config.enable_plotting is False

    def test_report_interval_equals_num_generations(self):
        """Report interval equal to num_generations should work."""
        config = EvolutionConfig(
            num_generations=100,
            report_interval=100,
        )
        assert config.report_interval == config.num_generations

    def test_very_high_mutation_rate_with_small_population(self):
        """High mutation rate with small population should be valid."""
        config = PopulationConfig(size=2, mutation_rate=0.99)
        assert config.size == 2
        assert config.mutation_rate == pytest.approx(0.99)

    def test_genome_config_with_single_coding_segment(self):
        """Single coding segment should be valid."""
        config = GenomeConfig(num_coding_segments=1)
        assert config.num_coding_segments == 1

    def test_genome_config_with_many_coding_segments(self):
        """Genome with many coding segments should be valid."""
        config = GenomeConfig(num_coding_segments=1000)
        assert config.num_coding_segments == 1000

    def test_mutation_weights_unequal_distribution(self):
        """Highly skewed weight distribution should normalize correctly."""
        weights = {
            PointMutation: 1000.0,
            SmallDeletion: 1.0,
            SmallInsertion: 1.0,
            Deletion: 1.0,
            Duplication: 1.0,
            Inversion: 1.0,
        }
        config = MutationConfig(weights=weights)
        total = sum(config.weights.values())
        assert pytest.approx(total) == pytest.approx(1.0)
        # PointMutation should dominate
        assert config.weights[PointMutation] > 0.99

    def test_circular_vs_linear_genome_config(self):
        """Both circular and linear should be valid independently."""
        circular = GenomeConfig(is_circular=True)
        linear = GenomeConfig(is_circular=False)
        assert circular.is_circular is True
        assert linear.is_circular is False
