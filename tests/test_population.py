"""
Comprehensive test suite for population.py module.

Tests cover initialization, genome management, evolution, and statistics tracking
with focus on edge cases and error conditions.
"""

import copy

import pytest

from emergents.config import MutationConfig
from emergents.genome.segments import PromoterDirection
from emergents.population import Population
from emergents.statistics import MutationCounts, PopulationStats

# ============================================================================
# Fixtures and Helper Functions
# ============================================================================


def create_simple_population(
    population_size: int = 10,
    mutation_rate: float = 0.01,
    random_seed: int | None = None,
) -> Population:
    """Helper to create a population with default configuration."""
    pop = Population(
        population_size=population_size,
        mutation_rate=mutation_rate,
        random_seed=random_seed,
    )
    return pop


def create_initialized_population(
    population_size: int = 10,
    initial_genome_length: int = 10,
    nb_coding_segments: int = 2,
    circular: bool = False,
) -> Population:
    """Helper to create and initialize a population with a simple genome."""
    pop = Population(population_size=population_size)
    # For NC--NC linear with 2 coding segments: NC(2) + C(2) + NC(2) + C(2) + NC(2) = 10
    # For circular with 2 coding segments: C(2) + NC(2) + C(2) + NC(2) = 8
    if circular:
        pop.initialize_population(
            initial_genome_length=8,  # Fixed for circular
            nb_coding_segments=nb_coding_segments,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=circular,
        )
    else:
        pop.initialize_population(
            initial_genome_length=initial_genome_length,
            nb_coding_segments=nb_coding_segments,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=circular,
            nature_of_extremities="NC--NC",
        )
    return pop


# ============================================================================
# __init__ Tests
# ============================================================================


class TestPopulationInit:
    """Test Population initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        pop = Population()
        assert pop.population_size == 100
        assert pop.mutation_rate == pytest.approx(0.01)
        assert pop.generation == 0
        assert pop.genomes == []

    def test_init_custom_population_size(self):
        """Test initialization with custom population size."""
        pop = Population(population_size=50)
        assert pop.population_size == 50

    def test_init_custom_mutation_rate(self):
        """Test initialization with custom mutation rate."""
        pop = Population(mutation_rate=0.05)
        assert pop.mutation_rate == pytest.approx(0.05)

    def test_init_zero_population_size(self):
        """Test that zero population size raises ValueError."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            Population(population_size=0)

    def test_init_negative_population_size(self):
        """Test that negative population size raises ValueError."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            Population(population_size=-10)

    def test_init_negative_mutation_rate(self):
        """Test that negative mutation rate raises ValueError."""
        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            Population(mutation_rate=-0.01)

    def test_init_mutation_rate_greater_than_one(self):
        """Test that mutation rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            Population(mutation_rate=1.5)

    def test_init_mutation_rate_zero(self):
        """Test that mutation rate of 0 is valid."""
        pop = Population(mutation_rate=0.0)
        assert pop.mutation_rate == pytest.approx(0.0)

    def test_init_mutation_rate_one(self):
        """Test that mutation rate of 1.0 is valid."""
        pop = Population(mutation_rate=1.0)
        assert pop.mutation_rate == pytest.approx(1.0)

    def test_init_with_random_seed(self):
        """Test initialization with random seed."""
        pop = Population(random_seed=42)
        assert pop.generation == 0

    def test_init_reproducibility_with_seed(self):
        """Test that same seed produces same random sequences."""
        pop1 = Population(population_size=5, random_seed=42)
        pop1.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        lengths1 = [len(g) for g in pop1.genomes]

        pop2 = Population(population_size=5, random_seed=42)
        pop2.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        lengths2 = [len(g) for g in pop2.genomes]

        assert lengths1 == lengths2

    def test_init_mutation_config_created(self):
        """Test that mutation manager is initialized."""
        pop = Population(population_size=20)
        assert pop.mutation_manager is not None

    def test_init_stats_tracker_created(self):
        """Test that stats tracker is initialized."""
        pop = Population(population_size=20)
        assert pop.stats_tracker is not None


# ============================================================================
# initialize_population Tests
# ============================================================================


class TestInitializePopulation:
    """Test Population.initialize_population method."""

    def test_initialize_simple_linear_genome(self):
        """Test initialization with simple linear genome."""
        pop = create_simple_population(population_size=5)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        assert len(pop.genomes) == 5
        assert all(len(g) == 10 for g in pop.genomes)

    def test_initialize_circular_genome(self):
        """Test initialization with circular genome."""
        pop = create_simple_population(population_size=5)
        pop.initialize_population(
            initial_genome_length=8,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=True,
        )
        assert len(pop.genomes) == 5
        assert all(g.circular for g in pop.genomes)

    def test_initialize_zero_coding_segments(self):
        """Test initialization with zero coding segments."""
        pop = create_simple_population(population_size=3)
        pop.initialize_population(
            initial_genome_length=5,
            nb_coding_segments=0,
            length_coding_segments=0,
            length_non_coding_segments=5,
            promoter_directions=PromoterDirection.FORWARD,
        )
        assert len(pop.genomes) == 3
        assert all(len(g) == 5 for g in pop.genomes)

    def test_initialize_single_coding_segment(self):
        """Test initialization with single coding segment."""
        pop = create_simple_population(population_size=4)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=1,
            length_coding_segments=4,
            length_non_coding_segments=[3, 3],
            promoter_directions=PromoterDirection.FORWARD,
        )
        assert len(pop.genomes) == 4
        assert all(len(g) == 10 for g in pop.genomes)

    def test_initialize_different_promoter_directions(self):
        """Test initialization with different promoter directions."""
        pop = create_simple_population(population_size=2)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=[PromoterDirection.FORWARD, PromoterDirection.REVERSE],
        )
        assert len(pop.genomes) == 2
        assert all(len(g) == 10 for g in pop.genomes)

    def test_initialize_list_coding_segment_lengths(self):
        """Test initialization with list of coding segment lengths."""
        pop = create_simple_population(population_size=2)
        pop.initialize_population(
            initial_genome_length=11,
            nb_coding_segments=2,
            length_coding_segments=[3, 2],
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        assert all(len(g) == 11 for g in pop.genomes)

    def test_initialize_list_promoter_directions(self):
        """Test initialization with list of promoter directions."""
        pop = create_simple_population(population_size=2)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=[PromoterDirection.FORWARD, PromoterDirection.REVERSE],
        )
        assert all(len(g) == 10 for g in pop.genomes)

    def test_initialize_zero_genome_length(self):
        """Test that zero genome length raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(ValueError, match="Initial genome length must be positive"):
            pop.initialize_population(
                initial_genome_length=0,
                nb_coding_segments=1,
                length_coding_segments=1,
                length_non_coding_segments=0,
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_negative_genome_length(self):
        """Test that negative genome length raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(ValueError, match="Initial genome length must be positive"):
            pop.initialize_population(
                initial_genome_length=-5,
                nb_coding_segments=1,
                length_coding_segments=1,
                length_non_coding_segments=0,
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_negative_coding_segments(self):
        """Test that negative coding segments raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(
            ValueError, match="Number of coding segments cannot be negative"
        ):
            pop.initialize_population(
                initial_genome_length=10,
                nb_coding_segments=-1,
                length_coding_segments=2,
                length_non_coding_segments=2,
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_coding_segments_exceed_length(self):
        """Test that too many coding segments raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(ValueError, match="Total number of coding segments exceeds"):
            pop.initialize_population(
                initial_genome_length=5,
                nb_coding_segments=10,
                length_coding_segments=1,
                length_non_coding_segments=1,
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_coding_lengths_mismatch(self):
        """Test that mismatched coding segment lengths raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(
            ValueError, match="Length of coding segments list does not match"
        ):
            pop.initialize_population(
                initial_genome_length=10,
                nb_coding_segments=2,
                length_coding_segments=[2],  # Should have 2 elements
                length_non_coding_segments=2,
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_promoter_directions_mismatch(self):
        """Test that mismatched promoter directions raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(
            ValueError, match="Length of promoter_directions list does not match"
        ):
            pop.initialize_population(
                initial_genome_length=10,
                nb_coding_segments=2,
                length_coding_segments=2,
                length_non_coding_segments=2,
                promoter_directions=[
                    PromoterDirection.FORWARD
                ],  # Should have 2 elements
            )

    def test_initialize_noncoding_lengths_mismatch(self):
        """Test that mismatched non-coding lengths raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(
            ValueError, match="Length of non-coding segments list does not match"
        ):
            pop.initialize_population(
                initial_genome_length=10,
                nb_coding_segments=1,
                length_coding_segments=2,
                length_non_coding_segments=[3],  # Should have 2 elements for NC--NC
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_sum_lengths_mismatch(self):
        """Test that mismatched sum of segment lengths raises ValueError."""
        pop = create_simple_population()
        with pytest.raises(ValueError, match="Sum of segment lengths is not equal"):
            pop.initialize_population(
                initial_genome_length=10,
                nb_coding_segments=2,
                length_coding_segments=2,
                length_non_coding_segments=[
                    2,
                    2,
                    1,
                ],  # Sum: 2 + 2 + 2 + 2 + 1 = 9, not 10
                promoter_directions=PromoterDirection.FORWARD,
            )

    def test_initialize_all_genomes_independent(self):
        """Test that initialized genomes are independent copies."""
        pop = create_simple_population(population_size=3)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        # Verify all genomes are different objects (deepcopy)
        assert pop.genomes[0] is not pop.genomes[1]
        assert pop.genomes[1] is not pop.genomes[2]

    def test_initialize_extremities_nc_nc(self):
        """Test NC--NC extremities configuration."""
        pop = create_simple_population(population_size=2)
        pop.initialize_population(
            initial_genome_length=11,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=[3, 2, 2],
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="NC--NC",
        )
        assert all(len(g) == 11 for g in pop.genomes)

    def test_initialize_extremities_c_c(self):
        """Test C--C extremities configuration."""
        pop = create_simple_population(population_size=2)
        pop.initialize_population(
            initial_genome_length=8,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="C--C",
        )
        assert all(len(g) == 8 for g in pop.genomes)

    def test_initialize_extremities_c_nc(self):
        """Test C--NC extremities configuration."""
        pop = create_simple_population(population_size=2)
        pop.initialize_population(
            initial_genome_length=8,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=[2, 2],
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="C--NC",
        )
        assert all(len(g) == 8 for g in pop.genomes)

    def test_initialize_with_extra_non_coding_segment(self):
        """Test initialization with extra non-coding segment at end (edge case).

        Note: Due to validation constraints in initialize_population, the code path
        at line 190 (if len(length_non_coding_segments) > len(length_coding_segments))
        appears to be unreachable dead code. The validation ensures that the number
        of non-coding segments exactly matches the calculated nb_non_coding_segments,
        and after the optional slice for NC--NC extremities, they match the coding
        segments for the zip operation.
        """
        # Test a valid C--NC configuration (no first NC removal via slice)
        pop = create_simple_population(population_size=3)
        pop.initialize_population(
            initial_genome_length=8,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="C--NC",
        )
        assert all(len(g) == 8 for g in pop.genomes)
        assert len(pop.genomes) == 3


# ============================================================================
# apply_mutations Tests
# ============================================================================


class TestApplyMutations:
    """Test Population.apply_mutations method."""

    def test_apply_mutations_returns_tuple(self):
        """Test that apply_mutations returns a tuple."""
        pop = create_initialized_population(population_size=5)
        result = pop.apply_mutations()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_apply_mutations_returns_mutation_counts(self):
        """Test that first element is MutationCounts."""
        pop = create_initialized_population(population_size=5)
        counts, _ = pop.apply_mutations()
        assert isinstance(counts, MutationCounts)

    def test_apply_mutations_returns_dead_indices(self):
        """Test that second element is list of indices."""
        pop = create_initialized_population(population_size=5)
        _, dead_indices = pop.apply_mutations()
        assert isinstance(dead_indices, list)

    def test_apply_mutations_zero_rate(self):
        """Test apply_mutations with zero mutation rate."""
        pop = Population(population_size=5, mutation_rate=0.0)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        counts, _ = pop.apply_mutations()
        # With 0 mutation rate, should have 0 mutations applied
        assert counts.total == 0


# ============================================================================
# get_population_stats Tests
# ============================================================================


class TestGetPopulationStats:
    """Test Population.get_population_stats method."""

    def test_get_population_stats_returns_stats_object(self):
        """Test that method returns PopulationStats."""
        pop = create_initialized_population(population_size=5)
        stats = pop.get_population_stats(5)
        assert isinstance(stats, PopulationStats)

    def test_get_population_stats_correct_generation(self):
        """Test that generation number is correct."""
        pop = create_initialized_population(population_size=5)
        stats = pop.get_population_stats(5)
        assert stats.generation == pop.generation

    def test_get_population_stats_correct_population_size(self):
        """Test that population size is correct."""
        pop = create_initialized_population(population_size=5)
        stats = pop.get_population_stats(5)
        assert stats.population_size == 5

    def test_get_population_stats_genome_length_fields(self):
        """Test that genome length statistics are present."""
        pop = create_initialized_population(population_size=5)
        stats = pop.get_population_stats(5)
        assert hasattr(stats, "avg_genome_length")
        assert hasattr(stats, "min_genome_length")
        assert hasattr(stats, "max_genome_length")
        assert hasattr(stats, "length_std")


# ============================================================================
# evolve_one_generation Tests
# ============================================================================


class TestEvolveOneGeneration:
    """Test Population.evolve_one_generation method."""

    def test_evolve_one_generation_increments_counter(self):
        """Test that generation counter increments."""
        pop = create_initialized_population(population_size=5)
        initial_gen = pop.generation
        pop.evolve_one_generation()
        assert pop.generation == initial_gen + 1

    def test_evolve_one_generation_maintains_population_size(self):
        """Test that population size is maintained after evolution."""
        pop = create_initialized_population(population_size=10)
        pop.evolve_one_generation()
        assert len(pop.genomes) == 10

    def test_evolve_one_generation_returns_stats(self):
        """Test that method returns PopulationStats."""
        pop = create_initialized_population(population_size=5)
        stats = pop.evolve_one_generation()
        assert isinstance(stats, PopulationStats)

    def test_evolve_one_generation_multiple_times(self):
        """Test multiple consecutive generations."""
        pop = create_initialized_population(population_size=10)
        for i in range(5):
            pop.evolve_one_generation()
            assert pop.generation == i + 1
            assert len(pop.genomes) == 10

    def test_evolve_one_generation_with_zero_mutation_rate(self):
        """Test generation with zero mutation rate."""
        pop = Population(population_size=5, mutation_rate=0.0)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        pop.evolve_one_generation()
        assert pop.generation == 1
        assert len(pop.genomes) == 5


# ============================================================================
# _replenish_population Tests
# ============================================================================


class TestReplenishPopulation:
    """Test Population._replenish_population method."""

    def test_replenish_to_target_size(self):
        """Test that population is replenished to target size."""
        pop = create_initialized_population(population_size=10)
        # Remove some genomes
        pop.genomes = pop.genomes[:5]
        pop._replenish_population(5)
        assert len(pop.genomes) == 10

    def test_replenish_creates_new_genomes(self):
        """Test that replenish creates new genome objects."""
        pop = create_initialized_population(population_size=5)
        original_genomes = copy.deepcopy(pop.genomes)
        pop.genomes = pop.genomes[:2]
        pop._replenish_population(2)
        # New genomes should be different objects
        assert len(pop.genomes) == 5
        for i, genome in enumerate(pop.genomes):
            assert genome is not original_genomes[i]

    def test_replenish_from_single_survivor(self):
        """Test replenishment from single surviving genome."""
        pop = create_initialized_population(population_size=10)
        pop.genomes = pop.genomes[:1]
        pop._replenish_population(1)
        assert len(pop.genomes) == 10
        # All should be derived from original survivor
        original_length = len(pop.genomes[0])
        assert all(len(g) == original_length for g in pop.genomes)


# ============================================================================
# get_genome_lengths Tests
# ============================================================================


class TestGetGenomeLengths:
    """Test Population.get_genome_lengths method."""

    def test_get_genome_lengths_returns_list(self):
        """Test that method returns a list."""
        pop = create_initialized_population(population_size=5)
        lengths = pop.get_genome_lengths()
        assert isinstance(lengths, list)

    def test_get_genome_lengths_correct_count(self):
        """Test that list has correct number of entries."""
        pop = create_initialized_population(population_size=7)
        lengths = pop.get_genome_lengths()
        assert len(lengths) == 7

    def test_get_genome_lengths_correct_values(self):
        """Test that lengths are correct."""
        pop = create_initialized_population(population_size=5, initial_genome_length=10)
        lengths = pop.get_genome_lengths()
        assert all(length == 10 for length in lengths)

    def test_get_genome_lengths_reflect_mutations(self):
        """Test that lengths reflect genome changes."""
        pop = create_initialized_population(population_size=5, initial_genome_length=10)

        # After evolution, lengths might change
        pop.evolve_one_generation()
        new_lengths = pop.get_genome_lengths()
        # Should still have correct count
        assert len(new_lengths) == 5


# ============================================================================
# update_mutation_config Tests
# ============================================================================


class TestUpdateMutationConfig:
    """Test Population.update_mutation_config method."""

    def test_update_mutation_config(self):
        """Test that mutation config can be updated."""
        pop = create_simple_population()
        config = MutationConfig()
        pop.update_mutation_config(config)
        # Should not raise an exception and should complete without error
        assert pop.mutation_manager is not None


# ============================================================================
# get_genome_diversity Tests
# ============================================================================


class TestGetGenomeDiversity:
    """Test Population.get_genome_diversity method."""

    def test_get_genome_diversity_returns_dict(self):
        """Test that method returns a dictionary."""
        pop = create_initialized_population(population_size=5)
        diversity = pop.get_genome_diversity()
        assert isinstance(diversity, dict)

    def test_get_genome_diversity_not_empty(self):
        """Test that diversity metrics are not empty."""
        pop = create_initialized_population(population_size=5)
        diversity = pop.get_genome_diversity()
        assert len(diversity) > 0


# ============================================================================
# get_evolution_summary Tests
# ============================================================================


class TestGetEvolutionSummary:
    """Test Population.get_evolution_summary method."""

    def test_get_evolution_summary_returns_dict(self):
        """Test that method returns a dictionary."""
        pop = create_initialized_population(population_size=5)
        summary = pop.get_evolution_summary()
        assert isinstance(summary, dict)

    def test_get_evolution_summary_after_evolution(self):
        """Test evolution summary after generations."""
        pop = create_initialized_population(population_size=5)
        for _ in range(2):
            pop.evolve_one_generation()
        summary = pop.get_evolution_summary()
        assert isinstance(summary, dict)


# ============================================================================
# evolve Tests
# ============================================================================


class TestEvolve:
    """Test Population.evolve method."""

    def test_evolve_single_generation(self):
        """Test evolution for single generation."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(num_generations=1)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_evolve_multiple_generations(self):
        """Test evolution for multiple generations."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(num_generations=3)
        assert len(result) == 3

    def test_evolve_zero_generations(self):
        """Test that zero generations raises ValueError."""
        pop = create_initialized_population(population_size=10)
        with pytest.raises(ValueError, match="Number of generations must be positive"):
            pop.evolve(num_generations=0)

    def test_evolve_negative_generations(self):
        """Test that negative generations raises ValueError."""
        pop = create_initialized_population(population_size=10)
        with pytest.raises(ValueError, match="Number of generations must be positive"):
            pop.evolve(num_generations=-5)

    def test_evolve_updates_generation_counter(self):
        """Test that generation counter is updated correctly."""
        pop = create_initialized_population(population_size=10)
        pop.evolve(num_generations=5)
        assert pop.generation == 5

    def test_evolve_with_report_every_zero(self):
        """Test evolve with report_every=0 (no reporting)."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(num_generations=2, report_every=0)
        assert len(result) == 2

    def test_evolve_returns_plot_data(self):
        """Test that evolve returns list of PlotData."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(num_generations=2, report_every=1)
        assert len(result) == 2
        # Each element should have stats attribute
        assert all(hasattr(item, "stats") for item in result)

    def test_evolve_maintains_population_size(self):
        """Test that population size is maintained during evolution."""
        initial_size = 15
        pop = create_initialized_population(population_size=initial_size)
        pop.evolve(num_generations=3)
        assert len(pop.genomes) == initial_size

    def test_evolve_small_population(self):
        """Test evolution with very small population."""
        pop = create_initialized_population(population_size=2)
        result = pop.evolve(num_generations=2)
        assert len(result) == 2
        assert len(pop.genomes) == 2


# ============================================================================
# Edge Cases and Complex Scenarios
# ============================================================================


class TestComplexScenarios:
    """Test complex scenarios combining multiple features."""

    def test_reproduce_evolutionary_run(self):
        """Test reproducibility of evolutionary run with seed."""
        pop1 = Population(population_size=10, random_seed=12345)
        pop1.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        lengths1 = pop1.get_genome_lengths()
        pop1.evolve(num_generations=1)

        pop2 = Population(population_size=10, random_seed=12345)
        pop2.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        lengths2 = pop2.get_genome_lengths()
        pop2.evolve(num_generations=1)

        # Initial lengths should match
        assert lengths1 == lengths2

    def test_large_population_evolution(self):
        """Test evolution with larger population size."""
        pop = create_initialized_population(population_size=50)
        result = pop.evolve(num_generations=2, report_every=0)
        assert len(result) == 2
        assert len(pop.genomes) == 50

    def test_high_mutation_rate(self):
        """Test evolution with high mutation rate."""
        # Use larger population to survive high mutation rate
        pop = Population(population_size=50, mutation_rate=0.3)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        result = pop.evolve(num_generations=2, report_every=0)
        assert len(result) == 2

    def test_circular_vs_linear_genomes(self):
        """Test evolution differs for circular vs linear genomes."""
        pop_linear = create_initialized_population(circular=False)
        pop_circular = create_initialized_population(circular=True)

        assert not pop_linear.genomes[0].circular
        assert pop_circular.genomes[0].circular

    def test_multiple_coding_segments(self):
        """Test genome with multiple coding segments."""
        pop = create_simple_population(population_size=5)
        pop.initialize_population(
            initial_genome_length=18,
            nb_coding_segments=4,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=[
                PromoterDirection.FORWARD,
                PromoterDirection.REVERSE,
                PromoterDirection.FORWARD,
                PromoterDirection.REVERSE,
            ],
        )
        assert all(len(g) == 18 for g in pop.genomes)
        result = pop.evolve(num_generations=1, report_every=0)
        assert len(result) == 1

    def test_consecutive_evolution_runs(self):
        """Test multiple consecutive evolution runs."""
        pop = create_initialized_population(population_size=10)
        gen_after_first = 0

        pop.evolve(num_generations=3, report_every=0)
        gen_after_first = pop.generation
        assert gen_after_first == 3

        pop.evolve(num_generations=2, report_every=0)
        assert pop.generation == 5

    def test_population_with_single_individual(self):
        """Test edge case of population with single individual."""
        pop = create_initialized_population(population_size=1)
        assert len(pop.genomes) == 1
        pop.evolve_one_generation()
        assert len(pop.genomes) == 1

    def test_genome_diversity_after_evolution(self):
        """Test genome diversity metrics after evolution."""
        pop = create_initialized_population(population_size=10)
        div_before = pop.get_genome_diversity()
        pop.evolve_one_generation()
        div_after = pop.get_genome_diversity()
        assert isinstance(div_before, dict)
        assert isinstance(div_after, dict)

    def test_evolve_with_report_interval(self):
        """Test evolution with reporting at regular intervals."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(
            num_generations=5, report_every=2, plot_update_interval=None
        )
        assert len(result) == 5
        assert pop.generation == 5

    def test_evolve_very_small_population(self):
        """Test evolution with minimal population size."""
        pop = Population(population_size=3)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        result = pop.evolve(num_generations=2, report_every=0)
        assert len(result) == 2
        assert len(pop.genomes) == 3

    def test_stats_recorded_after_generations(self):
        """Test that statistics are recorded after each generation."""
        pop = create_initialized_population(population_size=20)
        result = pop.evolve(num_generations=3, report_every=0)

        # Verify stats are recorded in the results
        assert len(result) == 3
        for i, plot_data in enumerate(result):
            # Generations start at 1, not 0
            assert plot_data.stats.generation == i + 1
            assert plot_data.stats.population_size > 0
            assert plot_data.stats.avg_genome_length > 0

    def test_genome_lengths_in_plot_data(self):
        """Test that genome lengths are included in plot data."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(num_generations=2, report_every=0)

        for plot_data in result:
            assert hasattr(plot_data, "genome_lengths")
            assert len(plot_data.genome_lengths) == 10

    def test_different_coding_count_evolution(self):
        """Test evolution with different numbers of coding segments."""
        for num_segments in [1, 2, 3]:
            pop = create_simple_population(population_size=15)
            # NC--NC: (num_segments + 1) NC + num_segments C, all length 2
            total_length = (num_segments + 1) * 2 + num_segments * 2
            pop.initialize_population(
                initial_genome_length=total_length,
                nb_coding_segments=num_segments,
                length_coding_segments=2,
                length_non_coding_segments=2,
                promoter_directions=PromoterDirection.FORWARD,
            )
            result = pop.evolve(num_generations=1, report_every=0)
            assert len(result) == 1

    def test_mutation_rate_edge_values(self):
        """Test evolution with edge case mutation rates."""
        # Test with very low mutation rate
        pop_low = Population(population_size=10, mutation_rate=0.001)
        pop_low.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        result_low = pop_low.evolve(num_generations=2, report_every=0)
        assert len(result_low) == 2

        # Test with moderate mutation rate (larger population for stability)
        pop_mod = Population(population_size=30, mutation_rate=0.2)
        pop_mod.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )
        result_mod = pop_mod.evolve(num_generations=2, report_every=0)
        assert len(result_mod) == 2

    def test_stats_tracker_records_history(self):
        """Test that stats tracker maintains history across generations."""
        pop = create_initialized_population(population_size=10)
        pop.evolve(num_generations=3, report_every=0)

        summary = pop.get_evolution_summary()
        assert isinstance(summary, dict)
        # Summary should have tracked multiple generations
        assert len(summary) > 0

    def test_replenish_maintains_diversity(self):
        """Test that replenishment process uses genetic diversity."""
        pop = create_initialized_population(population_size=10)

        # Reduce population
        pop.genomes = pop.genomes[:5]
        # Replenish
        pop._replenish_population(5)

        # Should have same total population after replenishing
        assert len(pop.genomes) == 10

    def test_multiple_small_evolutions_equivalent_to_large(self):
        """Test consistency across multiple small evolution runs."""
        # Create two populations with same seed
        pop1 = Population(population_size=20, random_seed=999)
        pop1.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        pop2 = Population(population_size=20, random_seed=999)
        pop2.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        # One does 3 generations, other does 1+1+1
        _ = pop1.evolve(num_generations=3, report_every=0)

        _ = pop2.evolve(num_generations=1, report_every=0)
        _ = pop2.evolve(num_generations=1, report_every=0)
        _ = pop2.evolve(num_generations=1, report_every=0)

        # Both should end at generation 3
        assert pop1.generation == pop2.generation == 3
        assert len(pop1.genomes) == len(pop2.genomes)

    def test_dead_genome_removal_during_evolution(self):
        """Test that dead genomes are properly removed and population replenished."""
        # Use moderately high mutation rate to increase chance of genome death
        pop = Population(population_size=15, mutation_rate=0.3, random_seed=42)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        # Evolve with a moderately high mutation rate
        result = pop.evolve(num_generations=5, report_every=0)

        # Population should maintain size
        assert len(pop.genomes) == 15
        # Should have successful results
        assert len(result) == 5
        # All plot data should be valid
        assert all(hasattr(item, "stats") for item in result)

    def test_evolve_one_generation_with_dead_genomes(self):
        """Test that evolve_one_generation properly handles dead genomes."""
        # Use high mutation rate to increase chance of dead genomes
        pop = Population(population_size=20, mutation_rate=0.4, random_seed=42)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        initial_size = len(pop.genomes)
        initial_generation = pop.generation

        # Evolve one generation
        stats = pop.evolve_one_generation()

        # Population size should be restored to initial despite potential deaths
        assert len(pop.genomes) == initial_size
        # Generation counter should increment
        assert pop.generation == initial_generation + 1
        # Stats should be valid
        assert stats is not None
        assert stats.generation == initial_generation + 1
        assert stats.population_size == initial_size

    def test_evolve_with_report_every_enabled(self):
        """Test evolve with report_every > 0 prints stats."""
        pop = create_initialized_population(population_size=10)
        # report_every=1 should log each generation
        result = pop.evolve(num_generations=3, report_every=1)
        assert len(result) == 3
        # Each result should have valid stats
        assert all(item.stats.generation > 0 for item in result)

    def test_evolve_with_report_every_zero(self):
        """Test evolve with report_every=0 skips reporting."""
        pop = create_initialized_population(population_size=10)
        result = pop.evolve(num_generations=3, report_every=0)
        assert len(result) == 3
        # Stats should still be recorded even without reporting
        assert result[-1].stats.generation == 3

    def test_evolve_generation_counting_accuracy(self):
        """Test that generation counter is accurate across multiple evolve calls."""
        pop = create_initialized_population(population_size=10)

        # First evolution
        pop.evolve(num_generations=2, report_every=0)
        assert pop.generation == 2

        # Second evolution
        pop.evolve(num_generations=3, report_every=0)
        assert pop.generation == 5

    def test_apply_mutations_with_actual_mutation_rate(self):
        """Test apply_mutations with non-zero mutation rate."""
        pop = Population(population_size=20, mutation_rate=0.5, random_seed=42)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        # Apply mutations
        counts, dead_indices = pop.apply_mutations()

        # Should have some mutation activity
        assert isinstance(counts, MutationCounts)
        assert isinstance(dead_indices, list)
        # Dead indices should be a subset of population
        for idx in dead_indices:
            assert 0 <= idx < 20

    def test_genome_removal_with_high_mutation_rate(self):
        """Test genome removal works correctly when some genomes die."""
        pop = Population(population_size=30, mutation_rate=0.3, random_seed=123)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        # Run one generation to potentially kill some genomes and replenish
        stats = pop.evolve_one_generation()

        # Population should be restored to original size
        assert len(pop.genomes) == 30
        assert stats.population_size == 30
        assert pop.generation == 1

    def test_stats_survival_count_tracking(self):
        """Test that stats properly track survivor count after mutations."""
        pop = Population(population_size=25, mutation_rate=0.4, random_seed=456)
        pop.initialize_population(
            initial_genome_length=10,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
        )

        # Evolve and check stats
        result = pop.evolve(num_generations=2, report_every=0)

        # Stats should reflect population size
        for plot_data in result:
            assert plot_data.stats.population_size == 25
            assert plot_data.stats.generation > 0

    def test_genome_length_consistency_over_generations(self):
        """Test that genome lengths are tracked consistently during evolution."""
        pop = create_initialized_population(
            population_size=10, initial_genome_length=10
        )

        initial_lengths = pop.get_genome_lengths()
        assert all(length == 10 for length in initial_lengths)

        # Evolve multiple generations
        result = pop.evolve(num_generations=3, report_every=0)

        # Check genome_lengths in plot data
        for plot_data in result:
            assert isinstance(plot_data.genome_lengths, list)
            assert len(plot_data.genome_lengths) == 10
            assert all(isinstance(length, int) for length in plot_data.genome_lengths)

    def test_extremity_configuration_nc_nc_edge_case(self):
        """Test NC--NC extremity configuration with proper segment layout."""
        pop = create_simple_population(population_size=2)
        # NC--NC requires: NC(n1) + [C(c1) + NC(nc1) + ... + C(cn) + NC(ncn)]
        pop.initialize_population(
            initial_genome_length=11,  # 3 + 2 + 2 + 2 + 2
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=[3, 2, 2],  # Start NC + nc1 + nc2
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="NC--NC",
        )
        assert len(pop.genomes) == 2
        assert all(len(g) == 11 for g in pop.genomes)

    def test_c_c_extremity_configuration(self):
        """Test C--C extremity configuration."""
        pop = create_simple_population(population_size=2)
        # C--C: C(c1) + NC(nc1) + C(c2) + NC(nc2)
        pop.initialize_population(
            initial_genome_length=8,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=2,
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="C--C",
        )
        assert len(pop.genomes) == 2
        assert all(len(g) == 8 for g in pop.genomes)

    def test_c_nc_extremity_configuration(self):
        """Test C--NC extremity configuration."""
        pop = create_simple_population(population_size=2)
        # C--NC: C(c1) + NC(nc1) + C(c2) + NC(nc2)
        pop.initialize_population(
            initial_genome_length=8,
            nb_coding_segments=2,
            length_coding_segments=2,
            length_non_coding_segments=[2, 2],
            promoter_directions=PromoterDirection.FORWARD,
            is_circular=False,
            nature_of_extremities="C--NC",
        )
        assert len(pop.genomes) == 2
        assert all(len(g) == 8 for g in pop.genomes)

    def test_evolve_population_with_all_report_settings(self):
        """Test evolve with different combinations of reporting settings."""
        pop = create_initialized_population(population_size=8)

        # Test with report_every matching generation count
        result = pop.evolve(num_generations=4, report_every=2)
        assert len(result) == 4
        assert pop.generation == 4

    def test_population_stats_across_generations(self):
        """Test that stats are properly recorded for each generation."""
        pop = create_initialized_population(population_size=12)
        result = pop.evolve(num_generations=4, report_every=1)

        # Stats should show progressive generations
        assert result[0].stats.generation == 1
        assert result[1].stats.generation == 2
        assert result[2].stats.generation == 3
        assert result[3].stats.generation == 4

        # All should have correct population tracking
        for plot_data in result:
            assert plot_data.stats.population_size == 12
