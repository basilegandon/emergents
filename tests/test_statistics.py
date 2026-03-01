"""
Comprehensive tests for statistics module.

Tests cover statistics collection, analysis, and tracking functionality
with focus on edge cases and diverse scenarios.
"""

import statistics

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection
from emergents.statistics import (
    MutationCounts,
    PopulationStats,
    StatsCalculator,
    StatsTracker,
)

# ============================================================================
# Helper Functions
# ============================================================================


def make_genome(segments=None, circular=False):
    """
    Create a test genome.

    Default structure (if no segments provided):
    [NonCodingSegment(3) | CodingSegment(4) | NonCodingSegment(2)]
    Length: 9

    Args:
        segments: List of segments. If None, uses default structure.
        circular: Whether genome is circular.

    Returns:
        Genome instance.
    """
    if segments is None:
        segments = [
            NonCodingSegment(length=3),
            CodingSegment(length=4, promoter_direction=PromoterDirection.FORWARD),
            NonCodingSegment(length=2),
        ]
    return Genome(segments, circular=circular)


def make_population(sizes):
    """
    Create a population with genomes of specified lengths.

    Args:
        sizes: List of genome lengths.

    Returns:
        List of genomes.
    """
    population = []
    for size in sizes:
        noncoding_size = max(1, size // 3)
        coding_size = max(1, size - 2 * noncoding_size)
        genome = Genome(
            [
                NonCodingSegment(length=noncoding_size),
                CodingSegment(
                    length=coding_size, promoter_direction=PromoterDirection.FORWARD
                ),
                NonCodingSegment(length=noncoding_size),
            ]
        )
        population.append(genome)
    return population


# ============================================================================
# PopulationStats Tests
# ============================================================================


class TestPopulationStats:
    """Tests for PopulationStats dataclass."""

    def test_initialization_with_valid_data(self):
        """Test PopulationStats initialization with valid data."""
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )

        assert stats.generation == 1
        assert stats.population_size == 10
        assert stats.avg_genome_length == pytest.approx(50.0)
        assert stats.survivors_after_mutations == 8

    def test_post_init_calculates_survival_rate(self):
        """Test __post_init__ calculates mutation_survival_rate correctly."""
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )

        expected_rate = 10 / 20  # 0.5
        assert stats.mutation_survival_rate == expected_rate

    def test_survival_rate_zero_mutations(self):
        """Test mutation_survival_rate when no mutations applied."""
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=0,
            neutral_mutations=0,
            non_neutral_mutations=0,
            survivors_after_mutations=10,
        )

        assert stats.mutation_survival_rate == pytest.approx(0.0)

    def test_survival_rate_all_neutral(self):
        """Test mutation_survival_rate when all mutations are neutral."""
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=50,
            neutral_mutations=50,
            non_neutral_mutations=0,
            survivors_after_mutations=10,
        )

        assert stats.mutation_survival_rate == pytest.approx(1.0)

    def test_survival_rate_all_non_neutral(self):
        """Test mutation_survival_rate when no neutral mutations."""
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=50,
            neutral_mutations=0,
            non_neutral_mutations=50,
            survivors_after_mutations=10,
        )

        assert stats.mutation_survival_rate == pytest.approx(0.0)

    def test_str_representation(self):
        """Test __str__ method produces valid string."""
        stats = PopulationStats(
            generation=5,
            population_size=20,
            avg_genome_length=45.5,
            min_genome_length=30,
            max_genome_length=60,
            length_std=8.2,
            total_mutations_applied=100,
            neutral_mutations=60,
            non_neutral_mutations=40,
            survivors_after_mutations=18,
        )

        stats_str = str(stats)
        assert "Gen 5" in stats_str
        assert "Pop=20" in stats_str
        assert "45.5" in stats_str
        assert "100" in stats_str
        assert "Survivors: 18" in stats_str

    def test_to_dict_conversion(self):
        """Test to_dict() method returns correct dictionary."""
        stats = PopulationStats(
            generation=3,
            population_size=15,
            avg_genome_length=42.0,
            min_genome_length=35,
            max_genome_length=50,
            length_std=4.5,
            total_mutations_applied=75,
            neutral_mutations=45,
            non_neutral_mutations=30,
            survivors_after_mutations=12,
            length_diversity=0.8,
        )

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["generation"] == 3
        assert stats_dict["population_size"] == 15
        assert stats_dict["avg_genome_length"] == pytest.approx(42.0)
        assert stats_dict["min_genome_length"] == 35
        assert stats_dict["max_genome_length"] == 50
        assert stats_dict["length_std"] == pytest.approx(4.5)
        assert stats_dict["total_mutations_applied"] == 75
        assert stats_dict["neutral_mutations"] == 45
        assert stats_dict["non_neutral_mutations"] == 30
        assert stats_dict["survivors_after_mutations"] == 12
        assert stats_dict["length_diversity"] == pytest.approx(0.8)
        assert stats_dict["mutation_survival_rate"] == pytest.approx(45 / 75)

    def test_edge_case_zero_survivors(self):
        """Test PopulationStats with zero survivors."""
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=5,
            non_neutral_mutations=15,
            survivors_after_mutations=0,
        )

        assert stats.survivors_after_mutations == 0
        assert stats.mutation_survival_rate == 5 / 20

    def test_edge_case_single_genome(self):
        """Test PopulationStats with population of one."""
        stats = PopulationStats(
            generation=1,
            population_size=1,
            avg_genome_length=50.0,
            min_genome_length=50,
            max_genome_length=50,
            length_std=0.0,
            total_mutations_applied=1,
            neutral_mutations=1,
            non_neutral_mutations=0,
            survivors_after_mutations=1,
        )

        assert stats.population_size == 1
        assert stats.length_std == pytest.approx(0.0)
        assert stats.mutation_survival_rate == pytest.approx(1.0)


# ============================================================================
# MutationCounts Tests
# ============================================================================


class TestMutationCounts:
    """Tests for MutationCounts dataclass."""

    def test_initialization_defaults(self):
        """Test MutationCounts initializes with zeros."""
        counts = MutationCounts()

        assert counts.total == 0
        assert counts.neutral == 0
        assert counts.non_neutral == 0
        assert counts.failed == 0

    def test_initialization_with_values(self):
        """Test MutationCounts initialization with custom values."""
        counts = MutationCounts(total=100, neutral=60, non_neutral=30, failed=10)

        assert counts.total == 100
        assert counts.neutral == 60
        assert counts.non_neutral == 30
        assert counts.failed == 10

    def test_reset_to_zeros(self):
        """Test reset() method clears all counters."""
        counts = MutationCounts(total=100, neutral=60, non_neutral=30, failed=10)
        counts.reset()

        assert counts.total == 0
        assert counts.neutral == 0
        assert counts.non_neutral == 0
        assert counts.failed == 0

    def test_to_dict_conversion(self):
        """Test to_dict() method."""
        counts = MutationCounts(total=50, neutral=30, non_neutral=15, failed=5)

        counts_dict = counts.to_dict()

        assert isinstance(counts_dict, dict)
        assert counts_dict["total"] == 50
        assert counts_dict["neutral"] == 30
        assert counts_dict["non_neutral"] == 15
        assert counts_dict["failed"] == 5

    def test_to_dict_after_reset(self):
        """Test to_dict() after reset."""
        counts = MutationCounts(total=100, neutral=60, non_neutral=30, failed=10)
        counts.reset()

        counts_dict = counts.to_dict()

        assert all(v == 0 for v in counts_dict.values())


# ============================================================================
# StatsCalculator Tests
# ============================================================================


class TestStatsCalculatorPopulationStats:
    """Tests for StatsCalculator.calculate_population_stats()."""

    def test_empty_population(self):
        """Test with empty population."""
        stats = StatsCalculator.calculate_population_stats(
            genomes=[],
            generation=1,
            mutation_counts=MutationCounts(total=0, neutral=0),
            survivors_count=0,
        )

        assert stats.generation == 1
        assert stats.population_size == 0
        assert stats.avg_genome_length == pytest.approx(0.0)
        assert stats.min_genome_length == 0
        assert stats.max_genome_length == 0
        assert stats.length_std == pytest.approx(0.0)
        assert stats.total_mutations_applied == 0

    def test_single_genome(self):
        """Test with single genome in population."""
        genome = make_genome()
        stats = StatsCalculator.calculate_population_stats(
            genomes=[genome],
            generation=1,
            mutation_counts=MutationCounts(total=5, neutral=3, non_neutral=2),
            survivors_count=1,
        )

        assert stats.population_size == 1
        assert stats.avg_genome_length == len(genome)
        assert stats.min_genome_length == len(genome)
        assert stats.max_genome_length == len(genome)
        assert stats.length_std == pytest.approx(0.0)
        assert stats.total_mutations_applied == 5
        assert stats.survivors_after_mutations == 1

    def test_multiple_genomes_uniform_length(self):
        """Test with multiple genomes of same length."""
        genomes = make_population([10, 10, 10])
        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=5,
            mutation_counts=MutationCounts(total=30, neutral=20, non_neutral=10),
            survivors_count=3,
        )

        assert stats.population_size == 3
        assert stats.avg_genome_length == pytest.approx(10.0)
        assert stats.min_genome_length == 10
        assert stats.max_genome_length == 10
        assert stats.length_std == pytest.approx(0.0)
        assert stats.length_diversity == pytest.approx(
            1.0 / 3
        )  # One unique length out of 3

    def test_multiple_genomes_varied_length(self):
        """Test with multiple genomes of different lengths."""
        genomes = make_population([8, 10, 12, 15])
        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=2,
            mutation_counts=MutationCounts(total=50, neutral=25, non_neutral=25),
            survivors_count=3,
        )

        assert stats.population_size == 4
        lengths = [len(g) for g in genomes]
        expected_mean = statistics.mean(lengths)
        expected_std = statistics.stdev(lengths)

        assert stats.avg_genome_length == expected_mean
        assert stats.length_std == expected_std
        assert stats.min_genome_length == min(lengths)
        assert stats.max_genome_length == max(lengths)
        assert stats.length_diversity == pytest.approx(1.0)  # All different lengths

    def test_large_population(self):
        """Test with large population."""
        genomes = make_population(list(range(50, 100)))
        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=100,
            mutation_counts=MutationCounts(total=500, neutral=250, non_neutral=250),
            survivors_count=49,
        )

        assert stats.population_size == 50
        assert 50 <= stats.min_genome_length < stats.max_genome_length < 100
        assert stats.survivors_after_mutations == 49

    def test_mutation_counts_preserved(self):
        """Test that mutation counts are correctly recorded."""
        genomes = make_population([10, 10])
        mutation_counts = MutationCounts(total=75, neutral=40, non_neutral=35)

        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=1,
            mutation_counts=mutation_counts,
            survivors_count=2,
        )

        assert stats.total_mutations_applied == mutation_counts.total
        assert stats.neutral_mutations == mutation_counts.neutral
        assert stats.non_neutral_mutations == mutation_counts.non_neutral
        assert stats.mutation_survival_rate == 40 / 75

    def test_length_diversity_calculation(self):
        """Test length_diversity metric is calculated correctly."""
        # 3 unique lengths out of 5 genomes
        genomes = make_population([10, 10, 12, 12, 15])
        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=1,
            mutation_counts=MutationCounts(),
            survivors_count=5,
        )

        assert stats.length_diversity == pytest.approx(3.0 / 5)  # 3 unique out of 5

    def test_standard_deviation_single_value(self):
        """Test stdev with population of one (edge case)."""
        genomes = make_population([10])
        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=1,
            mutation_counts=MutationCounts(),
            survivors_count=1,
        )

        assert stats.length_std == pytest.approx(0.0)


class TestStatsCalculatorDiversityMetrics:
    """Tests for StatsCalculator.calculate_diversity_metrics()."""

    def test_empty_population_metrics(self):
        """Test diversity metrics with empty population."""
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=[])

        expected = {
            "length_diversity": 0.0,
            "length_std": 0.0,
            "length_coefficient_variation": 0.0,
            "unique_length_count": 0,
        }

        assert metrics == expected

    def test_single_genome_metrics(self):
        """Test metrics with single genome."""
        genome = make_genome()
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=[genome])

        assert metrics["length_diversity"] == pytest.approx(1.0)  # 1 unique out of 1
        assert metrics["length_std"] == pytest.approx(0.0)
        assert metrics["length_coefficient_variation"] == pytest.approx(0.0)
        assert metrics["unique_length_count"] == 1

    def test_uniform_population_metrics(self):
        """Test metrics with all genomes same length."""
        genomes = make_population([20, 20, 20, 20])
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=genomes)

        assert metrics["length_diversity"] == pytest.approx(0.25)  # 1 unique out of 4
        assert metrics["length_std"] == pytest.approx(0.0)
        assert metrics["length_coefficient_variation"] == pytest.approx(0.0)
        assert metrics["unique_length_count"] == 1

    def test_varied_population_metrics(self):
        """Test metrics with varied population."""
        genomes = make_population([10, 15, 20, 25])
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=genomes)

        lengths = [len(g) for g in genomes]
        expected_std = statistics.stdev(lengths)
        expected_mean = statistics.mean(lengths)
        expected_cv = expected_std / expected_mean

        assert metrics["length_diversity"] == pytest.approx(1.0)  # 4 unique out of 4
        assert metrics["length_std"] == pytest.approx(expected_std)
        assert metrics["length_coefficient_variation"] == pytest.approx(expected_cv)
        assert metrics["unique_length_count"] == 4

    def test_partial_diversity(self):
        """Test metrics with partial diversity."""
        genomes = make_population([10, 10, 15, 15, 15])
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=genomes)

        assert metrics["unique_length_count"] == 2
        assert metrics["length_diversity"] == pytest.approx(2.0 / 5)

    def test_coefficient_variation_calculation(self):
        """Test coefficient of variation is calculated correctly."""
        genomes = make_population([10, 20, 30])
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=genomes)

        expected_cv = statistics.stdev([10, 20, 30]) / statistics.mean([10, 20, 30])
        assert abs(metrics["length_coefficient_variation"] - expected_cv) < 1e-9

    def test_zero_mean_coefficient_variation(self):
        """Test coefficient variation when all genomes have zero length."""
        # This is theoretical - genomes cannot be zero length, but test the logic
        # (would need to mock, but for now we test the edge case logic in metrics)
        genomes = make_population(
            [10]
        )  # Single genome -> std=0, cv should not divide by zero
        metrics = StatsCalculator.calculate_diversity_metrics(genomes=genomes)

        assert metrics["length_coefficient_variation"] == pytest.approx(0.0)


# ============================================================================
# StatsTracker Tests
# ============================================================================


class TestStatsTracker:
    """Tests for StatsTracker functionality."""

    def test_initialization(self):
        """Test StatsTracker initializes correctly."""
        tracker = StatsTracker()

        assert tracker.history == []
        assert isinstance(tracker.current_mutation_counts, MutationCounts)
        assert tracker.current_mutation_counts.total == 0

    def test_record_single_generation(self):
        """Test recording stats for a single generation."""
        tracker = StatsTracker()
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )

        tracker.record_generation(stats)

        assert len(tracker.history) == 1
        assert tracker.history[0] == stats

    def test_record_multiple_generations(self):
        """Test recording multiple generations."""
        tracker = StatsTracker()

        for gen in range(1, 6):
            stats = PopulationStats(
                generation=gen,
                population_size=10 + gen,
                avg_genome_length=50.0 + gen,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20 + gen * 5,
                neutral_mutations=10 + gen * 2,
                non_neutral_mutations=10 + gen * 3,
                survivors_after_mutations=8,
            )
            tracker.record_generation(stats)

        assert len(tracker.history) == 5
        assert tracker.history[0].generation == 1
        assert tracker.history[4].generation == 5

    def test_get_latest_stats_empty(self):
        """Test get_latest_stats with no recorded generations."""
        tracker = StatsTracker()

        assert tracker.get_latest_stats() is None

    def test_get_latest_stats_with_data(self):
        """Test get_latest_stats returns most recent."""
        tracker = StatsTracker()

        stats_list = []
        for gen in range(1, 4):
            stats = PopulationStats(
                generation=gen,
                population_size=10,
                avg_genome_length=50.0,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20,
                neutral_mutations=10,
                non_neutral_mutations=10,
                survivors_after_mutations=8,
            )
            stats_list.append(stats)
            tracker.record_generation(stats)

        latest = tracker.get_latest_stats()
        assert latest == stats_list[-1]
        assert latest.generation == 3

    def test_get_generation_stats_exists(self):
        """Test get_generation_stats for existing generation."""
        tracker = StatsTracker()

        for gen in range(1, 5):
            stats = PopulationStats(
                generation=gen,
                population_size=10 + gen,
                avg_genome_length=50.0,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20,
                neutral_mutations=10,
                non_neutral_mutations=10,
                survivors_after_mutations=8,
            )
            tracker.record_generation(stats)

        stats_gen_2 = tracker.get_generation_stats(2)
        assert stats_gen_2 is not None
        assert stats_gen_2.generation == 2
        assert stats_gen_2.population_size == 12

    def test_get_generation_stats_not_exists(self):
        """Test get_generation_stats for non-existent generation."""
        tracker = StatsTracker()
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )
        tracker.record_generation(stats)

        assert tracker.get_generation_stats(999) is None

    def test_get_generation_stats_empty_tracker(self):
        """Test get_generation_stats with empty tracker."""
        tracker = StatsTracker()

        assert tracker.get_generation_stats(1) is None

    def test_get_stats_range(self):
        """Test get_stats_range returns correct subset."""
        tracker = StatsTracker()

        for gen in range(1, 11):
            stats = PopulationStats(
                generation=gen,
                population_size=10,
                avg_genome_length=50.0,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20,
                neutral_mutations=10,
                non_neutral_mutations=10,
                survivors_after_mutations=8,
            )
            tracker.record_generation(stats)

        range_stats = tracker.get_stats_range(3, 7)

        assert len(range_stats) == 5
        assert range_stats[0].generation == 3
        assert range_stats[-1].generation == 7

    def test_get_stats_range_empty(self):
        """Test get_stats_range with no data in range."""
        tracker = StatsTracker()

        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )
        tracker.record_generation(stats)

        range_stats = tracker.get_stats_range(10, 20)
        assert len(range_stats) == 0

    def test_get_stats_range_boundaries(self):
        """Test get_stats_range includes boundary values."""
        tracker = StatsTracker()

        for gen in [1, 5, 10, 15]:
            stats = PopulationStats(
                generation=gen,
                population_size=10,
                avg_genome_length=50.0,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20,
                neutral_mutations=10,
                non_neutral_mutations=10,
                survivors_after_mutations=8,
            )
            tracker.record_generation(stats)

        range_stats = tracker.get_stats_range(5, 15)
        assert len(range_stats) == 3
        assert [s.generation for s in range_stats] == [5, 10, 15]

    def test_reset_mutation_counts(self):
        """Test reset_mutation_counts clears counters."""
        tracker = StatsTracker()
        tracker.current_mutation_counts.total = 100
        tracker.current_mutation_counts.neutral = 50

        tracker.reset_mutation_counts()

        assert tracker.current_mutation_counts.total == 0
        assert tracker.current_mutation_counts.neutral == 0

    def test_get_summary_metrics_empty(self):
        """Test get_summary_metrics with empty history."""
        tracker = StatsTracker()

        summary = tracker.get_summary_metrics()

        assert summary == {}

    def test_get_summary_metrics_single_generation(self):
        """Test get_summary_metrics with single generation."""
        tracker = StatsTracker()
        stats = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )
        tracker.record_generation(stats)

        summary = tracker.get_summary_metrics()

        assert summary["total_generations"] == 1
        assert summary["final_avg_length"] == pytest.approx(50.0)
        assert summary["initial_avg_length"] == pytest.approx(50.0)
        assert summary["length_change"] == pytest.approx(0.0)
        assert summary["max_population_size"] == 10
        assert summary["min_population_size"] == 10
        assert summary["total_mutations"] == 20
        assert summary["avg_survival_rate"] == 10 / 20

    def test_get_summary_metrics_multiple_generations(self):
        """Test get_summary_metrics across multiple generations."""
        tracker = StatsTracker()

        for gen in range(1, 6):
            stats = PopulationStats(
                generation=gen,
                population_size=10 + gen,
                avg_genome_length=50.0 + gen,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20 + gen * 5,
                neutral_mutations=10 + gen * 2,
                non_neutral_mutations=10 + gen * 3,
                survivors_after_mutations=8,
            )
            tracker.record_generation(stats)

        summary = tracker.get_summary_metrics()

        assert summary["total_generations"] == 5
        assert summary["final_avg_length"] == pytest.approx(55.0)  # 50 + 5
        assert summary["initial_avg_length"] == pytest.approx(51.0)  # 50 + 1
        assert summary["length_change"] == pytest.approx(4.0)  # 55 - 51
        assert summary["max_population_size"] == 15
        assert summary["min_population_size"] == 11
        assert summary["total_mutations"] == 25 + 30 + 35 + 40 + 45  # 175

    def test_get_summary_metrics_population_size_range(self):
        """Test get_summary_metrics correctly tracks population size extremes."""
        tracker = StatsTracker()

        population_sizes = [5, 15, 10, 20, 8]
        for gen, pop_size in enumerate(population_sizes, 1):
            stats = PopulationStats(
                generation=gen,
                population_size=pop_size,
                avg_genome_length=50.0,
                min_genome_length=40,
                max_genome_length=60,
                length_std=5.0,
                total_mutations_applied=20,
                neutral_mutations=10,
                non_neutral_mutations=10,
                survivors_after_mutations=pop_size - 1,
            )
            tracker.record_generation(stats)

        summary = tracker.get_summary_metrics()

        assert summary["max_population_size"] == max(population_sizes)
        assert summary["min_population_size"] == min(population_sizes)

    def test_get_summary_metrics_zero_mutations_skipped(self):
        """Test avg_survival_rate skips generations with zero mutations."""
        tracker = StatsTracker()

        # Gen 1: 20 mutations, 50% survival
        stats1 = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=50.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )

        # Gen 2: 0 mutations (should be skipped)
        stats2 = PopulationStats(
            generation=2,
            population_size=10,
            avg_genome_length=51.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=0,
            neutral_mutations=0,
            non_neutral_mutations=0,
            survivors_after_mutations=10,
        )

        # Gen 3: 30 mutations, 60% survival
        stats3 = PopulationStats(
            generation=3,
            population_size=10,
            avg_genome_length=52.0,
            min_genome_length=40,
            max_genome_length=60,
            length_std=5.0,
            total_mutations_applied=30,
            neutral_mutations=18,
            non_neutral_mutations=12,
            survivors_after_mutations=8,
        )

        tracker.record_generation(stats1)
        tracker.record_generation(stats2)
        tracker.record_generation(stats3)

        summary = tracker.get_summary_metrics()

        # Average should be (0.5 + 0.6) / 2 = 0.55
        expected_avg = (10 / 20 + 18 / 30) / 2
        assert abs(summary["avg_survival_rate"] - expected_avg) < 1e-9

    def test_get_summary_metrics_length_change_negative(self):
        """Test length_change when average length decreases."""
        tracker = StatsTracker()

        stats1 = PopulationStats(
            generation=1,
            population_size=10,
            avg_genome_length=100.0,
            min_genome_length=90,
            max_genome_length=110,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )

        stats2 = PopulationStats(
            generation=2,
            population_size=10,
            avg_genome_length=75.0,
            min_genome_length=65,
            max_genome_length=85,
            length_std=5.0,
            total_mutations_applied=20,
            neutral_mutations=10,
            non_neutral_mutations=10,
            survivors_after_mutations=8,
        )

        tracker.record_generation(stats1)
        tracker.record_generation(stats2)

        summary = tracker.get_summary_metrics()

        assert summary["length_change"] == -25.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestStatsIntegration:
    """Integration tests combining multiple components."""

    def test_full_simulation_workflow(self):
        """Test complete workflow from calculation to tracking."""
        tracker = StatsTracker()

        for gen in range(1, 6):
            # Simulate population at this generation
            genomes = make_population([40 + gen * 2, 50 + gen, 60 - gen])

            # Record mutation application
            mutation_counts = MutationCounts(
                total=30 + gen * 5,
                neutral=15 + gen * 3,
                non_neutral=15 + gen * 2,
            )

            # Calculate stats
            stats = StatsCalculator.calculate_population_stats(
                genomes=genomes,
                generation=gen,
                mutation_counts=mutation_counts,
                survivors_count=3,
            )

            tracker.record_generation(stats)

        # Verify tracking
        assert len(tracker.history) == 5
        summary = tracker.get_summary_metrics()
        assert summary["total_generations"] == 5
        assert summary["total_mutations"] == sum(30 + gen * 5 for gen in range(1, 6))

    def test_calculator_and_tracker_consistency(self):
        """Test that calculator output integrates correctly with tracker."""
        genomes = make_population([10, 15, 20])
        mutation_counts = MutationCounts(total=50, neutral=30, non_neutral=20)
        survivors = 3

        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=1,
            mutation_counts=mutation_counts,
            survivors_count=survivors,
        )

        tracker = StatsTracker()
        tracker.record_generation(stats)

        # Verify retrieval
        retrieved_stats = tracker.get_latest_stats()
        assert retrieved_stats.total_mutations_applied == mutation_counts.total
        assert retrieved_stats.survivors_after_mutations == survivors

    def test_diversity_metrics_with_population_stats(self):
        """Test that diversity metrics align with PopulationStats."""
        genomes = make_population([10, 10, 15, 15, 15])

        # Calculate both metrics
        diversity = StatsCalculator.calculate_diversity_metrics(genomes=genomes)
        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=1,
            mutation_counts=MutationCounts(),
            survivors_count=5,
        )

        # Verify consistency
        assert stats.length_diversity == diversity["length_diversity"]
        assert stats.length_std == diversity["length_std"]
