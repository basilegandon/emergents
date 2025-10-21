"""
Comprehensive unit tests for emergents.statistics module.
Tests PopulationStats, MutationCounts, StatsCalculator, and StatsTracker classes.
"""

from unittest.mock import Mock

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, Segment
from emergents.statistics import (
    MutationCounts,
    PopulationStats,
    StatsCalculator,
    StatsTracker,
)


class TestPopulationStats:
    """Test PopulationStats dataclass."""

    def test_basic_initialization(self) -> None:
        """Test basic PopulationStats creation."""
        stats = PopulationStats(
            generation=5,
            population_size=1000,
            avg_genome_length=10000.0,
            min_genome_length=9500,
            max_genome_length=10500,
            length_std=150.0,
            total_mutations_applied=100,
            neutral_mutations=85,
            non_neutral_mutations=15,
            survivors_after_mutations=985,
        )

        assert stats.generation == 5
        assert stats.population_size == 1000
        assert stats.avg_genome_length == 10000.0
        assert stats.min_genome_length == 9500
        assert stats.max_genome_length == 10500
        assert stats.length_std == 150.0
        assert stats.total_mutations_applied == 100
        assert stats.neutral_mutations == 85
        assert stats.non_neutral_mutations == 15
        assert stats.survivors_after_mutations == 985

    def test_post_init_calculation(self) -> None:
        """Test that __post_init__ calculates derived metrics correctly."""
        stats = PopulationStats(
            generation=1,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=900,
            max_genome_length=1100,
            length_std=50.0,
            total_mutations_applied=100,
            neutral_mutations=80,
            non_neutral_mutations=20,
            survivors_after_mutations=95,
        )

        # Should calculate mutation survival rate
        expected_rate = 80 / 100  # neutral / total
        assert stats.mutation_survival_rate == expected_rate

    def test_post_init_with_zero_mutations(self) -> None:
        """Test __post_init__ when no mutations were applied."""
        stats = PopulationStats(
            generation=0,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=1000,
            max_genome_length=1000,
            length_std=0.0,
            total_mutations_applied=0,
            neutral_mutations=0,
            non_neutral_mutations=0,
            survivors_after_mutations=100,
        )

        # Should not divide by zero
        assert stats.mutation_survival_rate == 0.0

    def test_str_representation(self) -> None:
        """Test string representation of PopulationStats."""
        stats = PopulationStats(
            generation=10,
            population_size=500,
            avg_genome_length=1500.7,
            min_genome_length=1400,
            max_genome_length=1600,
            length_std=75.3,
            total_mutations_applied=50,
            neutral_mutations=40,
            non_neutral_mutations=10,
            survivors_after_mutations=490,
        )

        str_repr = str(stats)

        # Check that key information is present
        assert "Gen 10" in str_repr
        assert "Pop=500" in str_repr
        assert "AvgLen=1500.7±75.3" in str_repr
        assert "Mutations=50" in str_repr
        assert "Survivors: 490" in str_repr
        assert "80.0%" in str_repr  # Survival rate

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = PopulationStats(
            generation=2,
            population_size=200,
            avg_genome_length=2000.0,
            min_genome_length=1800,
            max_genome_length=2200,
            length_std=100.0,
            total_mutations_applied=25,
            neutral_mutations=20,
            non_neutral_mutations=5,
            survivors_after_mutations=195,
            length_diversity=0.75,
        )

        result_dict = stats.to_dict()

        expected_keys = {
            "generation",
            "population_size",
            "avg_genome_length",
            "min_genome_length",
            "max_genome_length",
            "length_std",
            "total_mutations_applied",
            "neutral_mutations",
            "non_neutral_mutations",
            "survivors_after_mutations",
            "length_diversity",
            "mutation_survival_rate",
        }

        assert set(result_dict.keys()) == expected_keys
        assert result_dict["generation"] == 2
        assert result_dict["population_size"] == 200
        assert result_dict["avg_genome_length"] == 2000.0
        assert result_dict["length_diversity"] == 0.75


class TestMutationCounts:
    """Test MutationCounts dataclass."""

    def test_default_initialization(self) -> None:
        """Test default MutationCounts creation."""
        counts = MutationCounts()

        assert counts.total == 0
        assert counts.neutral == 0
        assert counts.non_neutral == 0
        assert counts.failed == 0

    def test_custom_initialization(self) -> None:
        """Test MutationCounts with custom values."""
        counts = MutationCounts(total=100, neutral=75, non_neutral=20, failed=5)

        assert counts.total == 100
        assert counts.neutral == 75
        assert counts.non_neutral == 20
        assert counts.failed == 5

    def test_reset(self) -> None:
        """Test reset method."""
        counts = MutationCounts(total=50, neutral=40, non_neutral=8, failed=2)

        counts.reset()

        assert counts.total == 0
        assert counts.neutral == 0
        assert counts.non_neutral == 0
        assert counts.failed == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        counts = MutationCounts(total=30, neutral=25, non_neutral=4, failed=1)

        result_dict = counts.to_dict()

        expected_dict = {"total": 30, "neutral": 25, "non_neutral": 4, "failed": 1}

        assert result_dict == expected_dict


class TestStatsCalculator:
    """Test StatsCalculator class."""

    def create_test_genomes(self, lengths: list[int]) -> list[Genome]:
        """Helper to create test genomes with specified lengths."""
        genomes = []
        for length in lengths:
            if length <= 0:
                genomes.append(Genome())
            else:
                segments: list[Segment] = [NonCodingSegment(length=length)]
                genomes.append(Genome(segments=segments))
        return genomes

    def test_calculate_population_stats_empty(self) -> None:
        """Test calculating stats for empty population."""
        mutation_counts = MutationCounts(total=5, neutral=3, non_neutral=2)

        stats = StatsCalculator.calculate_population_stats(
            genomes=[], generation=1, mutation_counts=mutation_counts, survivors_count=0
        )

        assert stats.generation == 1
        assert stats.population_size == 0
        assert stats.avg_genome_length == 0.0
        assert stats.min_genome_length == 0
        assert stats.max_genome_length == 0
        assert stats.length_std == 0.0
        assert stats.total_mutations_applied == 5
        assert stats.neutral_mutations == 3
        assert stats.non_neutral_mutations == 2
        assert stats.survivors_after_mutations == 0

    def test_calculate_population_stats_single_genome(self) -> None:
        """Test calculating stats for single genome."""
        genomes = self.create_test_genomes([1000])
        mutation_counts = MutationCounts(total=10, neutral=8, non_neutral=2)

        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=5,
            mutation_counts=mutation_counts,
            survivors_count=1,
        )

        assert stats.generation == 5
        assert stats.population_size == 1
        assert stats.avg_genome_length == 1000.0
        assert stats.min_genome_length == 1000
        assert stats.max_genome_length == 1000
        assert stats.length_std == 0.0  # Single genome, no deviation
        assert stats.length_diversity == 1.0  # One unique length out of one genome

    def test_calculate_population_stats_multiple_genomes(self) -> None:
        """Test calculating stats for multiple genomes."""
        genomes = self.create_test_genomes([900, 1000, 1100, 950, 1050])
        mutation_counts = MutationCounts(total=50, neutral=40, non_neutral=10)

        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=10,
            mutation_counts=mutation_counts,
            survivors_count=5,
        )

        assert stats.generation == 10
        assert stats.population_size == 5
        assert stats.avg_genome_length == 1000.0  # (900+1000+1100+950+1050)/5
        assert stats.min_genome_length == 900
        assert stats.max_genome_length == 1100
        assert stats.length_std > 0  # Should have some deviation
        assert stats.length_diversity == 1.0  # All lengths are unique (5/5)

    def test_calculate_population_stats_identical_genomes(self) -> None:
        """Test calculating stats for genomes with identical lengths."""
        genomes = self.create_test_genomes([1000, 1000, 1000])
        mutation_counts = MutationCounts(total=15, neutral=12, non_neutral=3)

        stats = StatsCalculator.calculate_population_stats(
            genomes=genomes,
            generation=3,
            mutation_counts=mutation_counts,
            survivors_count=3,
        )

        assert stats.population_size == 3
        assert stats.avg_genome_length == 1000.0
        assert stats.min_genome_length == 1000
        assert stats.max_genome_length == 1000
        assert stats.length_std == 0.0  # Identical lengths
        assert stats.length_diversity == 1.0 / 3.0  # One unique length out of three

    def test_calculate_diversity_metrics_empty(self) -> None:
        """Test diversity calculation for empty genome list."""
        result = StatsCalculator.calculate_diversity_metrics([])

        expected = {
            "length_diversity": 0.0,
            "length_std": 0.0,
            "length_coefficient_variation": 0.0,
            "unique_length_count": 0,
        }

        assert result == expected

    def test_calculate_diversity_metrics_single_genome(self) -> None:
        """Test diversity calculation for single genome."""
        genomes = self.create_test_genomes([1000])

        result = StatsCalculator.calculate_diversity_metrics(genomes)

        assert result["length_diversity"] == 1.0
        assert result["length_std"] == 0.0
        assert result["length_coefficient_variation"] == 0.0
        assert result["unique_length_count"] == 1

    def test_calculate_diversity_metrics_multiple_genomes(self) -> None:
        """Test diversity calculation for multiple genomes."""
        genomes = self.create_test_genomes([800, 1000, 1200, 1000, 800])

        result = StatsCalculator.calculate_diversity_metrics(genomes)

        assert result["length_diversity"] == 3.0 / 5.0  # 3 unique lengths out of 5
        assert result["length_std"] > 0
        assert result["length_coefficient_variation"] > 0
        assert result["unique_length_count"] == 3

    def test_calculate_diversity_metrics_zero_mean_length(self) -> None:
        """Test diversity calculation when mean length is zero."""
        genomes = self.create_test_genomes([0, 0, 0])

        result = StatsCalculator.calculate_diversity_metrics(genomes)

        assert result["length_diversity"] == 1.0 / 3.0  # One unique length
        assert result["length_std"] == 0.0
        assert (
            result["length_coefficient_variation"] == 0.0
        )  # Should handle division by zero


class TestStatsTracker:
    """Test StatsTracker class."""

    def create_test_stats(
        self, generation: int, avg_length: float = 1000.0
    ) -> PopulationStats:
        """Helper to create test PopulationStats."""
        return PopulationStats(
            generation=generation,
            population_size=100,
            avg_genome_length=avg_length,
            min_genome_length=int(avg_length - 100),
            max_genome_length=int(avg_length + 100),
            length_std=50.0,
            total_mutations_applied=10,
            neutral_mutations=8,
            non_neutral_mutations=2,
            survivors_after_mutations=98,
        )

    def test_initialization(self) -> None:
        """Test StatsTracker initialization."""
        tracker = StatsTracker()

        assert tracker.history == []
        assert isinstance(tracker.current_mutation_counts, MutationCounts)
        assert tracker.current_mutation_counts.total == 0

    def test_record_generation(self) -> None:
        """Test recording generation statistics."""
        tracker = StatsTracker()
        stats1 = self.create_test_stats(1)
        stats2 = self.create_test_stats(2)

        tracker.record_generation(stats1)
        tracker.record_generation(stats2)

        assert len(tracker.history) == 2
        assert tracker.history[0] == stats1
        assert tracker.history[1] == stats2

    def test_get_latest_stats_empty(self) -> None:
        """Test getting latest stats when no history exists."""
        tracker = StatsTracker()

        result = tracker.get_latest_stats()

        assert result is None

    def test_get_latest_stats_with_history(self) -> None:
        """Test getting latest stats with history."""
        tracker = StatsTracker()
        stats1 = self.create_test_stats(1)
        stats2 = self.create_test_stats(2)

        tracker.record_generation(stats1)
        tracker.record_generation(stats2)

        latest = tracker.get_latest_stats()

        assert latest == stats2

    def test_get_generation_stats_found(self) -> None:
        """Test getting stats for specific generation that exists."""
        tracker = StatsTracker()
        stats1 = self.create_test_stats(1)
        stats3 = self.create_test_stats(3)
        stats5 = self.create_test_stats(5)

        tracker.record_generation(stats1)
        tracker.record_generation(stats3)
        tracker.record_generation(stats5)

        result = tracker.get_generation_stats(3)

        assert result == stats3

    def test_get_generation_stats_not_found(self) -> None:
        """Test getting stats for generation that doesn't exist."""
        tracker = StatsTracker()
        stats1 = self.create_test_stats(1)

        tracker.record_generation(stats1)

        result = tracker.get_generation_stats(5)

        assert result is None

    def test_get_stats_range(self) -> None:
        """Test getting stats for a range of generations."""
        tracker = StatsTracker()
        stats_list = [self.create_test_stats(i) for i in range(1, 11)]

        for stats in stats_list:
            tracker.record_generation(stats)

        # Get stats for generations 3-7
        result = tracker.get_stats_range(3, 7)

        assert len(result) == 5
        assert result[0].generation == 3
        assert result[-1].generation == 7

    def test_get_stats_range_empty(self) -> None:
        """Test getting stats range when no matching generations exist."""
        tracker = StatsTracker()
        stats1 = self.create_test_stats(1)

        tracker.record_generation(stats1)

        result = tracker.get_stats_range(5, 10)

        assert result == []

    def test_reset_mutation_counts(self) -> None:
        """Test resetting mutation counts."""
        tracker = StatsTracker()

        # Modify mutation counts
        tracker.current_mutation_counts.total = 50
        tracker.current_mutation_counts.neutral = 40

        tracker.reset_mutation_counts()

        assert tracker.current_mutation_counts.total == 0
        assert tracker.current_mutation_counts.neutral == 0
        assert tracker.current_mutation_counts.non_neutral == 0
        assert tracker.current_mutation_counts.failed == 0

    def test_get_summary_metrics_empty(self) -> None:
        """Test getting summary metrics when no history exists."""
        tracker = StatsTracker()

        result = tracker.get_summary_metrics()

        assert result == {}

    def test_get_summary_metrics_with_history(self) -> None:
        """Test getting summary metrics with history."""
        tracker = StatsTracker()

        # Create stats with different values
        stats1 = self.create_test_stats(1, avg_length=1000.0)
        stats2 = self.create_test_stats(2, avg_length=1100.0)
        stats3 = self.create_test_stats(3, avg_length=1200.0)

        tracker.record_generation(stats1)
        tracker.record_generation(stats2)
        tracker.record_generation(stats3)

        result = tracker.get_summary_metrics()

        assert result["total_generations"] == 3
        assert result["final_avg_length"] == 1200.0
        assert result["initial_avg_length"] == 1000.0
        assert result["length_change"] == 200.0
        assert result["max_population_size"] == 100
        assert result["min_population_size"] == 100
        assert result["total_mutations"] == 30  # 3 generations * 10 mutations each
        assert "avg_survival_rate" in result

    def test_get_summary_metrics_no_mutations(self) -> None:
        """Test summary metrics when no mutations were applied."""
        tracker = StatsTracker()

        # Create stats with zero mutations
        stats = PopulationStats(
            generation=1,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=900,
            max_genome_length=1100,
            length_std=50.0,
            total_mutations_applied=0,
            neutral_mutations=0,
            non_neutral_mutations=0,
            survivors_after_mutations=100,
        )

        tracker.record_generation(stats)

        result = tracker.get_summary_metrics()

        assert result["avg_survival_rate"] == 0.0
