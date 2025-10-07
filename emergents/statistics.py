"""
Statistics collection and analysis for evolution simulation.

This module provides comprehensive statistics tracking and analysis tools
for monitoring population evolution over time.
"""

import statistics
from dataclasses import dataclass
from typing import Any, Optional

from emergents.genome.genome import Genome


@dataclass
class PopulationStats:
    """Statistics for a population at a given generation."""

    generation: int
    population_size: int
    avg_genome_length: float
    min_genome_length: int
    max_genome_length: int
    length_std: float
    total_mutations_applied: int
    neutral_mutations: int
    non_neutral_mutations: int
    survivors_after_mutations: int

    # Additional derived metrics
    length_diversity: float = 0.0
    mutation_survival_rate: float = 0.0

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.total_mutations_applied > 0:
            self.mutation_survival_rate = (
                self.neutral_mutations / self.total_mutations_applied
            )

    def __str__(self) -> str:
        return (
            f"Gen {self.generation}: Pop={self.population_size}, "
            f"AvgLen={self.avg_genome_length:.1f}±{self.length_std:.1f}, "
            f"Mutations={self.total_mutations_applied} "
            f"(Survival: {self.mutation_survival_rate:.1%}), "
            f"Survivors: {self.survivors_after_mutations}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "avg_genome_length": self.avg_genome_length,
            "min_genome_length": self.min_genome_length,
            "max_genome_length": self.max_genome_length,
            "length_std": self.length_std,
            "total_mutations_applied": self.total_mutations_applied,
            "neutral_mutations": self.neutral_mutations,
            "non_neutral_mutations": self.non_neutral_mutations,
            "survivors_after_mutations": self.survivors_after_mutations,
            "length_diversity": self.length_diversity,
            "mutation_survival_rate": self.mutation_survival_rate,
        }


@dataclass
class MutationCounts:
    """Track different types of mutation applications."""

    total: int = 0
    neutral: int = 0
    non_neutral: int = 0
    failed: int = 0

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total = 0
        self.neutral = 0
        self.non_neutral = 0
        self.failed = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "neutral": self.neutral,
            "non_neutral": self.non_neutral,
            "failed": self.failed,
        }


class StatsCalculator:
    """Calculate population statistics efficiently."""

    @staticmethod
    def calculate_population_stats(
        genomes: list[Genome],
        generation: int,
        mutation_counts: MutationCounts,
        survivors_count: int,
    ) -> PopulationStats:
        """
        Calculate comprehensive population statistics.

        Args:
            genomes: List of genomes in the population
            generation: Current generation number
            mutation_counts: Counts of mutations applied
            survivors_count: Number of genomes that survived mutations

        Returns:
            PopulationStats object with all metrics
        """
        if not genomes:
            return PopulationStats(
                generation=generation,
                population_size=0,
                avg_genome_length=0.0,
                min_genome_length=0,
                max_genome_length=0,
                length_std=0.0,
                total_mutations_applied=mutation_counts.total,
                neutral_mutations=mutation_counts.neutral,
                non_neutral_mutations=mutation_counts.non_neutral,
                survivors_after_mutations=survivors_count,
            )

        lengths = [len(genome) for genome in genomes]
        unique_lengths = set(lengths)

        return PopulationStats(
            generation=generation,
            population_size=len(genomes),
            avg_genome_length=statistics.mean(lengths),
            min_genome_length=min(lengths),
            max_genome_length=max(lengths),
            length_std=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            total_mutations_applied=mutation_counts.total,
            neutral_mutations=mutation_counts.neutral,
            non_neutral_mutations=mutation_counts.non_neutral,
            survivors_after_mutations=survivors_count,
            length_diversity=len(unique_lengths) / len(lengths),
        )

    @staticmethod
    def calculate_diversity_metrics(genomes: list[Genome]) -> dict[str, float]:
        """
        Calculate comprehensive diversity metrics.

        Args:
            genomes: List of genomes to analyze

        Returns:
            Dictionary with diversity statistics
        """
        if not genomes:
            return {
                "length_diversity": 0.0,
                "length_std": 0.0,
                "length_coefficient_variation": 0.0,
                "unique_length_count": 0,
            }

        lengths = [len(genome) for genome in genomes]
        unique_lengths = set(lengths)
        mean_length = statistics.mean(lengths)
        std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

        return {
            "length_diversity": len(unique_lengths) / len(lengths),
            "length_std": std_length,
            "length_coefficient_variation": (
                std_length / mean_length if mean_length > 0 else 0.0
            ),
            "unique_length_count": len(unique_lengths),
        }


class StatsTracker:
    """Track and manage evolution statistics over time."""

    def __init__(self) -> None:
        self.history: list[PopulationStats] = []
        self.current_mutation_counts = MutationCounts()

    def record_generation(self, stats: PopulationStats) -> None:
        """Record statistics for a generation."""
        self.history.append(stats)

    def get_latest_stats(self) -> Optional[PopulationStats]:
        """Get the most recent statistics."""
        return self.history[-1] if self.history else None

    def get_generation_stats(self, generation: int) -> Optional[PopulationStats]:
        """Get statistics for a specific generation."""
        for stats in self.history:
            if stats.generation == generation:
                return stats
        return None

    def get_stats_range(self, start_gen: int, end_gen: int) -> list[PopulationStats]:
        """Get statistics for a range of generations."""
        return [
            stats for stats in self.history if start_gen <= stats.generation <= end_gen
        ]

    def reset_mutation_counts(self) -> None:
        """Reset mutation counters for the next generation."""
        self.current_mutation_counts.reset()

    def get_summary_metrics(self) -> dict[str, Any]:
        """Get summary metrics across all generations."""
        if not self.history:
            return {}

        avg_lengths = [stats.avg_genome_length for stats in self.history]
        pop_sizes = [stats.population_size for stats in self.history]

        return {
            "total_generations": len(self.history),
            "final_avg_length": avg_lengths[-1],
            "initial_avg_length": avg_lengths[0],
            "length_change": avg_lengths[-1] - avg_lengths[0],
            "max_population_size": max(pop_sizes),
            "min_population_size": min(pop_sizes),
            "total_mutations": sum(
                stats.total_mutations_applied for stats in self.history
            ),
            "avg_survival_rate": (
                statistics.mean(
                    [
                        stats.mutation_survival_rate
                        for stats in self.history
                        if stats.total_mutations_applied > 0
                    ]
                )
                if any(stats.total_mutations_applied > 0 for stats in self.history)
                else 0.0
            ),
        }
