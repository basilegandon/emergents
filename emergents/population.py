"""
Optimized population-based evolution simulation for Genomes.

This module provides a high-performance Population class with clean separation
of concerns, efficient algorithms, and comprehensive analysis capabilities.
"""

import copy
import logging
import random
from typing import Any, Literal, Optional, Union

from tqdm import tqdm

from emergents.config import MutationConfig
from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutation_manager import MutationManager
from emergents.statistics import (
    MutationCounts,
    PopulationStats,
    StatsCalculator,
    StatsTracker,
)

logger = logging.getLogger(__name__)


# PopulationStats class moved to emergents.statistics module


class Population:
    """
    Optimized population manager for genome evolution simulation.

    This class provides high-performance population management with clean
    separation of concerns:
    - Genome initialization and management
    - Evolution orchestration (delegates mutation logic)
    - Statistics tracking and reporting
    - Efficient genome selection and replication
    """

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.01,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a population with validation.

        Args:
            population_size: Number of genomes in the population
            mutation_rate: Probability that each genome gets mutated per generation
            random_seed: Seed for reproducible random number generation

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_init_params(population_size, mutation_rate)

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.genomes: list[Genome] = []
        self.generation = 0

        # Set up random number generator
        if random_seed is not None:
            random.seed(random_seed)

        # Initialize components with default configuration
        mutation_config = MutationConfig()
        self.mutation_manager = MutationManager(mutation_config)
        self.stats_tracker = StatsTracker()

        logger.info(
            f"Initialized population: size={population_size}, "
            f"mutation_rate={mutation_rate}, seed={random_seed}"
        )

    @staticmethod
    def _validate_init_params(population_size: int, mutation_rate: float) -> None:
        """Validate initialization parameters."""
        if population_size <= 0:
            raise ValueError("Population size must be positive")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")

    def initialize_population(
        self,
        initial_genome_length: int,
        nb_coding_segments: int,
        length_coding_segments: Union[int, list[int]],
        length_non_coding_segments: Union[int, list[int]],
        promoter_directions: Union[PromoterDirection, list[PromoterDirection]],
        is_circular: bool = False,
        nature_of_extremities: Literal["C--C", "NC--NC", "C--NC"] = "NC--NC",
    ) -> None:
        """
        Initialize the population with a random genome copied for all individuals.

        Args:
            initial_genome_length: Length of the initial genome
            nb_coding_segments: Number of coding segments to include
            length_coding_segments: Length(s) of coding segments (int or list of ints). If int, all coding segments will have this length. If list, it should have length equal to nb_coding_segments. Sum of all lengths should be less than initial_genome_length.
            length_non_coding_segments: Length(s) of non-coding segments (int or list of ints). If int, all non-coding segments will have this length. If list, it should have length equal to the number of non-coding segments. Sum of all segment lengths should be less than initial_genome_length.
            promoter_directions: Direction(s) of promoters for coding segments (PromoterDirection or list of PromoterDirection). If single value, all coding segments will have this direction. If list, it should have length equal to nb_coding_segments.
            is_circular: Whether the genomes are circular
            nature_of_extremities: Nature of linear genome ends ('C--C', 'NC--NC', 'C--NC')
        """
        if initial_genome_length <= 0:
            raise ValueError("Initial genome length must be positive")
        if nb_coding_segments < 0:
            raise ValueError("Number of coding segments cannot be negative")

        if nb_coding_segments == 0:
            nb_non_coding_segments = 1
        else:
            nb_non_coding_segments = nb_coding_segments
            if not is_circular and nature_of_extremities == "NC--NC":
                nb_non_coding_segments = nb_coding_segments + 1

        # Check that lengths are valid
        if nb_coding_segments > initial_genome_length:
            raise ValueError(
                "Total number of coding segments exceeds initial genome length"
            )

        if isinstance(length_coding_segments, int):
            length_coding_segments = [length_coding_segments] * nb_coding_segments
        elif len(length_coding_segments) != nb_coding_segments:
            raise ValueError(
                "Length of coding segments list does not match number of coding segments"
            )

        if isinstance(promoter_directions, PromoterDirection):
            promoter_directions = [promoter_directions] * nb_coding_segments
        elif len(promoter_directions) != nb_coding_segments:
            raise ValueError(
                "Length of promoter_directions list does not match number of coding segments"
            )

        if isinstance(length_non_coding_segments, int):
            length_non_coding_segments = [
                length_non_coding_segments
            ] * nb_non_coding_segments
        elif len(length_non_coding_segments) != nb_non_coding_segments:
            raise ValueError(
                "Length of non-coding segments list does not match number of non-coding segments"
            )

        if (
            sum(length_coding_segments) + sum(length_non_coding_segments)
            != initial_genome_length
        ):
            raise ValueError(
                "Sum of segment lengths is not equal to initial genome length"
            )

        segments: list[Segment] = []

        if not is_circular:
            if nature_of_extremities == "NC--NC":
                segments.append(NonCodingSegment(length=length_non_coding_segments[0]))
                length_non_coding_segments = length_non_coding_segments[1:]
        for coding_length, promoter_direction, non_coding_length in zip(
            length_coding_segments, promoter_directions, length_non_coding_segments
        ):
            segments.append(
                CodingSegment(
                    length=coding_length, promoter_direction=promoter_direction
                )
            )
            segments.append(NonCodingSegment(length=non_coding_length))

        if len(length_non_coding_segments) > len(length_coding_segments):
            segments.append(NonCodingSegment(length=length_non_coding_segments[-1]))

        genome = Genome(segments=segments, circular=is_circular)

        self.genomes = [copy.deepcopy(genome) for _ in range(self.population_size)]

    def update_mutation_config(self, config: MutationConfig) -> None:
        """Update mutation configuration."""
        self.mutation_manager.update_config(config)
        logger.info("Updated mutation configuration")

    def apply_mutations(self) -> tuple[MutationCounts, list[int]]:
        """
        Apply mutations to the population using the mutation manager.

        Returns:
            Tuple containing mutation counts and indices of dead genomes
        """
        return self.mutation_manager.apply_mutations_to_population(
            self.genomes, self.mutation_rate
        )

    def get_population_stats(self, current_pop_size: int) -> PopulationStats:
        """
        Calculate and return statistics for the current population.

        Args:
            current_pop_size: Number of survivors after mutations

        Returns:
            PopulationStats object with current generation statistics
        """
        # Get the last mutation counts from stats tracker
        last_mutation_counts = self.stats_tracker.current_mutation_counts

        return StatsCalculator.calculate_population_stats(
            genomes=self.genomes,
            generation=self.generation,
            mutation_counts=last_mutation_counts,
            survivors_count=current_pop_size,
        )

    def evolve_one_generation(self) -> PopulationStats:
        """
        Evolve the population for one generation with optimized performance.

        Returns:
            Statistics for this generation
        """
        # Reset mutation counts for new generation
        self.stats_tracker.reset_mutation_counts()

        # Apply mutations using the mutation manager
        mutation_counts, dead_genomes_indices = self.apply_mutations()

        # Update mutation counts in stats tracker
        self.stats_tracker.current_mutation_counts = mutation_counts

        # Remove dead genomes efficiently (in reverse order to maintain indices)
        survivors_count = len(self.genomes)
        if dead_genomes_indices:
            dead_set = set(dead_genomes_indices)
            survivors_count = len(self.genomes) - len(dead_set)

            # Create new list without dead genomes
            self.genomes = [
                genome for idx, genome in enumerate(self.genomes) if idx not in dead_set
            ]

        if survivors_count == 0:
            raise RuntimeError("All genomes died! Population extinct.")

        self._replenish_population(survivors_count)

        # Update generation counter
        self.generation += 1

        # Calculate and store statistics
        stats = self.get_population_stats(survivors_count)
        self.stats_tracker.record_generation(stats)

        return stats

    def _replenish_population(self, survivors_count: int) -> None:
        """
        Replenish population to target size.
        We need to replenish from 0 to simulate luck: even a good genome can fail
        to reproduce. This is key to model the drift effect.

        Args:
            survivors_count: Number of genomes that survived
        """

        # Create new genomes by copying from survivors
        new_genomes: list[Genome] = []
        for _ in range(self.population_size):
            parent = random.choice(self.genomes)
            new_genomes.append(copy.deepcopy(parent))

        self.genomes = new_genomes

    def evolve(
        self,
        num_generations: int,
        report_every: int = 100,
    ) -> list[PopulationStats]:
        """
        Evolve the population for multiple generations with comprehensive reporting.

        Args:
            num_generations: Number of generations to evolve
            report_every: Print stats every N generations (0 = no reporting)

        Returns:
            List of statistics for each generation

        Raises:
            ValueError: If num_generations is not positive
            RuntimeError: If population goes extinct
        """
        if num_generations <= 0:
            raise ValueError("Number of generations must be positive")

        logger.info(
            f"Starting evolution: {num_generations} generations, "
            f"population_size={len(self.genomes)}, "
            f"mutation_rate={self.mutation_rate}"
        )

        print(f"Starting evolution for {num_generations} generations...")
        print(f"Initial population size: {len(self.genomes)}")
        print(f"Mutation rate: {self.mutation_rate}")
        print("-" * 60)

        generation_stats: list[PopulationStats] = []

        try:
            for gen in tqdm(
                range(num_generations), desc="Evolving Generations", unit="gen"
            ):
                stats = self.evolve_one_generation()
                generation_stats.append(stats)

                if report_every > 0 and gen % report_every == 0:
                    print(stats)
                    logger.info(f"Generation {gen}: {stats}")

        except RuntimeError as e:
            logger.error(f"Evolution failed at generation {self.generation}: {e}")
            raise

        print("-" * 60)
        print("Evolution complete!")

        if generation_stats:
            final_stats = generation_stats[-1]
            print(f"Final: {final_stats}")
            logger.info(f"Evolution completed successfully: {final_stats}")

        return generation_stats

    def get_genome_diversity(self) -> dict[str, float]:
        """
        Calculate comprehensive diversity metrics for the population.

        Returns:
            Dictionary with diversity statistics
        """
        return StatsCalculator.calculate_diversity_metrics(self.genomes)

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get a summary of the evolution history."""
        return self.stats_tracker.get_summary_metrics()
