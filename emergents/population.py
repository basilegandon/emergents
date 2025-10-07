"""
Population-based evolution simulation for Genomes.

This module provides classes for managing populations of genomes and simulating
evolutionary processes through repeated application of mutations over generations.
"""

import copy
import random
import statistics
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
from tqdm import tqdm

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.base import Mutation
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion


@dataclass
class PopulationStats:
    """Statistics for a population at a given generation."""

    generation: int
    population_size: int
    avg_genome_length: float
    min_genome_length: int
    max_genome_length: int
    total_mutations_applied: int
    neutral_mutations: int
    non_neutral_mutations: int
    livings_after_mutations: int

    def __str__(self) -> str:
        return (
            f"\nGen {self.generation}: Pop={self.population_size}, "
            f"AvgLen={self.avg_genome_length:.1f}, "
            f"Mutations={self.total_mutations_applied} "
            f"({self.neutral_mutations}Neutral/{self.non_neutral_mutations}NonNeutral) "
            f"{self.livings_after_mutations} Survivors"
        )


class Population:
    """
    A population of genomes that can evolve through mutation and selection.

    The population manages a collection of genomes and provides methods for:
    - Random initialization of genomes
    - Application of mutations according to specified rates
    - Selection pressure based on fitness functions
    - Evolution over multiple generations
    """

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.01,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize a population.

        Args:
            population_size: Number of genomes in the population
            mutation_rate: Probability that each genome gets mutated per generation
            random_seed: Seed for reproducible random number generation
        """
        if population_size <= 0:
            raise ValueError("Population size must be positive")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.genomes: list[Genome] = []
        self.generation = 0
        self.stats_history: list[PopulationStats] = []

        # Set up random number generator
        if random_seed is not None:
            random.seed(random_seed)

        # Available mutation types and their weights
        uniform_weight = 1.0 / 6.0
        self.mutation_types: dict[type[Mutation], float] = {
            PointMutation: uniform_weight,
            SmallDeletion: uniform_weight,
            SmallInsertion: uniform_weight,
            Deletion: uniform_weight,
            Duplication: uniform_weight,
            Inversion: uniform_weight,
        }

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

    def add_genome(self, genome: Genome) -> None:
        """Add a genome to the population."""
        self.genomes.append(genome)
        if len(self.genomes) > self.population_size:
            self.population_size = len(self.genomes)

    def set_mutation_weights(self, weights: dict[type[Mutation], float]) -> None:
        """
        Set the relative weights for different mutation types.

        Args:
            weights: Dictionary mapping mutation types to their relative weights
        """
        if not weights:
            raise ValueError("Weights dictionary cannot be empty")

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        self.mutation_types = {
            mut_type: weight / total_weight for mut_type, weight in weights.items()
        }

    def _select_mutation_type(self) -> type[Mutation]:
        """Randomly select a mutation type based on weights."""
        types = list(self.mutation_types.keys())
        weights = list(self.mutation_types.values())
        return random.choices(types, weights=weights)[0]

    def _create_random_mutation(self, genome: Genome) -> Mutation:
        """Create a random mutation appropriate for the given genome."""
        mutation_type = self._select_mutation_type()

        if mutation_type == PointMutation:
            position = random.randint(0, len(genome) - 1)
            return PointMutation(position=position)

        elif mutation_type == SmallDeletion:
            max_del_size = min(10, len(genome))
            del_size = random.randint(1, max_del_size)
            start_pos = random.randint(0, len(genome) - del_size)
            return SmallDeletion(position=start_pos, length=del_size)

        elif mutation_type == SmallInsertion:
            position = random.randint(0, len(genome))
            length = random.randint(1, 10)
            return SmallInsertion(position=position, length=length)

        elif mutation_type == Deletion:
            del_size = random.randint(1, len(genome))
            start_pos = random.randint(0, len(genome) - del_size)
            end_pos = start_pos + del_size - 1
            return Deletion(start_pos=start_pos, end_pos=end_pos)

        elif mutation_type == Duplication:
            dup_size = random.randint(1, len(genome))
            start_pos = random.randint(0, len(genome) - dup_size)
            end_pos = start_pos + dup_size - 1
            insertion_pos = random.randint(0, len(genome))
            return Duplication(
                start_pos=start_pos, end_pos=end_pos, insertion_pos=insertion_pos
            )

        elif mutation_type == Inversion:
            inv_size = random.randint(1, len(genome))
            start_pos = random.randint(0, len(genome) - inv_size)
            end_pos = start_pos + inv_size
            return Inversion(start_pos=start_pos, end_pos=end_pos)

        else:
            # Fallback
            return PointMutation(position=random.randint(0, len(genome) - 1))

    def apply_mutations(self) -> tuple[dict[str, int], list[int]]:
        """
        Apply mutations to the population based on mutation rate.

        Returns:
            A tuple containing:
            - Dictionary with counts of mutations applied
            - List of indices of genomes that had non-neutral mutations
        """
        mutation_counts = {"total": 0, "neutral": 0, "non_neutral": 0, "failed": 0}

        dead_genomes_indices: list[int] = []
        for genome_idx, genome in enumerate(self.genomes):
            nb_mutations: int = np.random.binomial(genome.length, self.mutation_rate)
            for _ in range(nb_mutations):
                if len(self.genomes) == 0:
                    dead_genomes_indices.append(genome_idx)
                    break
                mutation = self._create_random_mutation(genome)

                if mutation.is_neutral(genome):
                    mutation_counts["neutral"] += 1
                    mutation.apply(genome)

                else:
                    mutation_counts["non_neutral"] += 1
                    dead_genomes_indices.append(genome_idx)

                mutation_counts["total"] += 1
                genome.coalesce_all()
        return mutation_counts, dead_genomes_indices

    def get_population_stats(self, current_pop_size: int) -> PopulationStats:
        """Calculate and return statistics for the current population."""
        if not self.genomes:
            return PopulationStats(
                generation=self.generation,
                population_size=0,
                avg_genome_length=0.0,
                min_genome_length=0,
                max_genome_length=0,
                total_mutations_applied=0,
                neutral_mutations=0,
                non_neutral_mutations=0,
                livings_after_mutations=0,
            )

        lengths = [len(genome) for genome in self.genomes]

        # Get mutation counts from the last evolution step
        last_mutation_counts = getattr(
            self, "_last_mutation_counts", {"total": 0, "neutral": 0, "non_neutral": 0}
        )

        return PopulationStats(
            generation=self.generation,
            population_size=len(self.genomes),
            avg_genome_length=statistics.mean(lengths),
            min_genome_length=min(lengths),
            max_genome_length=max(lengths),
            total_mutations_applied=last_mutation_counts.get("total", 0),
            neutral_mutations=last_mutation_counts.get("neutral", 0),
            non_neutral_mutations=last_mutation_counts.get("non_neutral", 0),
            livings_after_mutations=current_pop_size,
        )

    def evolve_one_generation(self) -> PopulationStats:
        """
        Evolve the population for one generation.

        Args:
            selection_pressure: Fraction of population to replace via selection

        Returns:
            Statistics for this generation
        """
        # Apply mutations
        mutation_counts, dead_genomes_indices = self.apply_mutations()
        self._last_mutation_counts = mutation_counts

        # Remove dead genomes (in reverse order to maintain indices)
        for idx in sorted(set(dead_genomes_indices), reverse=True):
            if 0 <= idx < len(self.genomes):
                del self.genomes[idx]

        # Replenish population by copying surviving genomes
        current_pop_size = len(self.genomes)
        if current_pop_size == 0:
            raise RuntimeError("All genomes died! Population extinct.")

        new_genomes: list[Genome] = []
        for _ in range(self.population_size):
            parent = random.choice(self.genomes)
            new_genomes.append(copy.deepcopy(parent))

        self.genomes = new_genomes

        # Update generation counter
        self.generation += 1

        # Calculate and store statistics
        stats = self.get_population_stats(current_pop_size)
        self.stats_history.append(stats)

        return stats

    def evolve(
        self,
        num_generations: int,
        report_every: int = 100,
    ) -> list[PopulationStats]:
        """
        Evolve the population for multiple generations.

        Args:
            num_generations: Number of generations to evolve
            selection_pressure: Selection pressure per generation
            report_every: Print stats every N generations (0 = no reporting)

        Returns:
            List of statistics for each generation
        """
        if num_generations <= 0:
            raise ValueError("Number of generations must be positive")

        print(f"Starting evolution for {num_generations} generations...")
        print(f"Initial population size: {len(self.genomes)}")
        print(f"Mutation rate: {self.mutation_rate}")
        print("-" * 60)

        generation_stats: list[PopulationStats] = []

        for gen in tqdm(
            range(num_generations), desc="Evolving Generations", unit="gen"
        ):
            stats = self.evolve_one_generation()
            generation_stats.append(stats)

            if report_every > 0 and gen % report_every == 0:
                print(stats)

        print("-" * 60)
        print("Evolution complete!")
        final_stats: PopulationStats = (
            generation_stats[-1]
            if generation_stats
            else self.get_population_stats(current_pop_size=len(self.genomes))
        )
        print(f"Final: {final_stats}")

        return generation_stats

    def get_genome_diversity(self) -> dict[str, Any]:
        """
        Calculate diversity metrics for the population.

        Returns:
            Dictionary with diversity statistics
        """
        if not self.genomes:
            return {"length_diversity": 0, "length_std": 0}

        lengths = [len(genome) for genome in self.genomes]

        return {
            "length_diversity": len(set(lengths))
            / len(lengths),  # Fraction of unique lengths
            "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        }
