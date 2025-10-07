"""
Mutation management and application for genome evolution.

This module provides efficient and configurable mutation application
with proper separation of concerns from population management.
"""

import random

import numpy as np

from emergents.config import MutationConfig
from emergents.genome.genome import Genome
from emergents.mutations.base import Mutation
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion
from emergents.statistics import MutationCounts


class MutationManager:
    """
    Manages mutation application to genomes with configurable parameters.

    This class separates mutation logic from population management,
    providing efficient and flexible mutation application.
    """

    def __init__(self, config: MutationConfig) -> None:
        """
        Initialize the mutation manager.

        Args:
            config: Configuration for mutation parameters
        """
        self.config = config
        self.mutation_types = list(config.weights.keys())
        self.mutation_weights = list(config.weights.values())

    def update_config(self, config: MutationConfig) -> None:
        """Update mutation configuration."""
        self.config = config
        self.mutation_types = list(config.weights.keys())
        self.mutation_weights = list(config.weights.values())

    def select_mutation_type(self) -> type[Mutation]:
        """Randomly select a mutation type based on configured weights."""
        return random.choices(self.mutation_types, weights=self.mutation_weights)[0]

    def create_mutation(
        self, genome: Genome, mutation_type: type[Mutation]
    ) -> Mutation:
        """
        Create a mutation of the specified type for the given genome.

        Args:
            genome: Target genome for the mutation
            mutation_type: Type of mutation to create

        Returns:
            Configured mutation instance
        """
        genome_length = len(genome)

        if mutation_type == SmallDeletion:
            max_del_size = min(self.config.small_mutation_max_size, genome_length)
            del_size = random.randint(1, max_del_size)
            start_pos = random.randint(0, genome_length - del_size)
            return SmallDeletion(position=start_pos, length=del_size)

        elif mutation_type == SmallInsertion:
            position = random.randint(0, genome_length)
            length = random.randint(1, self.config.small_mutation_max_size)
            return SmallInsertion(position=position, length=length)

        elif mutation_type == Deletion:
            del_size = random.randint(1, genome_length)
            start_pos = random.randint(0, genome_length - del_size)
            end_pos = start_pos + del_size - 1
            return Deletion(start_pos=start_pos, end_pos=end_pos)

        elif mutation_type == Duplication:
            dup_size = random.randint(1, genome_length)
            start_pos = random.randint(0, genome_length - dup_size)
            end_pos = start_pos + dup_size - 1
            insertion_pos = random.randint(0, genome_length)
            return Duplication(
                start_pos=start_pos, end_pos=end_pos, insertion_pos=insertion_pos
            )

        elif mutation_type == Inversion:
            inv_size = random.randint(1, genome_length)
            start_pos = random.randint(0, genome_length - inv_size)
            end_pos = start_pos + inv_size
            return Inversion(start_pos=start_pos, end_pos=end_pos)

        else:  # PointMutation
            position = random.randint(0, genome_length - 1)
            return PointMutation(position=position)

    def apply_mutations_to_genome(
        self, genome: Genome, mutation_rate: float
    ) -> tuple[int, int, bool]:
        """
        Apply mutations to a single genome.

        Args:
            genome: Target genome
            mutation_rate: Probability of mutation per base pair

        Returns:
            Tuple of (neutral_mutations, non_neutral_mutations, genome_survived)
        """
        if len(genome) == 0:
            return 0, 0, False

        num_mutations = np.random.binomial(len(genome), mutation_rate)
        neutral_count = 0
        non_neutral_count = 0

        for _ in range(num_mutations):
            if len(genome) == 0:
                # Genome died during mutation
                return neutral_count, non_neutral_count, False

            mutation_type = self.select_mutation_type()
            mutation = self.create_mutation(genome, mutation_type)

            try:
                if mutation.is_neutral(genome):
                    mutation.apply(genome)
                    neutral_count += 1
                    # Coalesce segments after each mutation for efficiency
                    genome.coalesce_all()
                else:
                    non_neutral_count += 1
                    # Non-neutral mutation kills the genome
                    return neutral_count, non_neutral_count, False

            except Exception:
                # Mutation failed - treat as non-neutral
                non_neutral_count += 1
                return neutral_count, non_neutral_count, False

        return neutral_count, non_neutral_count, True

    def apply_mutations_to_population(
        self, genomes: list[Genome], mutation_rate: float
    ) -> tuple[MutationCounts, list[int]]:
        """
        Apply mutations to a population of genomes.

        Args:
            genomes: List of genomes to mutate
            mutation_rate: Probability of mutation per base pair

        Returns:
            Tuple of (mutation_counts, indices_of_dead_genomes)
        """
        mutation_counts = MutationCounts()
        dead_genome_indices: list[int] = []

        for genome_idx, genome in enumerate(genomes):
            neutral, non_neutral, survived = self.apply_mutations_to_genome(
                genome, mutation_rate
            )

            mutation_counts.neutral += neutral
            mutation_counts.non_neutral += non_neutral
            mutation_counts.total += neutral + non_neutral

            if not survived:
                dead_genome_indices.append(genome_idx)

        return mutation_counts, dead_genome_indices
