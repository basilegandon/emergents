"""
Optimized population-based evolution simulation for Genomes.

This module provides a high-performance Population class with clean separation
of concerns, efficient algorithms, and comprehensive analysis capabilities.
"""

import copy
import random
import signal
from types import FrameType
from typing import Any, Literal, Optional

from tqdm import tqdm

from emergents.config import MutationConfig
from emergents.file_plotter import MultiprocessFilePlotter, PlotData
from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.logging_config import get_logger
from emergents.mutation_manager import MutationManager
from emergents.statistics import (
    MutationCounts,
    PopulationStats,
    StatsCalculator,
    StatsTracker,
)

logger = get_logger(__name__)


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
            "Initialized population: size=%d, mutation_rate=%g, seed=%s",
            population_size,
            mutation_rate,
            random_seed,
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
        length_coding_segments: int | list[int],
        length_non_coding_segments: int | list[int],
        promoter_directions: PromoterDirection | list[PromoterDirection],
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
        plot_update_interval: Optional[int] = None,
        plot_filename: str = "evolution_progress.png",
    ) -> list[PlotData]:
        """
        Evolve the population for multiple generations with comprehensive reporting.

        Args:
            num_generations: Number of generations to evolve
            report_every: Print stats every N generations (0 = no reporting)
            plot_update_interval: Update plots every N generations (None = no plotting)
            plot_filename: Filename for saved plots

        Returns:
            List of PlotData objects for each generation

        Raises:
            ValueError: If num_generations is not positive
            RuntimeError: If population goes extinct
        """
        if num_generations <= 0:
            raise ValueError("Number of generations must be positive")

        logger.info(
            "Starting evolution: %d generations, population_size=%d, mutation_rate=%g",
            num_generations,
            len(self.genomes),
            self.mutation_rate,
        )

        if plot_update_interval:
            logger.info(
                "File-based plotting enabled (save to '%s' every %d generations)",
                plot_filename,
                plot_update_interval,
            )
            logger.info("Press Ctrl+C to interrupt and save all data...")
        logger.info("-" * 60)

        plot_data_list: list[PlotData] = []  # For enhanced plotting with genome lengths

        # Set up plotting if enabled
        plotter: MultiprocessFilePlotter | None = None
        interrupted = False

        if plot_update_interval:
            try:
                plotter = MultiprocessFilePlotter(
                    filename=plot_filename,
                    save_history=True,
                    history_dir="evolution_plots",
                    title="Evolution Progress",
                    max_queue_size=100,
                )

                if plotter:
                    plotter.initialize()
            except Exception as e:
                logger.warning("Failed to initialize plotter: %s", e)
                plotter = None

        def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
            nonlocal interrupted
            interrupted = True
            logger.warning(
                f"Keyboard interrupt received at generation {len(plot_data_list)}!"
            )
            logger.warning("Current data will be preserved.")
            if plotter:
                try:
                    # Only update the plotter with current data, don't close here
                    # The finally block will handle closing properly
                    plotter.update(plot_data_list)
                except Exception as e:
                    logger.warning("Error updating plotter on interrupt: %s", e)

        # Set up keyboard interrupt handler
        original_handler = signal.signal(signal.SIGINT, signal_handler)

        progress_bar = None
        try:
            progress_bar = tqdm(
                range(num_generations),
                desc="Evolving Generations",
                unit="gen",
                leave=False,
            )

            for gen in progress_bar:
                # Check for interruption
                if interrupted:
                    break

                stats = self.evolve_one_generation()
                plot_data_list.append(
                    PlotData(stats=stats, genome_lengths=self.get_genome_lengths())
                )

                if report_every > 0 and gen % report_every == 0:
                    logger.info("%s", stats)

                # Update real-time plot if enabled
                if plotter and plot_update_interval and gen % plot_update_interval == 0:
                    plotter.update(plot_data_list)

        except RuntimeError as e:
            logger.error("Evolution failed at generation %d: %s", self.generation, e)
            raise
        except Exception as e:
            logger.error("Unexpected error during evolution: %s", e)
            raise
        finally:
            # Restore original signal handler first
            try:
                signal.signal(signal.SIGINT, original_handler)
            except Exception as e:
                logger.warning("Error restoring signal handler: %s", e)

            logger.info("Evolution run ended, finalizing...")

            # Close progress bar explicitly
            if progress_bar:
                try:
                    logger.info("Closing progress bar...")
                    progress_bar.close()
                    logger.info("Progress bar closed")
                except Exception as e:
                    logger.warning("Error closing progress bar: %s", e)

            # Close plotter if it was created
            if plotter:
                try:
                    logger.info("Closing plotter...")
                    plotter.close()
                    logger.info("Plotter closed successfully")
                except Exception as e:
                    logger.warning("Error closing plotter: %s", e)

            # Force cleanup of all multiprocessing resources more aggressively
            logger.info("Cleaning up multiprocessing resources...")
            import multiprocessing as mp
            import time

            try:
                active_children = mp.active_children()
                logger.info("Found %d active child processes", len(active_children))

                for p in active_children:
                    if p.is_alive():
                        logger.info("Terminating process %s (PID: %s)", p.name, p.pid)
                        p.terminate()
                        p.join(timeout=1.0)

                        if p.is_alive():
                            logger.warning("Force killing process %s", p.name)
                            try:
                                import os

                                if hasattr(os, "kill") and p.pid:
                                    os.kill(p.pid, 9)
                            except Exception as e:
                                logger.error("Failed to kill process %s: %s", p.name, e)

                # Force cleanup of any remaining resources
                time.sleep(0.1)  # Small delay to allow cleanup
                logger.info("Multiprocessing cleanup completed")

            except Exception as e:
                logger.warning("Error cleaning up multiprocessing resources: %s", e)

            # Additional cleanup for potential hanging resources
            try:
                import gc

                gc.collect()  # Force garbage collection
            except Exception as e:
                logger.warning("Error during garbage collection: %s", e)

        logger.info("-" * 60)
        if interrupted:
            logger.info("Evolution interrupted at generation %d!", len(plot_data_list))
            logger.info("All data up to interruption has been preserved.")

            # After cleanup is complete, re-raise KeyboardInterrupt so calling code knows it was interrupted
            if plot_data_list:
                final_stats = plot_data_list[-1].stats
                logger.info("Final: %s", final_stats)
            raise KeyboardInterrupt("Evolution interrupted by user")
        else:
            logger.info("Evolution complete!")

        if plot_data_list:
            final_stats = plot_data_list[-1].stats
            logger.info("Final: %s", final_stats)
            logger.info("Evolution completed successfully: %s", final_stats)

        return plot_data_list

    def get_genome_diversity(self) -> dict[str, float]:
        """
        Calculate comprehensive diversity metrics for the population.

        Returns:
            Dictionary with diversity statistics
        """
        return StatsCalculator.calculate_diversity_metrics(self.genomes)

    def get_genome_lengths(self) -> list[int]:
        """
        Get the lengths of all genomes in the population.

        Returns:
            List of genome lengths
        """
        return [len(genome) for genome in self.genomes]

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get a summary of the evolution history."""
        return self.stats_tracker.get_summary_metrics()
