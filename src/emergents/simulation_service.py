"""
High-level orchestrator for the evolution simulation workflow.

This module provides a service-oriented architecture that manages the complete
simulation lifecycle from initialization through reporting results.
"""

from emergents.config import SimulationConfig
from emergents.file_plotter import PlotData
from emergents.logging_config import get_logger
from emergents.population import Population, PopulationStats

logger = get_logger(__name__)


class SimulationService:
    """
    High-level orchestrator for the entire simulation.

    This service manages the complete simulation lifecycle:
    - Initializing population with configured genome parameters
    - Running the evolution process
    - Reporting and analyzing final results

    Responsibilities:
    - Coordinate between configuration and domain objects
    - Orchestrate the workflow sequence
    - Handle logging of simulation progress
    - Provide clean separation between UI/entry point and domain logic
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation service.

        Args:
            config: Complete simulation configuration
        """
        self.config = config
        self.population: Population | None = None
        self.evolution_stats: list[PlotData] = []

    def run(self) -> None:
        """
        Execute the complete simulation workflow.

        This is the main entry point for executing a simulation.
        It orchestrates three sequential phases:
        1. Initialize - Create and configure the population
        2. Evolve - Run the evolution process
        3. Report - Display results and statistics
        """
        try:
            self._initialize()
            self._run_evolution()
            self._report_results()
        except Exception as e:
            logger.error("Simulation failed during execution: %s", e)
            raise

    def _initialize(self) -> None:
        """
        Initialize population and configure with genome settings.

        Creates a population with the specified size and mutation rate,
        then initializes all genomes with the configured genome parameters.
        """
        logger.debug("Initializing simulation resources")

        # Create population with configured parameters
        self.population = Population(
            population_size=self.config.population.size,
            mutation_rate=self.config.population.mutation_rate,
            random_seed=self.config.population.random_seed,
        )

        logger.info(
            "Created population: size=%d, mutation_rate=%g",
            self.config.population.size,
            self.config.population.mutation_rate,
        )

        # Initialize population with configured genome parameters
        self.population.initialize_population(
            initial_genome_length=self.config.genome.initial_length,
            nb_coding_segments=self.config.genome.num_coding_segments,
            length_coding_segments=self.config.genome.coding_segment_length,
            length_non_coding_segments=self.config.genome.non_coding_segment_length,
            promoter_directions=self.config.genome.promoter_direction,
            is_circular=self.config.genome.is_circular,
        )

        logger.info(
            "Initialized %d genomes with configured parameters",
            len(self.population.genomes),
        )

    def _run_evolution(self) -> None:
        """
        Execute the evolution process.

        Runs the population through the configured number of generations
        with optional plotting enabled based on configuration.

        Raises:
            RuntimeError: If population was not initialized
        """
        if self.population is None:
            raise RuntimeError("Population not initialized. Call _initialize() first.")

        logger.info("\n=== Starting Evolution ===")
        logger.info(
            "Configuration: %d generations, report every %d",
            self.config.evolution.num_generations,
            self.config.evolution.report_interval,
        )

        # Run evolution with plotting if enabled
        self.evolution_stats = self.population.evolve(
            num_generations=self.config.evolution.num_generations,
            report_every=self.config.evolution.report_interval,
            plot_update_interval=(
                self.config.evolution.report_interval
                if self.config.evolution.enable_plotting
                else None
            ),
            plot_filename=self.config.evolution.plot_filename,
        )

        logger.info("Evolution completed successfully")

    def _report_results(self) -> None:
        """
        Report and display final simulation results.

        Displays:
        - Final population statistics
        - Genome diversity metrics
        - Summary information
        """
        if self.population is None or not self.evolution_stats:
            logger.warning("No evolution statistics available to report")
            return

        logger.info("\n=== Simulation Results ===")

        # Display final statistics
        final_stats: PopulationStats = self.evolution_stats[-1].stats
        logger.info("Final Statistics: %s", final_stats)

        # Display diversity metrics
        diversity = self.population.get_genome_diversity()
        logger.info("Genome Diversity Metrics:")
        logger.info("  Length Diversity: %.3f", diversity["length_diversity"])

        logger.info("=== Simulation Complete ===")
