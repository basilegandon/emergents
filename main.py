"""
Evolution simulation demonstrating population dynamics.
"""

import sys

from emergents.config import SimulationConfig
from emergents.file_plotter import PlotData
from emergents.logging_config import get_logger
from emergents.population import Population, PopulationStats

# Get logger using the centralized logging configuration
logger = get_logger(__name__)


def cleanup_resources() -> None:
    """Clean up all system resources to ensure proper shutdown."""
    try:
        # Clean up matplotlib
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.ioff()  # Turn off interactive mode

        # Clean up any remaining multiprocessing resources
        import multiprocessing as mp

        for p in mp.active_children():
            if p.is_alive():
                logger.warning("Terminating remaining process: %s", p.name)
                p.terminate()
                p.join(timeout=2.0)
                if p.is_alive():
                    logger.error("Failed to terminate process: %s", p.name)

        # Force garbage collection
        import gc

        gc.collect()

        logger.info("Resource cleanup completed")

    except Exception as e:
        logger.warning("Error during resource cleanup: %s", e)


def run() -> None:
    """Run with configuration management."""
    logger.info("Starting evolution simulation run")

    try:
        # Create configuration
        config = SimulationConfig.create_default()

        logger.info("\n=== Evolution Simulation ===")
        logger.info(
            "Configuration: %d genomes, %d generations",
            config.population.size,
            config.evolution.num_generations,
        )

        # Create and configure population
        population = Population(
            population_size=config.population.size,
            mutation_rate=config.population.mutation_rate,
            random_seed=config.population.random_seed,
        )

        # Initialize population with configured genome
        population.initialize_population(
            initial_genome_length=config.genome.initial_length,
            nb_coding_segments=config.genome.num_coding_segments,
            length_coding_segments=config.genome.coding_segment_length,
            length_non_coding_segments=config.genome.non_coding_segment_length,
            promoter_directions=config.genome.promoter_direction,
            is_circular=config.genome.is_circular,
        )

        logger.info("Initialized population with %d genomes", len(population.genomes))

        # Run evolution with plotting if enabled
        evolution_stats: list[PlotData] = population.evolve(
            num_generations=config.evolution.num_generations,
            report_every=config.evolution.report_interval,
            plot_update_interval=(
                config.evolution.report_interval
                if config.evolution.enable_plotting
                else None
            ),
            plot_filename=config.evolution.plot_filename,
        )

        # Display final results
        if evolution_stats:
            final_stats: PopulationStats = evolution_stats[-1].stats
            logger.info("Final results: %s", final_stats)

            diversity = population.get_genome_diversity()
            logger.info("Final diversity: %.3f", diversity["length_diversity"])

        logger.info("Comprehensive demonstration completed successfully")

    except Exception as e:
        logger.error("Error in comprehensive demo: %s", e)
        raise


def main() -> None:
    """Main entry point with comprehensive error handling."""
    try:
        logger.info("Application started")

        run()

        logger.info("Application completed successfully")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)
    finally:
        # Ensure complete cleanup of all resources
        cleanup_resources()


if __name__ == "__main__":
    main()
    logger.info("Program exited cleanly")
