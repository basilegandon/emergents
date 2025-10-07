"""
Evolution simulation demonstrating optimized population dynamics.

This module showcases the new evolution engine with clean architecture,
efficient algorithms, and comprehensive analysis tools.
"""

import logging
import sys

import matplotlib.pyplot as plt

from emergents.config import SimulationConfig
from emergents.genome.segments import PromoterDirection
from emergents.population import Population, PopulationStats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evolution_simulation.log"),
    ],
)
logger = logging.getLogger(__name__)


def demo_population_evolution() -> None:
    """Demo the new population evolution system with comprehensive error handling."""
    logger.info("Starting population evolution demonstration")

    try:
        print("\n=== Population Evolution Demo ===")

        # Create a population with validated parameters
        population = Population(population_size=100, mutation_rate=1e-3, random_seed=42)

        # Initialize population to a copied single genome
        initial_genome_length = 100
        nb_coding_segments = 2
        length_coding_segments = 25
        length_non_coding_segments = 25
        promoter_directions = PromoterDirection.FORWARD

        population.initialize_population(
            initial_genome_length=initial_genome_length,
            nb_coding_segments=nb_coding_segments,
            length_coding_segments=length_coding_segments,
            length_non_coding_segments=length_non_coding_segments,
            promoter_directions=promoter_directions,
            is_circular=True,
        )

        print(f"Created population with {len(population.genomes)} genomes")
        logger.info(f"Initialized population with {len(population.genomes)} genomes")

        # Show initial stats
        initial_stats = population.get_population_stats(
            current_pop_size=len(population.genomes)
        )
        print(f"Initial stats: {initial_stats}")

        # Run evolution for a few generations
        print("\nRunning evolution...")
        evolution_stats: list[PopulationStats] = population.evolve(
            num_generations=10000, report_every=1000
        )

        # Show final diversity
        diversity = population.get_genome_diversity()
        print("\nFinal diversity metrics:")
        print(f"  Length diversity: {diversity['length_diversity']:.3f}")
        print(f"  Length std dev: {diversity['length_std']:.1f}")

        # Plot results if matplotlib is available
        try:
            plot_evolution_results(evolution_stats)
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")

        logger.info("Population evolution demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error in population evolution demo: {e}")
        raise


def plot_evolution_results(stats: list[PopulationStats]) -> None:
    """Create plots showing evolution results."""
    try:
        avg_lengths = [stat.avg_genome_length for stat in stats]
        generations = list(range(len(avg_lengths)))

        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_lengths, marker="o", markersize=3, linewidth=1)
        plt.title("Average Genome Length Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Average Genome Length")
        plt.grid(True, alpha=0.3)

        # Add some styling
        plt.tight_layout()

        try:
            plt.show()
        except Exception as e:
            logger.warning(f"Could not display plot: {e}")
            # Try to save instead
            plt.savefig("evolution_results.png", dpi=150, bbox_inches="tight")
            print("Plot saved as 'evolution_results.png'")
    except Exception as e:
        logger.error(f"Error in plotting: {e}")


def run_comprehensive_demo() -> None:
    """Run a comprehensive evolution demo with configuration management."""
    logger.info("Starting comprehensive evolution demonstration")

    try:
        # Create configuration
        config = SimulationConfig.create_default()

        print("\n=== Comprehensive Evolution Demo ===")
        print(
            f"Configuration: {config.population.size} genomes, "
            f"{config.evolution.num_generations} generations"
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

        logger.info(f"Initialized population with {len(population.genomes)} genomes")

        # Run evolution
        evolution_stats = population.evolve(
            num_generations=config.evolution.num_generations,
            report_every=config.evolution.report_interval,
        )

        # Display final results
        if evolution_stats:
            final_stats = evolution_stats[-1]
            print(f"\nFinal Results: {final_stats}")

            diversity = population.get_genome_diversity()
            print(f"Final diversity: {diversity['length_diversity']:.3f}")

            if config.evolution.enable_plotting:
                plot_evolution_results(evolution_stats)

        logger.info("Comprehensive demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error in comprehensive demo: {e}")
        raise


def main() -> None:
    """Main entry point with comprehensive error handling."""
    try:
        print("Starting Evolution Simulation")
        logger.info("Application started")

        demo_population_evolution()

        # Run comprehensive demo
        run_comprehensive_demo()

        logger.info("Application completed successfully")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nSimulation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
