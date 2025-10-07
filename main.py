"""
Evolution simulation demonstrating optimized population dynamics.

This module showcases the new evolution engine with clean architecture,
efficient algorithms, and comprehensive analysis tools.
"""

import logging

import matplotlib.pyplot as plt

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.population import Population, PopulationStats

logger = logging.getLogger(__name__)


def demo_single_genome() -> None:
    """Demonstrate basic genome functionality."""
    print("=== Single Genome Demo ===")

    segments: list[Segment] = [
        NonCodingSegment(10),
        CodingSegment(200, promoter_direction=PromoterDirection.FORWARD),
        NonCodingSegment(50),
        CodingSegment(100, promoter_direction=PromoterDirection.FORWARD),
        CodingSegment(150, promoter_direction=PromoterDirection.REVERSE),
        CodingSegment(100, promoter_direction=PromoterDirection.FORWARD),
        NonCodingSegment(100),
    ]
    genome = Genome(segments)
    print(f"Created genome with length: {len(genome)} bp")
    print(f"Genome is circular: {genome.circular}")
    print(f"Number of segments: {len(segments)}")


def demo_population_evolution() -> None:
    """Demo the new population evolution system."""
    print("\n=== Population Evolution Demo ===")

    # Create a population
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

    # Show initial stats
    initial_stats = population.get_population_stats(
        current_pop_size=len(population.genomes)
    )
    print(f"Initial stats: {initial_stats}")

    # Run evolution for a few generations
    print("\nRunning evolution...")
    stats: list[PopulationStats] = population.evolve(
        num_generations=10000, report_every=1000
    )

    # Show final diversity
    diversity = population.get_genome_diversity()
    print("\nFinal diversity metrics:")
    print(f"  Length diversity: {diversity['length_diversity']:.3f}")
    print(f"  Length std dev: {diversity['length_std']:.1f}")

    # Plot average genome length over generations
    avg_lengths = [stat.avg_genome_length for stat in stats]
    generations = list(range(len(avg_lengths)))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_lengths, marker="o")
    plt.title("Average Genome Length Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Genome Length")
    plt.grid(True)
    plt.show()


def main() -> None:
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run both demos
    demo_single_genome()
    demo_population_evolution()


if __name__ == "__main__":
    main()
