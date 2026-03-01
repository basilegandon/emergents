"""
Configuration management for evolution simulation.

This module provides centralized configuration management with validation,
default values, and type safety for all simulation parameters.
"""

from dataclasses import dataclass, field

from emergents.genome.segments import PromoterDirection
from emergents.mutations.base import Mutation
from emergents.mutations.deletion import Deletion
from emergents.mutations.duplication import Duplication
from emergents.mutations.inversion import Inversion
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion


@dataclass
class GenomeConfig:
    """Configuration for genome initialization."""

    initial_length: int = 1000
    num_coding_segments: int = 10

    coding_segment_length: int = 10
    non_coding_segment_length: int = 90
    promoter_direction: PromoterDirection = PromoterDirection.FORWARD
    # is_circular: bool = False
    is_circular: bool = True
    nature_of_extremities: str = "NC--NC"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.initial_length <= 0:
            raise ValueError("Initial length must be positive")
        if self.num_coding_segments < 0:
            raise ValueError("Number of coding segments cannot be negative")


@dataclass
class PopulationConfig:
    """Configuration for population parameters."""

    size: int = 1000
    mutation_rate: float = 1e-4
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.size <= 0:
            raise ValueError("Population size must be positive")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")


@dataclass
class EvolutionConfig:
    """Configuration for evolution simulation."""

    num_generations: int = 10000
    report_interval: int = 50
    enable_progress_bar: bool = True
    enable_plotting: bool = True
    plot_filename: str = "evolution_progress.png"  # Filename for saved plots

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_generations <= 0:
            raise ValueError("Number of generations must be positive")
        if self.report_interval <= 0:
            raise ValueError("Report interval must be positive")


@dataclass
class MutationConfig:
    """Configuration for mutation parameters."""

    weights: dict[type[Mutation], float] = field(
        default_factory=lambda: {
            PointMutation: 1.0,
            SmallDeletion: 1.0,
            SmallInsertion: 1.0,
            Deletion: 1.0,
            Duplication: 1.0,
            Inversion: 1.0,
        }
    )

    seed: int = 42

    # Mutation-specific parameters
    small_mutation_max_size: int = 6

    def __post_init__(self) -> None:
        """Validate and normalize mutation weights."""
        if not self.weights:
            raise ValueError("Mutation weights cannot be empty")

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            raise ValueError("Total mutation weight must be positive")

        self.weights = {
            mut_type: weight / total_weight for mut_type, weight in self.weights.items()
        }

        if self.small_mutation_max_size <= 0:
            raise ValueError("Small mutation max size must be positive")


@dataclass
class SimulationConfig:
    """Main configuration class combining all simulation parameters."""

    genome: GenomeConfig = field(default_factory=GenomeConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    mutations: MutationConfig = field(default_factory=MutationConfig)

    @classmethod
    def create_default(cls) -> "SimulationConfig":
        """Create a configuration with all default values."""
        return cls()

    def validate_all(self) -> None:
        """Validate all configuration components."""
        # Post-init validation is automatically called for dataclasses
        pass

    @classmethod
    def from_file(cls, path: str) -> "SimulationConfig":
        """Load config from YAML/JSON file"""
        pass

    @classmethod
    def from_env(cls) -> "SimulationConfig":
        """Load from environment variables"""
        pass
