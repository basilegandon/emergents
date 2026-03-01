"""
Evolution simulation demonstrating population dynamics.

This module serves as the application entry point, handling:
- Configuration loading
- Service orchestration via SimulationService
- Resource cleanup
- Error handling and logging
"""

import sys

from emergents.config import SimulationConfig
from emergents.logging_config import get_logger
from emergents.simulation_service import SimulationService

# Get logger using the centralized logging configuration
logger = get_logger(__name__)


def cleanup_resources() -> None:
    """
    Clean up all system resources to ensure proper shutdown.

    This function handles cleanup of:
    - Matplotlib resources
    - Multiprocessing processes
    - Garbage collection
    """
    try:
        # Clean up matplotlib
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.ioff()  # type: ignore # Turn off interactive mode

        # Clean up any remaining multiprocessing resources
        import multiprocessing as mp

        for p in mp.active_children():
            if p.is_alive():
                logger.debug("Terminating remaining process: %s", p.name)
                p.terminate()
                p.join(timeout=2.0)
                if p.is_alive():
                    logger.warning("Failed to terminate process: %s", p.name)

        # Force garbage collection
        import gc

        gc.collect()

        logger.debug("Resource cleanup completed")

    except Exception as e:
        logger.warning("Error during resource cleanup: %s", e)


def main() -> None:
    """
    Main entry point with comprehensive error handling.

    Workflow:
    1. Load or create simulation configuration
    2. Create and run simulation service
    3. Handle errors gracefully
    4. Ensure complete cleanup
    """
    try:
        logger.info("Starting evolution simulation")

        # Create configuration with defaults
        config = SimulationConfig.create_default()

        logger.info("\n=== Evolution Simulation ===")
        logger.info(
            "Configuration: %d genomes, %d generations",
            config.population.size,
            config.evolution.num_generations,
        )

        # Create and run simulation service
        simulation = SimulationService(config)
        simulation.run()

        logger.info("Evolution simulation completed successfully")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        sys.exit(1)
    finally:
        # Ensure complete cleanup of all resources
        cleanup_resources()


if __name__ == "__main__":
    main()
