"""
File-based plotting module for evolution simulation.

This module provides reliable file-based plotting that avoids the issues
with interactive matplotlib windows becoming unresponsive.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

from emergents.logging_config import get_logger  # noqa: E402
from emergents.statistics import PopulationStats  # noqa: E402

logger = get_logger(__name__)


@dataclass
class PlotData:
    """Data structure for enhanced plotting with histogram support."""

    stats: PopulationStats
    genome_lengths: list[int]


class MultiprocessFilePlotter:
    """Multiprocessing-based enhanced file plotter with histogram support."""

    def __init__(
        self,
        filename: str = "evolution_progress.png",
        save_history: bool = True,
        history_dir: str = "evolution_plots",
        title: str = "Evolution Progress",
        max_queue_size: int = 100,
    ):
        """
        Initialize the multiprocess file plotter.

        Args:
            filename: Name of the main file to save plots to
            save_history: Whether to save timestamped versions
            history_dir: Directory to save historical plots
            title: Title for the plot
            max_queue_size: Maximum size of the data queue
        """
        self.filename = filename
        self.save_history = save_history
        self.history_dir = Path(history_dir)
        self.title = title
        self.max_queue_size = max_queue_size
        self.data_queue: (
            mp.Queue[dict[str, list[PopulationStats] | list[int] | int]] | None
        ) = None
        self.control_queue: mp.Queue[str] | None = None
        self.plot_process: mp.Process | None = None
        self.is_active = False

        # Create history directory if needed
        if self.save_history:
            self.history_dir.mkdir(exist_ok=True)

    def initialize(self) -> None:
        """Initialize the plotting process."""
        try:
            # Create queues for communication
            self.data_queue = mp.Queue(maxsize=self.max_queue_size)
            self.control_queue = mp.Queue()

            # Start the plotting process
            self.plot_process = mp.Process(
                target=self._plot_worker,
                args=(
                    self.data_queue,
                    self.control_queue,
                    self.filename,
                    self.save_history,
                    str(self.history_dir),
                    self.title,
                ),
            )
            self.plot_process.start()
            self.is_active = True

            logger.debug(
                "Multiprocess file plotter initialized - plots will be saved to '%s'",
                self.filename,
            )
            if self.save_history:
                logger.debug(
                    "Historical plots will be saved to '%s/'", self.history_dir
                )

        except Exception as e:
            logger.error("Failed to initialize multiprocess file plotter: %s", e)
            self.close()

    def update(self, data: list[PlotData]) -> None:
        """Send new statistics to the plotting process."""
        if not self.is_active or not self.data_queue:
            return

        try:
            if hasattr(data, "__len__") and len(data) > 0:
                stats = [item.stats for item in data]
                genome_lengths = data[-1].genome_lengths if data else []
            else:
                return

            if not stats:
                return

            # Prepare plot data for worker process
            plot_data: dict[str, list[PopulationStats] | list[int] | int] = {
                "stats": stats,
                "genome_lengths": genome_lengths,
                "generation": len(stats),
            }

            # Try to send data without blocking
            try:
                self.data_queue.put_nowait(plot_data)
            except queue.Full:
                # If queue is full, remove old data and add new
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(plot_data)
                except queue.Empty:
                    pass

        except Exception as e:
            logger.warning("Error sending data to multiprocess plotter: %s", e)

    def _send_stop_signal(self) -> None:
        """Send stop signal to worker process."""
        if self.control_queue:
            try:
                self.control_queue.put("STOP")
            except Exception as e:
                logger.warning("Error sending STOP signal: %s", e)

    def _terminate_process(self) -> None:
        """Terminate the plot process with escalating force."""
        if not self.plot_process or not self.plot_process.is_alive():
            return

        self.plot_process.join(timeout=3.0)
        if self.plot_process.is_alive():
            logger.debug("Forcefully terminating plot process")
            self.plot_process.terminate()
            self.plot_process.join(timeout=2.0)

        if self.plot_process.is_alive():
            self._force_kill_process()
        else:
            logger.debug("Multiprocess file plotter process terminated gracefully")

    def _force_kill_process(self) -> None:
        """Force kill the process using OS-level signals."""
        if not self.plot_process or not self.plot_process.pid:
            return

        logger.debug("Process still alive after terminate, using kill")
        try:
            import os

            if hasattr(os, "kill"):
                os.kill(self.plot_process.pid, 9)
                self.plot_process.join(timeout=1.0)
        except Exception as e:
            logger.error("Failed to kill plot process: %s", e)

    def _clear_queue(self, q: mp.Queue[any]) -> None:  # type: ignore
        """Clear a queue by draining all items."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _close_queue(self, q: mp.Queue[any], queue_name: str) -> None:  # type: ignore
        """Close and clean up a queue safely."""
        try:
            self._clear_queue(q)
            q.close()
        except Exception as e:
            logger.warning("Error closing %s: %s", queue_name, e)

    def close(self) -> None:
        """Close the plotting process safely (idempotent)."""
        if not self.is_active:
            return  # Already closed

        # Mark as inactive immediately to prevent multiple close attempts
        self.is_active = False

        try:
            self._send_stop_signal()
            self._terminate_process()
        except Exception as e:
            logger.error("Error closing multiprocess file plotter: %s", e)
        finally:
            if self.data_queue:
                self._close_queue(self.data_queue, "data queue")

            if self.control_queue:
                self._close_queue(self.control_queue, "control queue")

            # Clean up references
            self.plot_process = None
            self.data_queue = None
            self.control_queue = None

    @staticmethod
    def _create_plots(  # pragma: no cover
        ax1: any,  # type: ignore
        ax2: any,  # type: ignore
        ax3: any,  # type: ignore
        stats: list[PopulationStats],
        genome_lengths: list[int],
        generation: int,
        title: str,
    ) -> None:
        """Create all plot components."""

        MultiprocessFilePlotter._plot_average_length(ax1, stats, title, generation)
        MultiprocessFilePlotter._plot_diversity(ax2, stats)
        MultiprocessFilePlotter._plot_histogram(ax3, genome_lengths, generation)
        MultiprocessFilePlotter._add_statistics_text(ax1, stats, generation)

    @staticmethod
    def _plot_average_length(
        ax1: any,  # type: ignore
        stats: list[PopulationStats],
        title: str,
        generation: int,
    ) -> None:
        """Plot average length with error bars."""
        generations = list(range(len(stats)))
        avg_lengths = [stat.avg_genome_length for stat in stats]
        length_stds = [stat.length_std for stat in stats]

        ax1.plot(
            generations,
            avg_lengths,
            "b-",
            linewidth=2,
            alpha=0.8,
            label="Average Length",
        )
        ax1.fill_between(
            generations,
            [avg - std for avg, std in zip(avg_lengths, length_stds, strict=False)],
            [avg + std for avg, std in zip(avg_lengths, length_stds, strict=False)],
            alpha=0.3,
            color="lightblue",
            label="±1 Std Dev",
        )
        ax1.set_title(
            f"{title} - Generation {generation}",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel("Generation", fontsize=12)
        ax1.set_ylabel("Average Genome Length", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    @staticmethod
    def _plot_diversity(ax2: any, stats: list[PopulationStats]) -> None:  # type: ignore
        """Plot population diversity over time."""
        generations = list(range(len(stats)))
        length_stds = [stat.length_std for stat in stats]

        ax2.plot(
            generations,
            length_stds,
            "-",
            linewidth=2,
            alpha=0.8,
            color="orange",
        )
        ax2.set_xlabel("Generation", fontsize=12)
        ax2.set_ylabel("Length Std Dev", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Population Diversity", fontsize=14)

    @staticmethod
    def _plot_histogram(
        ax3: any,
        genome_lengths: list[int],
        generation: int,  # type: ignore
    ) -> None:
        """Plot genome length histogram."""
        import numpy as np

        if not genome_lengths:
            ax3.text(
                0.5,
                0.5,
                "No genome\nlength data",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightgray",
                    "alpha": 0.8,
                },
            )
        else:
            min_length = min(genome_lengths)
            max_length = max(genome_lengths)
            range_length = max_length - min_length

            if range_length > 0:
                n_bins = min(30, max(10, int(np.sqrt(len(genome_lengths)))))
                bins: list[float] = np.linspace(min_length, max_length, n_bins).tolist()
                ax3.hist(
                    genome_lengths,
                    bins=bins,
                    alpha=0.7,
                    color="green",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax3.axvline(
                    np.mean(genome_lengths).astype(float),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {np.mean(genome_lengths):.1f}",
                )
                ax3.axvline(
                    np.median(genome_lengths).astype(float),
                    color="blue",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {np.median(genome_lengths):.1f}",
                )
            else:
                ax3.bar([min_length], [len(genome_lengths)], alpha=0.7, color="green")
                ax3.axvline(
                    min_length,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Length: {min_length}",
                )

        ax3.set_xlabel("Genome Length", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.set_title(f"Current Distribution (Gen {generation})", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    @staticmethod
    def _add_statistics_text(
        ax1: any,
        stats: list[PopulationStats],
        generation: int,  # type: ignore
    ) -> None:
        """Add statistics text box to plot."""
        if stats:
            current_stats = stats[-1]
            info_text = (
                f"Generation: {generation}\n"
                f"Population: {current_stats.population_size}\n"
                f"Avg Length: {current_stats.avg_genome_length:.1f} ± {current_stats.length_std:.1f}\n"
                f"Range: {current_stats.min_genome_length}-{current_stats.max_genome_length}\n"
                f"Mutations: {current_stats.total_mutations_applied}\n"
                f"Survivors: {current_stats.survivors_after_mutations}"
            )
            ax1.text(
                0.02,
                0.98,
                info_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "wheat",
                    "alpha": 0.8,
                },
            )

    @staticmethod
    def _should_stop(control_queue: mp.Queue[str]) -> bool:
        """Check if a STOP signal was received."""
        try:
            control_msg = control_queue.get_nowait()
            return control_msg == "STOP"
        except queue.Empty:
            return False

    @staticmethod
    def _process_plot_data(  # pragma: no cover
        plot_data: dict[str, list[PopulationStats] | list[int] | int],
        filename: str,
        save_history: bool,
        history_dir: str,
        title: str,
        plot_counter: int,
    ) -> int:
        """Process and save plot data. Returns updated plot_counter."""
        import matplotlib.pyplot as plt

        stats: list[PopulationStats] = plot_data["stats"]  # type: ignore
        genome_lengths: list[int] = plot_data["genome_lengths"]  # type: ignore
        generation: int = plot_data["generation"]  # type: ignore

        plot_counter += 1

        # Create comprehensive plot with histogram
        fig = plt.figure(figsize=(15, 12))

        # Create grid layout: 2 rows, 2 columns
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        # Create all plots
        MultiprocessFilePlotter._create_plots(
            ax1, ax2, ax3, stats, genome_lengths, generation, title
        )

        plt.tight_layout()

        # Save main plot
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")

        # Save historical version if enabled
        if save_history:
            from pathlib import Path

            history_path = Path(history_dir)
            history_filename = history_path / f"evolution_gen_{generation:06d}.png"
            plt.savefig(
                history_filename, dpi=100, bbox_inches="tight", facecolor="white"
            )

        plt.close(fig)  # Important: close figure to free memory

        if plot_counter % 10 == 0:
            logger.debug(
                "Multiprocess plot updated and saved (Generation %d, Update #%d)",
                generation,
                plot_counter,
            )

        return plot_counter

    @staticmethod
    def _plot_worker(  # pragma: no cover
        data_queue: mp.Queue[dict[str, list[PopulationStats] | list[int] | int]],
        control_queue: mp.Queue[str],
        filename: str,
        save_history: bool,
        history_dir: str,
        title: str,
    ) -> None:
        """Worker function for the plotting process."""
        try:
            # Import matplotlib in worker process and use non-interactive backend
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend for file plotting
            import matplotlib.pyplot as plt

            plt.ioff()  # Disable interactive mode
            plot_counter = 0

            logger.debug("Multiprocess file plot worker started")

            while True:
                if MultiprocessFilePlotter._should_stop(control_queue):
                    break

                try:
                    # Get plot data (with timeout)
                    plot_data = data_queue.get(timeout=0.1)
                    plot_counter = MultiprocessFilePlotter._process_plot_data(
                        plot_data,
                        filename,
                        save_history,
                        history_dir,
                        title,
                        plot_counter,
                    )
                except queue.Empty:
                    # No new data, just continue
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.debug("KeyboardInterrupt received in plot worker")
        except Exception as e:
            logger.error("Fatal error in multiprocess plot worker: %s", e)
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception as e:
                logger.error("Error closing plots: %s", e)
            logger.debug("Multiprocess file plot worker stopped")
