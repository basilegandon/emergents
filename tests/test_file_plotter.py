"""
Comprehensive test suite for file_plotter.py module.

Tests cover the MultiprocessFilePlotter initialization, lifecycle,
data processing, and error handling.
"""

from pathlib import Path

import pytest

from emergents.file_plotter import MultiprocessFilePlotter, PlotData
from emergents.statistics import PopulationStats


@pytest.fixture
def sample_population_stats():
    """Create sample population statistics."""
    return PopulationStats(
        generation=1,
        population_size=10,
        avg_genome_length=50.5,
        min_genome_length=40,
        max_genome_length=60,
        length_std=5.2,
        total_mutations_applied=15,
        neutral_mutations=10,
        non_neutral_mutations=5,
        survivors_after_mutations=10,
    )


@pytest.fixture
def sample_plot_data(sample_population_stats):
    """Create sample plot data."""
    return PlotData(
        stats=sample_population_stats,
        genome_lengths=[45, 50, 55, 48, 52, 51, 49, 50, 53, 54],
    )


@pytest.fixture
def temp_plot_dir(tmp_path):
    """Create a temporary directory for plot files."""
    return str(tmp_path)


class TestMultiprocessFilePlotterInit:
    """Test MultiprocessFilePlotter initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        plotter = MultiprocessFilePlotter()

        assert plotter.filename == "evolution_progress.png"
        assert plotter.save_history is True
        assert plotter.history_dir == Path("evolution_plots")
        assert plotter.title == "Evolution Progress"
        assert plotter.max_queue_size == 100
        assert plotter.data_queue is None
        assert plotter.control_queue is None
        assert plotter.plot_process is None
        assert plotter.is_active is False

    def test_init_creates_history_directory(self, temp_plot_dir):
        """Test that initialization creates history directory when save_history is True."""
        history_path = Path(temp_plot_dir) / "test_history"
        assert not history_path.exists()

        plotter = MultiprocessFilePlotter(
            save_history=True,
            history_dir=str(history_path),
        )

        # Directory should be created during initialization (via mkdir in __init__)
        assert history_path.exists()
        plotter.close()


class TestMultiprocessFilePlotterInitialize:
    """Test plotter initialization and process management."""

    def test_initialize_activates_plotter(self):
        """Test that initialize activates the plotter process."""
        plotter = MultiprocessFilePlotter()
        plotter.initialize()

        try:
            # Test behavior: plotter is active and can process data
            assert plotter.is_active is True
            # Test behavior: process is running
            assert plotter.plot_process is not None
            assert plotter.plot_process.is_alive()
        finally:
            plotter.close()


class TestMultiprocessFilePlotterUpdate:
    """Test data update functionality."""

    def test_update_with_valid_data(self, sample_plot_data):
        """Test sending valid data to the plotter."""
        plotter = MultiprocessFilePlotter()
        plotter.initialize()

        try:
            # Should not raise
            plotter.update([sample_plot_data])
        finally:
            plotter.close()

    def test_update_when_inactive(self, sample_plot_data):
        """Test update when plotter is not active."""
        plotter = MultiprocessFilePlotter()

        # Update without initializing should not raise
        plotter.update([sample_plot_data])
        assert plotter.is_active is False


class TestMultiprocessFilePlotterClose:
    """Test plotter closure and process cleanup."""

    def test_close_terminates_process(self):
        """Test that close properly terminates the plotting process."""
        plotter = MultiprocessFilePlotter()
        plotter.initialize()

        process = plotter.plot_process
        assert process is not None
        assert process.is_alive()

        plotter.close()

        assert plotter.is_active is False
        assert plotter.plot_process is None
        assert plotter.data_queue is None
        assert plotter.control_queue is None

    def test_close_without_initialize(self):
        """Test that close works even if initialize was never called."""
        plotter = MultiprocessFilePlotter()

        # Should not raise even though initialize was never called
        plotter.close()
        assert plotter.is_active is False

    def test_close_is_idempotent_after_init(self):
        """Test that close is idempotent and doesn't raise errors on second call."""
        plotter = MultiprocessFilePlotter()
        plotter.initialize()

        assert plotter.is_active is True
        plotter.close()
        assert plotter.is_active is False

        # Second close should not raise
        plotter.close()
        assert plotter.is_active is False


class TestMultiprocessFilePlotterLifecycle:
    """Test complete lifecycle scenarios."""

    def test_full_lifecycle_init_update_close(self, sample_plot_data, tmp_path):
        """Test complete lifecycle: init -> update -> close with default and custom params."""
        # Default parameters
        plotter = MultiprocessFilePlotter(
            filename="test_evolution.png",
            save_history=False,
        )

        # Initialize
        plotter.initialize()
        assert plotter.is_active is True
        assert plotter.plot_process is not None
        assert plotter.plot_process.is_alive()

        # Update with data
        plotter.update([sample_plot_data])

        # Close
        plotter.close()
        assert plotter.is_active is False
        assert plotter.plot_process is None

        # With history enabled
        history_dir = tmp_path / "history"
        plotter2 = MultiprocessFilePlotter(
            filename=str(tmp_path / "main_plot.png"),
            save_history=True,
            history_dir=str(history_dir),
        )

        try:
            plotter2.initialize()
            assert history_dir.exists()

            plotter2.update([sample_plot_data])

        finally:
            plotter2.close()

    def test_no_operations_after_close(self, sample_plot_data):
        """Test that operations after close are handled gracefully."""
        plotter = MultiprocessFilePlotter()
        plotter.initialize()
        plotter.close()

        # Update after close should be handled gracefully (not raise)
        plotter.update([sample_plot_data])

        # Close again should also be fine
        plotter.close()


class TestMultiprocessFilePlotterIntegration:
    """Integration tests for common usage patterns."""

    def test_evolution_simulation_pattern(self, sample_population_stats):
        """Test typical usage pattern in evolution simulation."""
        plotter = MultiprocessFilePlotter(
            filename="evolution_test.png",
            save_history=False,
        )

        plotter.initialize()

        try:
            # Simulate multiple generations
            for gen in range(5):
                stats = PopulationStats(
                    generation=gen + 1,
                    population_size=10,
                    avg_genome_length=50.0 + gen,
                    min_genome_length=40,
                    max_genome_length=60 + gen,
                    length_std=5.0,
                    total_mutations_applied=10 + gen * 2,
                    neutral_mutations=7 + gen,
                    non_neutral_mutations=3 + gen,
                    survivors_after_mutations=10,
                )

                plot_data = PlotData(
                    stats=stats,
                    genome_lengths=[45 + i + gen for i in range(10)],
                )

                plotter.update([plot_data])

            assert plotter.is_active is True
        finally:
            plotter.close()
