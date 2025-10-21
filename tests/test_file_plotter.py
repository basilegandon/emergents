"""
Comprehensive unit tests for emergents.file_plotter module.
Tests PlotData, basic initialization, and data handling without multiprocessing.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from emergents.file_plotter import MultiprocessFilePlotter, PlotData
from emergents.statistics import PopulationStats


class TestPlotData:
    """Test PlotData dataclass."""

    def test_plot_data_creation(self) -> None:
        """Test basic PlotData creation."""
        stats = PopulationStats(
            generation=1,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=900,
            max_genome_length=1100,
            length_std=50.0,
            total_mutations_applied=10,
            neutral_mutations=8,
            non_neutral_mutations=2,
            survivors_after_mutations=98,
        )
        genome_lengths = [950, 1000, 1050, 980, 1020]

        plot_data = PlotData(stats=stats, genome_lengths=genome_lengths)

        assert plot_data.stats == stats
        assert plot_data.genome_lengths == genome_lengths
        assert len(plot_data.genome_lengths) == 5

    def test_plot_data_with_empty_genome_lengths(self) -> None:
        """Test PlotData with empty genome lengths."""
        stats = PopulationStats(
            generation=0,
            population_size=0,
            avg_genome_length=0.0,
            min_genome_length=0,
            max_genome_length=0,
            length_std=0.0,
            total_mutations_applied=0,
            neutral_mutations=0,
            non_neutral_mutations=0,
            survivors_after_mutations=0,
        )

        plot_data = PlotData(stats=stats, genome_lengths=[])

        assert plot_data.stats == stats
        assert plot_data.genome_lengths == []


class TestMultiprocessFilePlotterInit:
    """Test MultiprocessFilePlotter initialization without starting processes."""

    def test_default_initialization(self) -> None:
        """Test default plotter initialization."""
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

    def test_custom_initialization(self) -> None:
        """Test plotter with custom parameters."""
        plotter = MultiprocessFilePlotter(
            filename="custom_plot.png",
            save_history=False,
            history_dir="custom_plots",
            title="Custom Evolution",
            max_queue_size=50,
        )

        assert plotter.filename == "custom_plot.png"
        assert plotter.save_history is False
        assert plotter.history_dir == Path("custom_plots")
        assert plotter.title == "Custom Evolution"
        assert plotter.max_queue_size == 50
        assert plotter.is_active is False

    @patch("emergents.file_plotter.Path.mkdir")
    def test_history_directory_creation(self, mock_mkdir: Mock) -> None:
        """Test that history directory is created when save_history is True."""
        _ = MultiprocessFilePlotter(save_history=True, history_dir="test_plots")

        mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_no_history_directory_creation(self) -> None:
        """Test that history directory is not created when save_history is False."""
        with patch("emergents.file_plotter.Path.mkdir") as mock_mkdir:
            _ = MultiprocessFilePlotter(save_history=False)
            mock_mkdir.assert_not_called()


class TestMultiprocessFilePlotterMethods:
    """Test MultiprocessFilePlotter methods without actual multiprocessing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.plotter = MultiprocessFilePlotter()

    def test_update_when_inactive(self) -> None:
        """Test update method when plotter is not active."""
        stats = PopulationStats(
            generation=1,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=900,
            max_genome_length=1100,
            length_std=50.0,
            total_mutations_applied=10,
            neutral_mutations=8,
            non_neutral_mutations=2,
            survivors_after_mutations=98,
        )
        plot_data = [PlotData(stats=stats, genome_lengths=[1000, 1010, 990])]

        # Should not raise any exception when inactive
        self.plotter.update(plot_data)

        assert self.plotter.is_active is False

    def test_update_with_empty_data(self) -> None:
        """Test update method with empty data."""
        # Should not raise any exception with empty data
        self.plotter.update([])

        assert self.plotter.is_active is False

    def test_close_when_inactive(self) -> None:
        """Test close method when already inactive."""
        # Should not raise any exception when already inactive
        self.plotter.close()

        assert self.plotter.is_active is False

    @patch("emergents.file_plotter.mp.Queue")
    @patch("emergents.file_plotter.mp.Process")
    def test_initialize_success(self, mock_process: Mock, mock_queue: Mock) -> None:
        """Test successful initialization with mocked multiprocessing."""
        # Setup mocks
        mock_data_queue = Mock()
        mock_control_queue = Mock()
        mock_queue.side_effect = [mock_data_queue, mock_control_queue]

        mock_plot_process = Mock()
        mock_process.return_value = mock_plot_process

        # Initialize plotter
        self.plotter.initialize()

        # Verify initialization
        assert self.plotter.is_active is True
        assert self.plotter.data_queue == mock_data_queue
        assert self.plotter.control_queue == mock_control_queue
        assert self.plotter.plot_process == mock_plot_process

        # Verify process was started
        mock_plot_process.start.assert_called_once()

    @patch("emergents.file_plotter.mp.Queue")
    def test_initialize_failure(self, mock_queue: Mock) -> None:
        """Test initialization failure handling."""
        # Make Queue creation fail
        mock_queue.side_effect = Exception("Queue creation failed")

        # Should handle exception gracefully
        self.plotter.initialize()

        assert self.plotter.is_active is False

    @patch("emergents.file_plotter.mp.Queue")
    @patch("emergents.file_plotter.mp.Process")
    def test_update_with_active_plotter(
        self, mock_process: Mock, mock_queue: Mock
    ) -> None:
        """Test update method with active plotter."""
        # Setup mocks
        mock_data_queue = Mock()
        mock_control_queue = Mock()
        mock_queue.side_effect = [mock_data_queue, mock_control_queue]

        mock_plot_process = Mock()
        mock_process.return_value = mock_plot_process

        # Initialize and make active
        self.plotter.initialize()

        # Create test data
        stats = PopulationStats(
            generation=1,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=900,
            max_genome_length=1100,
            length_std=50.0,
            total_mutations_applied=10,
            neutral_mutations=8,
            non_neutral_mutations=2,
            survivors_after_mutations=98,
        )
        plot_data = [PlotData(stats=stats, genome_lengths=[1000, 1010, 990])]

        # Update should try to put data in queue
        self.plotter.update(plot_data)

        # Verify put_nowait was called with correct data structure
        mock_data_queue.put_nowait.assert_called_once()
        call_args = mock_data_queue.put_nowait.call_args[0][0]

        assert "stats" in call_args
        assert "genome_lengths" in call_args
        assert "generation" in call_args
        assert call_args["generation"] == 1
        assert call_args["genome_lengths"] == [1000, 1010, 990]

    @patch("emergents.file_plotter.mp.Queue")
    @patch("emergents.file_plotter.mp.Process")
    def test_update_queue_full_handling(
        self, mock_process: Mock, mock_queue: Mock
    ) -> None:
        """Test update method handling of full queue."""
        import queue as queue_module

        # Setup mocks
        mock_data_queue = Mock()
        mock_control_queue = Mock()
        mock_queue.side_effect = [mock_data_queue, mock_control_queue]

        mock_plot_process = Mock()
        mock_process.return_value = mock_plot_process

        # Make put_nowait raise Full exception first time
        mock_data_queue.put_nowait.side_effect = [
            queue_module.Full(),  # First call fails
            None,  # Second call succeeds
        ]

        # Initialize and make active
        self.plotter.initialize()

        # Create test data
        stats = PopulationStats(
            generation=1,
            population_size=100,
            avg_genome_length=1000.0,
            min_genome_length=900,
            max_genome_length=1100,
            length_std=50.0,
            total_mutations_applied=10,
            neutral_mutations=8,
            non_neutral_mutations=2,
            survivors_after_mutations=98,
        )
        plot_data = [PlotData(stats=stats, genome_lengths=[1000])]

        # Should handle queue full gracefully
        self.plotter.update(plot_data)

        # Should have tried to get old data and put new data
        assert mock_data_queue.put_nowait.call_count == 2
        mock_data_queue.get_nowait.assert_called_once()


class TestMultiprocessFilePlotterEdgeCases:
    """Test edge cases and error conditions."""

    def test_update_with_invalid_data_structure(self) -> None:
        """Test update with data that doesn't have expected structure."""
        plotter = MultiprocessFilePlotter()

        # These should not raise exceptions
        plotter.update(None)  # type: ignore
        plotter.update("invalid")  # type: ignore
        plotter.update(123)  # type: ignore

        assert plotter.is_active is False

    def test_multiple_close_calls(self) -> None:
        """Test that multiple close calls are safe."""
        plotter = MultiprocessFilePlotter()

        # Multiple close calls should be safe
        plotter.close()
        plotter.close()
        plotter.close()

        assert plotter.is_active is False

    @patch("emergents.file_plotter.Path.mkdir")
    def test_history_directory_with_pathlib(self, mock_mkdir: Mock) -> None:
        """Test history directory handling with Path objects."""
        history_path = Path("test/nested/plots")
        plotter = MultiprocessFilePlotter(
            save_history=True, history_dir=str(history_path)
        )

        assert plotter.history_dir == history_path
        mock_mkdir.assert_called_once_with(exist_ok=True)


class TestPlotDataIntegration:
    """Integration tests for PlotData with real statistics."""

    def test_plot_data_with_realistic_stats(self) -> None:
        """Test PlotData with realistic evolution statistics."""
        # Simulate statistics from multiple generations
        stats_gen_1 = PopulationStats(
            generation=1,
            population_size=1000,
            avg_genome_length=10000.0,
            min_genome_length=9500,
            max_genome_length=10500,
            length_std=150.0,
            total_mutations_applied=100,
            neutral_mutations=85,
            non_neutral_mutations=15,
            survivors_after_mutations=985,
        )

        genome_lengths_gen_1 = list(range(9500, 10501, 50))  # 21 genomes

        plot_data_1 = PlotData(stats=stats_gen_1, genome_lengths=genome_lengths_gen_1)

        assert plot_data_1.stats.generation == 1
        assert len(plot_data_1.genome_lengths) == 21
        assert min(plot_data_1.genome_lengths) == 9500
        assert max(plot_data_1.genome_lengths) == 10500

    def test_plot_data_evolution_sequence(self) -> None:
        """Test sequence of PlotData representing evolution over time."""
        plot_data_sequence = []

        for gen in range(5):
            stats = PopulationStats(
                generation=gen,
                population_size=1000,
                avg_genome_length=10000.0 + gen * 10,  # Growing over time
                min_genome_length=9500 + gen * 5,
                max_genome_length=10500 + gen * 15,
                length_std=150.0 - gen * 5,  # Decreasing diversity
                total_mutations_applied=100 + gen * 20,
                neutral_mutations=80 + gen * 15,
                non_neutral_mutations=20 + gen * 5,
                survivors_after_mutations=985 + gen * 2,
            )

            # Simulate genome lengths shifting over time
            genome_lengths = [int(10000 + gen * 10 + i) for i in range(-50, 51, 10)]

            plot_data_sequence.append(
                PlotData(stats=stats, genome_lengths=genome_lengths)
            )

        # Verify evolution trends
        assert len(plot_data_sequence) == 5
        assert (
            plot_data_sequence[0].stats.avg_genome_length
            < plot_data_sequence[4].stats.avg_genome_length
        )
        assert (
            plot_data_sequence[0].stats.length_std
            > plot_data_sequence[4].stats.length_std
        )

        # Verify genome lengths reflect the trends
        avg_len_0 = sum(plot_data_sequence[0].genome_lengths) / len(
            plot_data_sequence[0].genome_lengths
        )
        avg_len_4 = sum(plot_data_sequence[4].genome_lengths) / len(
            plot_data_sequence[4].genome_lengths
        )
        assert avg_len_0 < avg_len_4
