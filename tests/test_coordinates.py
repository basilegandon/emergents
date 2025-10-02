"""
Comprehensive unit tests for emergents.genome.coordinates module.
Tests coordinate validation, edge cases, and error handling.
"""

import pytest

from emergents.genome.coordinates import CoordinateSystem, DefaultCoordinateValidator


class TestCoordinateSystem:
    """Test CoordinateSystem enum."""

    def test_coordinate_system_values(self):
        """Test that coordinate system enum has expected values."""
        assert CoordinateSystem.BASE.value == "base"
        assert CoordinateSystem.GAP.value == "gap"

    def test_coordinate_system_equality(self):
        """Test coordinate system equality."""
        assert CoordinateSystem.BASE == CoordinateSystem.BASE
        assert CoordinateSystem.GAP == CoordinateSystem.GAP
        assert CoordinateSystem.BASE != CoordinateSystem.GAP


class TestDefaultCoordinateValidator:
    """Test DefaultCoordinateValidator implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DefaultCoordinateValidator()

    # Test BASE coordinate validation
    def test_validate_base_position_valid(self):
        """Test valid BASE coordinate positions."""
        # Empty genome - no valid positions in BASE coordinates
        with pytest.raises(IndexError):
            self.validator.validate_position(0, 0, CoordinateSystem.BASE)

        # Non-empty genome
        genome_length = 10
        # Valid positions: 0 to 9
        for pos in range(genome_length):
            self.validator.validate_position(pos, genome_length, CoordinateSystem.BASE)

    def test_validate_base_position_edges(self):
        """Test edge BASE coordinate positions."""
        with pytest.raises(IndexError, match="Base position 10 out of bounds"):
            self.validator.validate_position(10, 10, CoordinateSystem.BASE)

        with pytest.raises(IndexError, match="Base position -1 out of bounds"):
            self.validator.validate_position(-1, 10, CoordinateSystem.BASE)

    # Test GAP coordinate validation
    def test_validate_gap_position_valid(self):
        """Test valid GAP coordinate positions."""
        # Empty genome - only position 0 is valid
        self.validator.validate_position(0, 0, CoordinateSystem.GAP)

        # Non-empty genome
        genome_length = 10
        # Valid positions: 0 to 10 (inclusive)
        for pos in range(genome_length + 1):
            self.validator.validate_position(pos, genome_length, CoordinateSystem.GAP)

    def test_validate_gap_position_edges(self):
        """Test edge GAP coordinate positions."""
        with pytest.raises(IndexError, match="Gap position -1 out of bounds"):
            self.validator.validate_position(-1, 10, CoordinateSystem.GAP)

        with pytest.raises(IndexError, match="Gap position 11 out of bounds"):
            self.validator.validate_position(11, 10, CoordinateSystem.GAP)

    # Test base range validation
    def test_validate_base_range_valid(self):
        """Test valid base ranges."""
        genome_length = 10

        # Non-empty ranges
        self.validator.validate_base_range(0, 1, genome_length)  # Single base at start

        # Multi-element ranges
        self.validator.validate_base_range(0, 5, genome_length)
        self.validator.validate_base_range(3, 8, genome_length)
        self.validator.validate_base_range(0, 10, genome_length)  # Entire genome

    def test_validate_base_range_invalid_start(self):
        """Test base ranges with invalid start position."""
        genome_length = 10

        with pytest.raises(IndexError, match="Base range \\[-1, 5\\) out of bounds"):
            self.validator.validate_base_range(-1, 5, genome_length)

    def test_validate_base_range_invalid_end(self):
        """Test base ranges with invalid end position."""
        genome_length = 10

        with pytest.raises(IndexError, match="Base range \\[0, 11\\) out of bounds"):
            self.validator.validate_base_range(0, 11, genome_length)

    def test_validate_base_range_start_greater_than_end(self):
        """Test base ranges where start >= end."""
        genome_length = 10

        with pytest.raises(
            ValueError, match="Invalid range: start \\(5\\) must be < end \\(3\\)"
        ):
            self.validator.validate_base_range(5, 3, genome_length)

        with pytest.raises(
            ValueError, match="Invalid range: start \\(7\\) must be < end \\(7\\)"
        ):
            self.validator.validate_base_range(7, 7, genome_length)

    def test_validate_base_range_empty_genome(self):
        """Test base range validation on empty genome."""
        # No valid ranges in empty genome
        with pytest.raises(ValueError):
            self.validator.validate_base_range(0, 0, 0)

        # Any non-empty range should be invalid
        with pytest.raises(IndexError, match="Base range \\[0, 1\\) out of bounds"):
            self.validator.validate_base_range(0, 1, 0)
