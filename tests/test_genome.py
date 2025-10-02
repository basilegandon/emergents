"""
Comprehensive unit tests for emergents.genome.genome module.
Tests Genome class with all coordinate systems, operations, and edge cases.
"""

import pytest

from emergents.genome.coordinates import CoordinateSystem, DefaultCoordinateValidator
from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, Segment


class TestGenomeInitialization:
    """Test Genome initialization."""

    def test_empty_genome_initialization(self) -> None:
        """Test creating empty genome."""
        genome = Genome()

        assert genome.length == 0
        assert genome.root is None
        assert not genome.circular
        assert isinstance(genome._validator, DefaultCoordinateValidator)  # type: ignore

    def test_genome_with_segments(self) -> None:
        """Test creating genome with initial segments."""
        segments: list[Segment] = [
            NonCodingSegment(length=10),
            CodingSegment(length=20),
            NonCodingSegment(length=30),
        ]

        genome = Genome(segments=segments)

        assert genome.length == 10 + 20 + 30
        assert genome.root is not None

    def test_circular_genome(self) -> None:
        """Test creating circular genome."""
        genome = Genome(circular=True)
        assert genome.circular

        segments: list[Segment] = [NonCodingSegment(length=100)]
        circular_genome = Genome(segments=segments, circular=True)
        assert circular_genome.circular
        assert circular_genome.length == 100


class TestGenomeFindSegment:
    """Test segment finding methods."""

    def setup_method(self) -> None:
        """Set up test genome."""
        self.segments: list[Segment] = [
            NonCodingSegment(length=20),  # positions 0-19
            CodingSegment(length=30),  # positions 20-49
            NonCodingSegment(length=50),  # positions 50-99
        ]
        self.genome = Genome(segments=self.segments)

    def test_find_segment_at_base_valid_positions(self) -> None:
        """Test finding segments at valid base positions."""
        # First segment
        seg, offset, start, end = self.genome.find_segment_at_position(
            0, CoordinateSystem.BASE
        )
        assert seg == self.segments[0]
        assert offset == 0
        assert start == 0
        assert end == 20

        seg, offset, start, end = self.genome.find_segment_at_position(
            19, CoordinateSystem.BASE
        )
        assert seg == self.segments[0]
        assert offset == 19
        assert start == 0
        assert end == 20

        # Second segment
        seg, offset, start, end = self.genome.find_segment_at_position(
            20, CoordinateSystem.BASE
        )
        assert seg == self.segments[1]
        assert offset == 0
        assert start == 20
        assert end == 50

        seg, offset, start, end = self.genome.find_segment_at_position(
            35, CoordinateSystem.BASE
        )
        assert seg == self.segments[1]
        assert offset == 15
        assert start == 20
        assert end == 50

        # Third segment
        seg, offset, start, end = self.genome.find_segment_at_position(
            99, CoordinateSystem.BASE
        )
        assert seg == self.segments[2]
        assert offset == 49
        assert start == 50
        assert end == 100

    def test_find_segment_at_base_invalid_positions(self) -> None:
        """Test finding segments at invalid base positions."""
        with pytest.raises(IndexError):
            self.genome.find_segment_at_position(-1, CoordinateSystem.BASE)

        with pytest.raises(IndexError):
            self.genome.find_segment_at_position(
                100, CoordinateSystem.BASE
            )  # Length is 100, so max valid is 99

    def test_find_segment_at_position_with_coordinate_systems(self) -> None:
        """Test find_segment_at_position with different coordinate systems."""
        # BASE coordinates
        seg, offset, start, end = self.genome.find_segment_at_position(
            25, CoordinateSystem.BASE
        )
        assert seg == self.segments[1]  # Second segment
        assert offset == 5
        assert start == 20
        assert end == 50

        # GAP coordinates - same position should give same result for interior positions
        seg, offset, start, end = self.genome.find_segment_at_position(
            25, CoordinateSystem.GAP
        )
        assert seg == self.segments[1]
        assert offset == 5
        assert start == 20
        assert end == 50

    def test_find_segment_gap_at_end(self) -> None:
        """Test finding segment at gap position equal to genome length."""
        seg, offset, start, end = self.genome.find_segment_at_position(
            100, CoordinateSystem.GAP
        )
        assert seg == self.segments[2]  # Last segment
        assert offset == 50  # At end of segment
        assert start == 50
        assert end == 100

    def test_find_segment_empty_genome(self) -> None:
        """Test finding segments in empty genome."""
        empty_genome = Genome()

        with pytest.raises(IndexError, match=r"Base position 0 out of bounds \[0, 0\)"):
            empty_genome.find_segment_at_position(0, CoordinateSystem.BASE)

        with pytest.raises(IndexError, match="Cannot access position in empty genome"):
            empty_genome.find_segment_at_position(0, CoordinateSystem.GAP)


class TestGenomeInsertion:
    """Test genome insertion operations."""

    def setup_method(self) -> None:
        """Set up test genome."""
        self.genome = Genome([NonCodingSegment(length=100)])

    def test_insert_at_gap_beginning(self) -> None:
        """Test inserting at beginning of genome."""
        new_segment = NonCodingSegment(length=25)
        self.genome.insert_at_gap(0, new_segment)

        assert self.genome.length == 125
        seg, offset, start, end = self.genome.find_segment_at_position(
            0, CoordinateSystem.BASE
        )
        assert seg.length == 125
        assert offset == 0
        assert start == 0
        assert end == 125

    def test_insert_at_gap_middle(self) -> None:
        """Test inserting in middle of genome."""
        new_segment = NonCodingSegment(length=30)
        self.genome.insert_at_gap(50, new_segment)

        assert self.genome.length == 130
        seg, offset, start, end = self.genome.find_segment_at_position(
            50, CoordinateSystem.BASE
        )
        assert seg.length == 130
        assert offset == 50
        assert start == 0
        assert end == 130

    def test_insert_at_gap_end(self) -> None:
        """Test inserting at end of genome."""
        new_segment = NonCodingSegment(length=15)
        self.genome.insert_at_gap(100, new_segment)

        assert self.genome.length == 115
        seg, offset, start, end = self.genome.find_segment_at_position(
            100, CoordinateSystem.BASE
        )
        assert seg.length == 115
        assert offset == 100
        assert start == 0
        assert end == 115

    def test_insert_invalid_positions(self) -> None:
        """Test inserting at invalid positions."""
        segment = NonCodingSegment(length=10)

        with pytest.raises(IndexError):
            self.genome.insert_at_gap(-1, segment)

        with pytest.raises(IndexError):
            self.genome.insert_at_gap(101, segment)  # Max valid is 100


class TestGenomeDeletion:
    """Test genome deletion operations."""

    def setup_method(self) -> None:
        """Set up test genome."""
        self.segments: list[Segment] = [
            NonCodingSegment(length=20),  # 0-19
            NonCodingSegment(length=30),  # 20-49
            NonCodingSegment(length=50),  # 50-99
        ]
        self.genome = Genome(segments=self.segments)

    def test_delete_base_range_beginning(self) -> None:
        """Test deleting from beginning of genome."""
        self.genome.delete_range(0, 10)

        assert self.genome.length == 90
        # First segment should now be length 10
        seg, offset, start, end = self.genome.find_segment_at_position(
            0, CoordinateSystem.BASE
        )
        assert seg.length == 10
        assert offset == 0
        assert start == 0
        assert end == 10

    def test_delete_base_range_middle(self) -> None:
        """Test deleting from middle of genome."""
        self.genome.delete_range(25, 35)  # Delete 10 bases from second segment

        assert self.genome.length == 90
        seg, offset, start, end = self.genome.find_segment_at_position(
            19, CoordinateSystem.BASE
        )
        assert seg.length == 20
        assert offset == 19
        assert start == 0
        assert end == 20

        seg, offset, start, end = self.genome.find_segment_at_position(
            20, CoordinateSystem.BASE
        )
        assert seg.length == 5
        assert offset == 0
        assert start == 20
        assert end == 25

        seg, offset, start, end = self.genome.find_segment_at_position(
            25, CoordinateSystem.BASE
        )
        assert seg.length == 15
        assert offset == 0
        assert start == 25
        assert end == 40

    def test_delete_base_range_end(self) -> None:
        """Test deleting from end of genome."""
        self.genome.delete_range(90, 100)

        assert self.genome.length == 90
        # Last segment should be shorter
        seg, offset, start, end = self.genome.find_segment_at_position(
            89, CoordinateSystem.BASE
        )
        assert seg.length == 40
        assert offset == 39
        assert start == 50
        assert end == 90

    def test_delete_entire_segments(self) -> None:
        """Test deleting entire segments."""
        # Delete the entire second segment
        self.genome.delete_range(20, 50)

        assert self.genome.length == 70
        # Position 20 should now be in the third segment
        seg, offset, start, end = self.genome.find_segment_at_position(
            20, CoordinateSystem.BASE
        )
        assert seg == self.segments[2]
        assert offset == 0
        assert start == 20
        assert end == 70

    def test_delete_single_base(self) -> None:
        """Test inclusive deletion method using delete_range."""
        original_length = self.genome.length

        # Delete single base
        self.genome.delete_range(50, 51)  # delete_range uses [start, end)
        assert self.genome.length == original_length - 1

    def test_delete_empty_range(self) -> None:
        """Test deleting empty range."""
        original_length = self.genome.length

        self.genome.delete_range(50, 50)  # Empty range
        assert self.genome.length == original_length  # Length should be unchanged

    def test_delete_invalid_ranges(self) -> None:
        """Test deleting invalid ranges."""
        with pytest.raises(IndexError):
            self.genome.delete_range(-1, 10)

        with pytest.raises(IndexError):
            self.genome.delete_range(0, 101)

        with pytest.raises(ValueError):
            self.genome.delete_range(50, 25)  # start > end

    def test_delete_entire_genome(self) -> None:
        """Test deleting entire genome."""
        self.genome.delete_range(0, self.genome.length)
        assert self.genome.length == 0
        assert self.genome.root is None


class TestGenomeExtension:
    """Test genome segment extension operations."""

    def setup_method(self) -> None:
        """Set up test genome."""
        self.segments: list[Segment] = [
            NonCodingSegment(length=50),
            CodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        self.genome = Genome(segments=self.segments)

    def test_extend_noncoding_segment(self) -> None:
        """Test extending non-coding segment using extend_segment_at."""
        original_length = self.genome.length

        # Extend first non-coding segment
        self.genome.extend_segment_at(25, 10)

        assert self.genome.length == original_length + 10
        seg, offset, start, end = self.genome.find_segment_at_position(
            25, CoordinateSystem.BASE
        )
        assert seg.length == 60  # 50 + 10
        assert offset == 25
        assert start == 0
        assert end == 60

    def test_extend_noncoding_segment_at_the_end(self) -> None:
        """Test extending non-coding segment using extend_segment_at."""
        original_length = self.genome.length

        # Extend first non-coding segment
        self.genome.extend_segment_at(100, 10)

        assert self.genome.length == original_length + 10
        seg, offset, start, end = self.genome.find_segment_at_position(
            100, CoordinateSystem.BASE
        )
        assert seg.length == 30  # 20 + 10
        assert offset == 20
        assert start == 80
        assert end == 110

    def test_negative_extent(self) -> None:
        """Test shrinking non-coding segment."""

        with pytest.raises(ValueError, match="delta must be positive"):
            self.genome.extend_segment_at(25, -1)

    def test_extend_coding_segment_error(self) -> None:
        """Test that extending coding segment raises error."""
        with pytest.raises(
            TypeError,
            match="extend_segment_at only allowed on non-coding segments",
        ):
            self.genome.extend_segment_at(60, 10)  # Position 60 is in coding segment

    def test_extend_invalid_positions(self) -> None:
        """Test extending at invalid positions."""
        with pytest.raises(IndexError):
            self.genome.extend_segment_at(-1, 10)

        with pytest.raises(IndexError):
            self.genome.extend_segment_at(101, 10)


class TestGenomeCoalescing:
    """Test genome coalescing operations."""

    def test_coalesce_adjacent_noncoding(self) -> None:
        """Test coalescing adjacent non-coding segments."""
        segments: list[Segment] = [
            NonCodingSegment(length=10),
            NonCodingSegment(length=20),
            CodingSegment(length=15),
            NonCodingSegment(length=5),
            NonCodingSegment(length=25),
        ]
        genome = Genome(segments=segments)

        original_length = genome.length
        genome.coalesce_all()

        # Length should be preserved
        assert genome.length == original_length

        # Should now have fewer segments (coalesced non-coding ones)
        segment_list = genome.to_list()
        assert len(segment_list) == 3  # 30 (coalesced) + 15 (coding) + 30 (coalesced)
        instance_type_list = [type(seg) for seg in segment_list]
        assert instance_type_list == [NonCodingSegment, CodingSegment, NonCodingSegment]
        length_list = [seg.length for seg in segment_list]
        assert length_list == [30, 15, 30]

    def test_coalesce_no_adjacent_noncoding(self) -> None:
        """Test coalescing when no adjacent non-coding segments exist."""
        segments: list[Segment] = [
            NonCodingSegment(length=10),
            CodingSegment(length=20),
            NonCodingSegment(length=30),
        ]
        genome = Genome(segments=segments)

        original_length = genome.length
        original_segments = genome.to_list()

        genome.coalesce_all()

        assert genome.length == original_length
        new_segments = genome.to_list()
        assert len(new_segments) == len(original_segments)
        assert all(
            new_segment == original_segment
            for new_segment, original_segment in zip(new_segments, original_segments)
        )

    def test_coalesce_empty_genome(self) -> None:
        """Test coalescing empty genome."""
        genome = Genome()
        genome.coalesce_all()
        assert genome.length == 0
        assert genome.root is None

    def test_coalesce_single_segment(self) -> None:
        """Test coalescing single segment genome."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        genome.coalesce_all()
        assert genome.length == original_length


class TestGenomeIteration:
    """Test genome iteration methods."""

    def setup_method(self) -> None:
        """Set up test genome."""
        self.segments: list[Segment] = [
            NonCodingSegment(length=10),
            CodingSegment(length=20),
            NonCodingSegment(length=30),
        ]
        self.genome = Genome(segments=self.segments)

    def test_iter_segments(self) -> None:
        """Test iterating over segments."""
        segments_info = list(self.genome.iter_segments())

        assert len(segments_info) == 3

        # Check first segment
        seg, start, end = segments_info[0]
        assert seg.length == 10
        assert start == 0
        assert end == 9

        # Check second segment
        seg, start, end = segments_info[1]
        assert seg.length == 20
        assert start == 10
        assert end == 29

        # Check third segment
        seg, start, end = segments_info[2]
        assert seg.length == 30
        assert start == 30
        assert end == 59

    def test_iter_segments_empty_genome(self) -> None:
        """Test iterating over empty genome."""
        empty_genome = Genome()
        segments_info = list(empty_genome.iter_segments())
        assert segments_info == []

    def test_to_list(self) -> None:
        """Test converting genome to segment list."""
        segment_list = self.genome.to_list()

        assert len(segment_list) == 3
        assert sum(seg.length for seg in segment_list) == self.genome.length


class TestGenomeSpecialMethods:
    """Test genome special methods (__len__, __bool__)."""

    def test_len(self) -> None:
        """Test __len__ method."""
        empty_genome = Genome()
        assert len(empty_genome) == 0

        genome = Genome([NonCodingSegment(length=42)])
        assert len(genome) == 42

        genome = Genome([NonCodingSegment(length=10), CodingSegment(length=20)])
        assert len(genome) == 30

    def test_bool(self) -> None:
        """Test __bool__ method."""
        empty_genome = Genome()
        assert not empty_genome

        non_empty_genome = Genome([NonCodingSegment(length=1)])
        assert non_empty_genome
