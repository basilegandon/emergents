"""
Comprehensive unit tests for emergents.mutations.duplication module.
Tests the Duplication class with all edge cases and error conditions.
"""

import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.duplication import Duplication


class TestDuplicationInitialization:
    """Test Duplication initialization and parameter validation."""

    def test_initialization_basic(self) -> None:
        """Test basic Duplication initialization with valid parameters."""
        mutation = Duplication(start_pos=10, end_pos=50, insertion_pos=75)

        assert mutation.start_pos == 10
        assert mutation.end_pos == 50
        assert mutation.insertion_pos == 75
        assert mutation.rng_state is None

        with pytest.raises(ValueError):
            Duplication(start_pos=-5, end_pos=50, insertion_pos=75)

    def test_initialization_with_rng_state(self) -> None:
        """Test Duplication initialization with RNG state."""
        mutation = Duplication(
            start_pos=25, end_pos=75, insertion_pos=100, rng_state=11111
        )

        assert mutation.start_pos == 25
        assert mutation.end_pos == 75
        assert mutation.insertion_pos == 100
        assert mutation.rng_state == 11111

    def test_initialization_reverse_order(self) -> None:
        """Test Duplication initialization with end < start."""
        mutation = Duplication(start_pos=50, end_pos=25, insertion_pos=75)
        assert mutation.start_pos == 50
        assert mutation.end_pos == 25
        assert mutation.insertion_pos == 75

    def test_initialization_insert_within_duplicated_region(self) -> None:
        """Test Duplication initialization with insert_at within duplicated region."""
        mutation = Duplication(start_pos=10, end_pos=50, insertion_pos=30)
        assert mutation.start_pos == 10
        assert mutation.end_pos == 50
        assert mutation.insertion_pos == 30


class TestDuplicationNeutrality:
    """Test neutrality checking for Duplication across different genome structures."""

    def setup_method(self) -> None:
        """Set up test genomes with various segment configurations."""
        # Create genomes with known segment IDs for testing
        self.noncoding_seg1 = NonCodingSegment(length=30)
        self.coding_seg = CodingSegment(length=40)
        self.noncoding_seg2 = NonCodingSegment(length=30)

        # Genome with mixed segments
        self.mixed_genome = Genome(
            [
                self.noncoding_seg1,  # positions 0-29
                self.coding_seg,  # positions 30-69
                self.noncoding_seg2,  # positions 70-99
            ]
        )

        # Pure non-coding genome
        self.noncoding_genome = Genome([NonCodingSegment(length=100)])

        # Pure coding genome
        self.coding_genome = Genome(
            [CodingSegment(length=100, promoter_direction=PromoterDirection.FORWARD)]
        )

    def test_neutrality_within_single_noncoding_segment(self) -> None:
        """Test that duplications entirely within non-coding segments are neutral."""
        # Duplicate within first non-coding segment, insert into same segment
        mutation = Duplication(start_pos=5, end_pos=15, insertion_pos=20)
        assert mutation.is_neutral(self.mixed_genome)

        # Duplicate within first segment, insert into last non-coding segment
        mutation = Duplication(start_pos=5, end_pos=15, insertion_pos=80)
        assert mutation.is_neutral(self.mixed_genome)

        # Duplicate within last non-coding segment
        mutation = Duplication(start_pos=75, end_pos=85, insertion_pos=90)
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_within_coding_segment(self) -> None:
        """Test neutrality when duplication source is entirely within a coding segment."""
        # Duplicate within coding segment, insert in non-coding - should be neutral
        mutation = Duplication(start_pos=35, end_pos=45, insertion_pos=80)
        assert mutation.is_neutral(self.mixed_genome)

        # Duplicate within coding segment, insert in same coding segment - should not be neutral
        mutation = Duplication(start_pos=35, end_pos=45, insertion_pos=55)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_across_segment_boundaries(self) -> None:
        """Test neutrality when duplication source spans multiple segments."""
        # Duplicate across non-coding and coding boundary (promoter in between)
        mutation = Duplication(start_pos=25, end_pos=35, insertion_pos=80)
        assert not mutation.is_neutral(self.mixed_genome)

        # Duplicate across coding and non-coding boundary (no promoter in between)
        mutation = Duplication(start_pos=65, end_pos=75, insertion_pos=5)
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_insert_position_matters(self) -> None:
        """Test that neutrality depends on both source and insert positions."""
        # Duplicate non-coding region, insert into coding region - should not be neutral
        mutation = Duplication(start_pos=5, end_pos=15, insertion_pos=40)
        assert not mutation.is_neutral(self.mixed_genome)

        # Duplicate non-coding region, insert into non-coding region - should be neutral
        mutation = Duplication(start_pos=5, end_pos=15, insertion_pos=80)
        assert mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_single_base_duplications(self) -> None:
        """Test neutrality for single base duplications."""
        # Single base from non-coding, insert in non-coding
        mutation = Duplication(start_pos=29, end_pos=29, insertion_pos=80)
        assert mutation.is_neutral(self.mixed_genome)

        # Single base from coding, insert in non-coding
        mutation = Duplication(start_pos=31, end_pos=31, insertion_pos=80)
        assert mutation.is_neutral(self.mixed_genome)

        # Promoter base, should not be neutral
        mutation = Duplication(start_pos=30, end_pos=30, insertion_pos=80)
        assert not mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_pure_noncoding_genome(self) -> None:
        """Test neutrality in pure non-coding genome."""
        # All duplications should be neutral (single segment)
        test_cases = [(0, 10, 50), (25, 35, 70), (40, 60, 20)]

        for start, end, insert_at in test_cases:
            mutation = Duplication(
                start_pos=start, end_pos=end, insertion_pos=insert_at
            )
            assert mutation.is_neutral(self.noncoding_genome)

    def test_is_neutral_pure_coding_genome(self) -> None:
        """Test neutrality in pure coding genome."""
        # No duplications should be neutral (single coding segment) except zero-length if inserted at the end or start
        no_neutral_test_cases = [(0, 10, 50), (25, 35, 70), (40, 60, 20)]

        for start, end, insert_at in no_neutral_test_cases:
            mutation = Duplication(
                start_pos=start, end_pos=end, insertion_pos=insert_at
            )
            assert not mutation.is_neutral(self.coding_genome)

        neutral_test_cases = [(1, 10, 0), (25, 35, 0), (40, 60, 100)]

        for start, end, insert_at in neutral_test_cases:
            mutation = Duplication(
                start_pos=start, end_pos=end, insertion_pos=insert_at
            )
            assert mutation.is_neutral(self.coding_genome)

    def test_is_neutral_invalid_positions(self) -> None:
        """Test neutrality checking with invalid positions."""
        # Start beyond genome length
        mutation = Duplication(start_pos=100, end_pos=105, insertion_pos=10)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # End beyond genome length
        mutation = Duplication(start_pos=95, end_pos=105, insertion_pos=10)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

        # Insert position beyond genome length + duplication length
        mutation = Duplication(start_pos=10, end_pos=20, insertion_pos=200)
        with pytest.raises(IndexError):
            mutation.is_neutral(self.mixed_genome)

    def test_is_neutral_empty_genome(self) -> None:
        """Test neutrality checking with empty genome."""
        empty_genome = Genome()
        mutation = Duplication(start_pos=0, end_pos=1, insertion_pos=0)

        with pytest.raises(IndexError):
            mutation.is_neutral(empty_genome)

    def test_is_neutral_reverse_order_positions(self) -> None:
        """Test neutrality with end < start."""
        # Should raise error for invalid order
        mutation = Duplication(start_pos=30, end_pos=25, insertion_pos=75)
        assert not mutation.is_neutral(self.mixed_genome)

        mutation = Duplication(start_pos=31, end_pos=25, insertion_pos=75)
        assert mutation.is_neutral(self.mixed_genome)


class TestDuplicationApplication:
    """Test Duplication application to genomes."""

    def test_apply_basic(self) -> None:
        """Test basic duplication application."""
        genome = Genome([NonCodingSegment(length=100)])
        original_length = genome.length

        mutation = Duplication(
            start_pos=10, end_pos=20, insertion_pos=50
        )  # duplicate 11 bases (positions 10-20 inclusive)
        mutation.apply(genome)

        # Genome should be longer by 11 bases
        assert genome.length == original_length + 11

    def test_apply_at_beginning(self) -> None:
        """Test duplication inserting at beginning of genome."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        mutation = Duplication(start_pos=10, end_pos=20, insertion_pos=0)
        mutation.apply(genome)

        assert genome.length == original_length + 11  # 10-20 inclusive = 11 bases

    def test_apply_at_end(self) -> None:
        """Test duplication inserting at end of genome."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        mutation = Duplication(
            start_pos=10, end_pos=20, insertion_pos=50
        )  # insert at gap after last base, duplicates 11 bases
        mutation.apply(genome)

        assert genome.length == original_length + 11  # 10-20 inclusive = 11 bases

    def test_apply_single_base(self) -> None:
        """Test duplicating single base."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        mutation = Duplication(
            start_pos=25, end_pos=25, insertion_pos=40
        )  # Single base
        mutation.apply(genome)

        assert genome.length == original_length + 1

    def test_apply_entire_segment(self) -> None:
        """Test duplicating entire segment."""
        segments: list[Segment] = [
            NonCodingSegment(length=20),
            CodingSegment(length=30),
            NonCodingSegment(length=20),
        ]
        genome = Genome(segments=segments)
        original_length = genome.length

        # Duplicate the entire last segment (positions 50-69)
        mutation = Duplication(
            start_pos=50, end_pos=69, insertion_pos=70
        )  # insert at end, duplicates 30 bases
        mutation.apply(genome)

        assert genome.length == original_length + 20

    def test_apply_across_segments(self) -> None:
        """Test duplication spanning multiple segments."""
        segments: list[Segment] = [
            NonCodingSegment(length=30),
            CodingSegment(length=40),
            NonCodingSegment(length=30),
        ]
        # Make first two segments have same ID
        genome = Genome(segments=segments)
        original_length = genome.length

        mutation = Duplication(start_pos=10, end_pos=59, insertion_pos=90)  # 50 bases
        mutation.apply(genome)

        assert genome.length == original_length + 50

    def test_apply_reverse_order(self) -> None:
        """Test application with end < start."""
        genome = Genome([NonCodingSegment(length=50)])

        mutation = Duplication(start_pos=30, end_pos=20, insertion_pos=40)
        mutation.apply(genome)
        assert genome.length == 89  # 19 + 20 bases duplicated

    def test_apply_insert_within_duplicated_region_bug(self) -> None:
        """Test the bug where insert_at is within the duplicated region."""
        genome = Genome([NonCodingSegment(length=100)])

        mutation = Duplication(start_pos=20, end_pos=60, insertion_pos=40)
        mutation.apply(genome)
        assert genome.length == 141  # 41 bases duplicated

    def test_apply_duplicate_entire_genome(self) -> None:
        """Test duplicating entire genome."""
        genome = Genome([NonCodingSegment(length=50)])
        original_length = genome.length

        # Duplicate entire genome, insert at end
        mutation = Duplication(start_pos=0, end_pos=49, insertion_pos=50)
        mutation.apply(genome)

        assert genome.length == original_length * 2


class TestDuplicationSerialization:
    """Test Duplication serialization and string representation."""

    def test_serialization_methods_exist(self) -> None:
        """Test that Duplication implements serialization methods."""
        mutation = Duplication(start_pos=10, end_pos=50, insertion_pos=75)
        genome = Genome([NonCodingSegment(length=100)])

        # These methods should exist and return appropriate types
        description = mutation.describe(genome)
        assert isinstance(description, str)
        assert "10" in description
        assert "50" in description

        # Check if serialize is implemented or raises NotImplementedError
        try:
            serialized = mutation.serialize()
            assert isinstance(serialized, dict)
        except NotImplementedError:
            # This is acceptable if not yet implemented
            pass
