import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection
from emergents.mutations.duplication import Duplication


def make_genome(circular=False):
    """
    Helper to create a genome with coding and noncoding segments
    Genome structure:
    [NonCodingSegment(3) | CodingSegment(4) | NonCodingSegment(2)]
    0                    3                  7                    9

    :param circular: Whether the genome is circular
    :type circular: bool
    :return: Genome instance
    """
    return Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4, promoter_direction=PromoterDirection.FORWARD),
            NonCodingSegment(length=2),
        ],
        circular=circular,
    )


def test_duplication_init():
    d = Duplication(1, 2, 3)
    assert d.start_pos == 1
    assert d.end_pos == 2
    assert d.insertion_pos == 3
    with pytest.raises(ValueError, match="Positions must be non-negative"):
        Duplication(-1, 2, 3)
    with pytest.raises(ValueError, match="Positions must be non-negative"):
        Duplication(1, -2, 3)
    with pytest.raises(ValueError, match="Positions must be non-negative"):
        Duplication(1, 2, -3)


def test_duplication_get_length():
    g = make_genome()
    d = Duplication(1, 2, 3)
    assert d.get_length(g) == 2
    d = Duplication(6, 2, 3)
    g = make_genome(circular=True)
    assert d.get_length(g) == 5
    # Should raise if genome not provided on first call
    d = Duplication(1, 2, 3)
    d.length = None
    with pytest.raises(AttributeError):
        d.get_length()


def test_duplication_intervals_for_dup():
    g = make_genome()
    d = Duplication(1, 2, 3)
    assert d._intervals_for_dup(g) == [(1, 2)]
    g = make_genome(circular=True)
    d = Duplication(6, 2, 3)
    assert d._intervals_for_dup(g) == [(6, 9), (0, 2)]
    g = make_genome()
    d = Duplication(6, 2, 3)
    with pytest.raises(
        ValueError,
        match="Non-circular genome: duplication start must be < end",
    ):
        d._intervals_for_dup(g)


def test_duplication_is_neutral():
    g = make_genome()
    # Insert in noncoding at start
    d = Duplication(0, 1, 0)
    assert d.is_neutral(g)
    d = Duplication(0, 1, 3)
    assert d.is_neutral(g)
    # Insert in coding at offset (should not be neutral)
    d = Duplication(0, 1, 4)
    assert not d.is_neutral(g)
    # Out of bounds
    d = Duplication(0, 100, 0)
    with pytest.raises(IndexError):
        d.is_neutral(g)


def test_duplication_reverse_strand_promoter():
    g = Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4, promoter_direction=PromoterDirection.REVERSE),
            NonCodingSegment(length=2),
        ]
    )
    # Duplication includes end of coding segment (should not be neutral)
    d = Duplication(3, 6, 0)
    assert not d.is_neutral(g)


def test_duplication_insertion_at_coding_offset():
    g = make_genome()
    # Insert into coding segment at offset 1 (not at start, not at end)
    d = Duplication(0, 1, 4)
    assert not d.is_neutral(g)


def test_duplication_is_neutral_circular_two_intervals():
    g = make_genome(circular=True)
    # Both intervals neutral
    d = Duplication(7, 1, 0)
    assert d.is_neutral(g)
    # First interval not neutral
    d = Duplication(3, 1, 0)
    assert not d.is_neutral(g)
    # Second interval not neutral
    d = Duplication(7, 4, 0)
    assert not d.is_neutral(g)


def test_duplication_apply_and_describe():
    g = make_genome()
    d = Duplication(0, 1, 0)
    d.apply(g)
    assert len(g) == 11
    assert (
        "Duplication(duplicated segment start=0, duplicated segment end=1, length=2)"
        in d.describe(g)
    )
