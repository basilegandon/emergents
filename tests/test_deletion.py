import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment
from emergents.mutations.deletion import Deletion


def make_genome(circular=False):
    return Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4),
            NonCodingSegment(length=2),
        ],
        circular=circular,
    )


def test_deletion_init():
    d = Deletion(1, 2)
    assert d.start_pos == 1
    assert d.end_pos == 2


def test_deletion_intervals_for_del():
    g = make_genome()
    d = Deletion(1, 2)
    assert d._intervals_for_del(g) == [(1, 2)]
    g = make_genome(circular=True)
    d = Deletion(6, 2)
    assert d._intervals_for_del(g) == [(6, 8), (0, 2)]
    g = make_genome()
    d = Deletion(6, 2)
    with pytest.raises(
        ValueError, match="Non-circular genome: deletion start must be < end"
    ):
        d._intervals_for_del(g)


def test_deletion_is_neutral():
    g = make_genome()
    # Entirely within first noncoding
    d = Deletion(0, 1)
    assert d.is_neutral(g)
    # Spans into coding
    d = Deletion(2, 4)
    assert not d.is_neutral(g)
    # Spans two noncoding segments (should not be neutral)
    d = Deletion(0, 8)
    assert not d.is_neutral(g)
    # Out of bounds
    d = Deletion(0, 100)
    with pytest.raises(IndexError):
        d.is_neutral(g)


def test_deletion_is_neutral_circular_two_intervals():
    g = make_genome(circular=True)
    # Both intervals neutral
    d = Deletion(7, 1)
    assert d.is_neutral(g)
    # First interval not neutral
    d = Deletion(3, 1)
    assert not d.is_neutral(g)
    # Second interval not neutral
    d = Deletion(7, 4)
    assert not d.is_neutral(g)


def test_deletion_apply_and_describe():
    g = make_genome()
    d = Deletion(0, 1)
    d.apply(g)
    assert len(g) == 8
    d = Deletion(0, 8)
    d.apply(g)
    assert len(g) == 0
    assert "Deletion(start_pos=0, end_pos=8)" in d.describe()
