import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment
from emergents.mutations.small_insertion import SmallInsertion


def make_genome():
    return Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4),
            NonCodingSegment(length=2),
        ]
    )


def test_small_insertion_init():
    assert SmallInsertion(0, 1).position == 0
    assert SmallInsertion(0, 1).length == 1
    with pytest.raises(ValueError, match="Position must be non-negative"):
        SmallInsertion(-1, 1)
    with pytest.raises(ValueError, match="Length must be positive"):
        SmallInsertion(0, 0)
    with pytest.raises(ValueError, match="Length must be positive"):
        SmallInsertion(0, -1)


def test_small_insertion_is_neutral():
    g = make_genome()
    # Insert at end (should be neutral)
    m = SmallInsertion(len(g), 1)
    assert m.is_neutral(g)
    # Insert in noncoding (should be neutral)
    m = SmallInsertion(1, 1)
    assert m.is_neutral(g)
    # Insert at start of coding (should be neutral)
    m = SmallInsertion(3, 1)
    assert m.is_neutral(g)
    # Insert inside coding (should not be neutral)
    m = SmallInsertion(4, 1)
    assert not m.is_neutral(g)
    # Out of bounds
    m = SmallInsertion(100, 1)
    with pytest.raises(IndexError):
        m.is_neutral(g)


def test_small_insertion_apply_and_describe():
    g = make_genome()
    m = SmallInsertion(0, 2)
    m.apply(g)
    assert len(g) == 11  # 9+2
    m = SmallInsertion(len(g), 1)
    m.apply(g)
    assert len(g) == 12
    assert "SmallInsertion(position=11, length=1)" in m.describe()
