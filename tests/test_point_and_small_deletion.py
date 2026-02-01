import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion


# Helper to create a genome with coding and noncoding segments
def make_genome():
    """
    Helper to create a genome with coding and noncoding segments
    Genome structure:
    [NonCodingSegment(3) | CodingSegment(4) | NonCodingSegment(2)]
    0                    3                  7                    9
    """
    return Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4),
            NonCodingSegment(length=2),
        ]
    )


def make_other_genome():
    """
    Helper to create a different genome with coding and noncoding segments
    Genome structure:
    [CodingSegment(5) | NonCodingSegment(3) | CodingSegment(4)]
    0                 5                     8                 12
    """
    return Genome(
        [
            CodingSegment(length=5),
            NonCodingSegment(length=3),
            CodingSegment(length=4),
        ],
        circular=True,
    )


def test_point_mutation_init():
    # Valid
    m = PointMutation(2)
    assert m.position == 2
    # Invalid
    with pytest.raises(ValueError, match="Position must be non-negative"):
        PointMutation(-1)


def test_point_mutation_is_neutral():
    g = make_genome()
    # Noncoding base
    m = PointMutation(1)
    assert m.is_neutral(g)
    # Coding base
    m = PointMutation(3)
    assert not m.is_neutral(g)
    # Out of bounds
    m = PointMutation(100)
    with pytest.raises(IndexError):
        m.is_neutral(g)


def test_point_mutation_apply():
    g = make_genome()
    m = PointMutation(1)
    # Should not raise, even though it's a no-op
    m.apply(g)


def test_point_mutation_describe():
    m = PointMutation(5)
    assert "Point mutation at 5" in m.describe()


def test_small_deletion_init():
    # Valid
    m = SmallDeletion(1, 2)
    assert m.position == 1
    assert m.length == 2
    # Invalid position
    with pytest.raises(ValueError, match="Position must be non-negative"):
        SmallDeletion(-1, 2)
    # Invalid length
    with pytest.raises(ValueError, match="Length must be positive"):
        SmallDeletion(1, 0)
    with pytest.raises(ValueError, match="Length must be positive"):
        SmallDeletion(1, -1)


def test_small_deletion_is_neutral():
    g = make_genome()
    # Entirely within first noncoding
    m = SmallDeletion(0, 2)
    assert m.is_neutral(g)
    # Spans into coding
    m = SmallDeletion(2, 3)
    assert not m.is_neutral(g)
    # Spans two noncoding segments (should not be neutral)
    m = SmallDeletion(0, 6)
    assert not m.is_neutral(g)
    # Out of bounds
    m = SmallDeletion(0, 100)
    with pytest.raises(IndexError):
        m.is_neutral(g)
    g = make_other_genome()
    m = SmallDeletion(0, 3)
    assert not m.is_neutral(g)


def test_small_deletion_apply_and_describe():
    g = make_genome()
    m = SmallDeletion(0, 2)
    m.apply(g)
    assert len(g) == 7  # 9-2
    # Delete all
    m = SmallDeletion(0, 7)
    m.apply(g)
    assert len(g) == 0
    # Describe
    m = SmallDeletion(1, 2)
    assert "SmallDeletion(position=1, length=2)" in m.describe()
    g = make_other_genome()
    m = SmallDeletion(0, 3)
