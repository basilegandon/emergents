import pytest

from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection
from emergents.mutations.inversion import Inversion


def make_genome():
    return Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4, promoter_direction=PromoterDirection.FORWARD),
            NonCodingSegment(length=2),
        ]
    )


def test_inversion_init():
    # Valid
    i = Inversion(1, 2)
    assert i.start_pos == 1
    assert i.end_pos == 2
    # Negative
    with pytest.raises(ValueError, match="Inversion positions must be non-negative"):
        Inversion(-1, 2)
    with pytest.raises(ValueError, match="Inversion positions must be non-negative"):
        Inversion(1, -2)
    # Same
    with pytest.raises(
        ValueError, match="Inversion start and end positions cannot be the same"
    ):
        Inversion(2, 2)


def test_inversion_reverted_logic():
    i = Inversion(5, 2)
    assert i.start_pos == 2
    assert i.end_pos == 5
    assert i.reverted is True


def test_inversion_merge_noncoding_left():
    g = Genome(
        [
            NonCodingSegment(length=2),
            CodingSegment(length=1, promoter_direction=PromoterDirection.REVERSE),
            NonCodingSegment(length=3),
            CodingSegment(length=4, promoter_direction=PromoterDirection.FORWARD),
        ]
    )
    i = Inversion(1, 5)
    i.apply(g)
    segs = g.to_list()
    print(segs)
    # Should merge the two noncoding segments
    assert segs[0].length == 3
    assert segs[1].promoter_direction == PromoterDirection.FORWARD
    assert segs[2].length == 2


def test_inversion_merge_noncoding_right():
    g = Genome(
        [
            CodingSegment(length=4, promoter_direction=PromoterDirection.FORWARD),
            NonCodingSegment(length=2),
            CodingSegment(length=1, promoter_direction=PromoterDirection.REVERSE),
            NonCodingSegment(length=3),
        ]
    )
    i = Inversion(4, 8)
    i.apply(g)
    segs = g.to_list()
    print(segs)
    # Should merge the two noncoding segments
    assert any(s.length == 4 and s.is_noncoding() for s in segs)


def test_inversion_invert_coding_promoter():
    g = Genome(
        [
            NonCodingSegment(length=3),
            CodingSegment(length=4, promoter_direction=PromoterDirection.FORWARD),
            NonCodingSegment(length=2),
        ]
    )
    i = Inversion(3, 6)
    i.apply(g)
    segs = g.to_list()
    # Coding segment should have promoter switched
    coding = [s for s in segs if isinstance(s, CodingSegment)][0]
    assert coding.promoter_direction == PromoterDirection.REVERSE


def test_inversion_is_neutral():
    g = make_genome()
    # Invert in noncoding
    i = Inversion(0, 1)
    assert i.is_neutral(g)
    # Invert in coding (not at offset 0)
    i = Inversion(4, 5)
    assert not i.is_neutral(g)
    # Out of bounds
    i = Inversion(0, 100)
    with pytest.raises(IndexError):
        i.is_neutral(g)


def test_inversion_apply_and_describe():
    g = make_genome()
    i = Inversion(0, 1)
    i.apply(g)
    assert len(g) == 9
    assert "Inversion(start_pos=0, end_pos=1)" in i.describe()
