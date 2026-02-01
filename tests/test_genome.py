import pytest

from emergents.genome.coordinates import CoordinateSystem
from emergents.genome.genome import Genome
from emergents.genome.segments import CodingSegment, NonCodingSegment, PromoterDirection


# Helper to create a list of segments
def make_segments(lengths, cls=NonCodingSegment):
    return [cls(length=length) for length in lengths]


def test_genome_init_empty_and_bool():
    g = Genome()
    assert len(g) == 0
    assert not g
    assert g.root is None


def test_genome_init_with_segments():
    segs = make_segments([3, 2, 1])
    g = Genome(segs)
    assert len(g) == 6
    assert g.root is not None
    assert g.to_list() == segs


def test_find_segment_at_position_base():
    segs = make_segments([3, 2, 1])
    g = Genome(segs)
    # First base
    seg, offset, start, end = g.find_segment_at_position(0, CoordinateSystem.BASE)
    assert seg is segs[0]
    assert offset == 0
    # Last base
    seg, offset, start, end = g.find_segment_at_position(5, CoordinateSystem.BASE)
    assert seg is segs[2]
    assert offset == 0
    # Out of bounds
    with pytest.raises(IndexError):
        g.find_segment_at_position(6, CoordinateSystem.BASE)


def test_find_segment_at_position_gap():
    segs = make_segments([3, 2, 1])
    g = Genome(segs)
    # At the end (gap)
    seg, offset, start, end = g.find_segment_at_position(6, CoordinateSystem.GAP)
    assert seg is segs[2]
    assert start == 5
    assert end == 6
    # Out of bounds
    with pytest.raises(IndexError):
        g.find_segment_at_position(7, CoordinateSystem.GAP)
    g = Genome()
    with pytest.raises(IndexError):
        g.find_segment_at_position(0, CoordinateSystem.GAP)


def test_insert_at_gap_and_coalesce():
    g = Genome(make_segments([3, 2]))
    # Insert coding segment at gap 2
    coding = CodingSegment(length=2, promoter_direction=PromoterDirection.FORWARD)
    with pytest.raises(TypeError):
        g.insert_at_gap(2, coding)  # should raise
    # Insert noncoding at gap 0, should merge with first
    g.insert_at_gap(0, NonCodingSegment(length=1))
    segs = g.to_list()
    assert isinstance(segs[0], NonCodingSegment)
    # Insert noncoding at end, should merge with last
    g.insert_at_gap(len(g), NonCodingSegment(length=2))
    segs = g.to_list()
    assert isinstance(segs[-1], NonCodingSegment)
    # Coalesce all noncoding
    g.coalesce_all()
    segs = g.to_list()
    noncoding_count = sum(isinstance(s, NonCodingSegment) for s in segs)
    assert noncoding_count == 2 or noncoding_count == 1


def test_insert_at_gap_invalid():
    g = Genome(make_segments([3, 2]))
    with pytest.raises(IndexError):
        g.insert_at_gap(6, NonCodingSegment(length=1))


def test_delete_range_base_and_gap():
    g = Genome(make_segments([3, 2, 1]))
    print(g)
    # Delete nothing
    g.delete_range(2, 2)
    print(g)
    assert len(g) == 6
    # Delete first base
    g.delete_range(0, 1)
    print(g)
    assert len(g) == 5
    # Delete last base
    g.delete_range(4, 5)
    print(g)
    assert len(g) == 4
    # Delete all
    g.delete_range(0, 4)
    print(g)
    assert len(g) == 0


def test_delete_range_invalid():
    g = Genome(make_segments([3, 2, 1]))
    with pytest.raises(ValueError, match="start must be < end"):
        g.delete_range(4, 2)
    # Circular: should not raise, but delete from start after deleting to end
    g = Genome(make_segments([3, 2, 1]), circular=True)
    g.delete_range(5, 2)
    assert len(g) < 6


def test_extend_segment_at():
    g = Genome(make_segments([3, 2, 1]))
    # Extend noncoding at gap 0
    g.extend_segment_at(0, 2)
    segs = g.to_list()
    assert segs[0].length == 5
    # Extend noncoding at gap in middle
    g.extend_segment_at(5, 1)
    segs = g.to_list()
    assert segs[1].length == 3
    # Negative delta
    with pytest.raises(ValueError, match="delta must be positive"):
        g.extend_segment_at(0, -1)
    # On coding segment
    g = Genome(make_segments([3, 2], cls=CodingSegment))
    with pytest.raises(TypeError):
        g.extend_segment_at(2, 1)


def test_coalesce_all_empty():
    g = Genome()
    g.coalesce_all()
    assert g.root is None


def test_iter_segments_and_repr():
    segs = make_segments([3, 2, 1])
    g = Genome(segs)
    segs_out = [s for s, start, end in g.iter_segments()]
    assert segs_out == segs
    rep = repr(g)
    assert "Genome of length" in rep
    for s in segs:
        assert s.__class__.__name__ in rep
