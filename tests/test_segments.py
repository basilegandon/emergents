import uuid

import pytest

from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)


def test_promoter_direction_switch():
    assert PromoterDirection.FORWARD.switch() == PromoterDirection.REVERSE
    assert PromoterDirection.REVERSE.switch() == PromoterDirection.FORWARD


def test_segment_positive_length():
    s = Segment.__new__(Segment)
    s.length = 1
    s.sid = uuid.uuid4()
    # __post_init__ should not raise
    Segment.__post_init__(s)


def test_segment_zero_or_negative_length():
    with pytest.raises(ValueError, match="Segment length must be positive."):
        Segment(length=0)
    with pytest.raises(ValueError, match="Segment length must be positive."):
        Segment(length=-5)


def test_segment_clone_with_length_not_implemented():
    s = Segment(length=5)
    with pytest.raises(NotImplementedError):
        s.clone_with_length(10)


def test_segment_is_noncoding():
    nc = NonCodingSegment(length=5)
    c = CodingSegment(length=5)
    assert nc.is_noncoding() is True
    assert c.is_noncoding() is False


def test_segment_repr():
    nc = NonCodingSegment(length=5)
    c = CodingSegment(length=5)
    assert nc.__class__.__name__ in repr(nc)
    assert c.__class__.__name__ in repr(c)
    assert f"dir={c.promoter_direction.name}" in repr(c)


def test_noncodingsegment_clone_with_length():
    nc = NonCodingSegment(length=5)
    nc2 = nc.clone_with_length(10)
    assert isinstance(nc2, NonCodingSegment)
    assert nc2.length == 10
    assert nc2.sid != nc.sid
    with pytest.raises(ValueError, match="NonCodingSegment length must be positive."):
        nc.clone_with_length(0)
    with pytest.raises(ValueError, match="NonCodingSegment length must be positive."):
        nc.clone_with_length(-1)


def test_codingsegment_clone_with_length_and_direction():
    c = CodingSegment(length=5, promoter_direction=PromoterDirection.REVERSE)
    c2 = c.clone_with_length(8)
    assert isinstance(c2, CodingSegment)
    assert c2.length == 8
    assert c2.promoter_direction == c.promoter_direction
    assert c2.sid != c.sid
    with pytest.raises(ValueError, match="CodingSegment length must be positive."):
        c.clone_with_length(0)
    with pytest.raises(ValueError, match="CodingSegment length must be positive."):
        c.clone_with_length(-2)


def test_codingsegment_repr_direction():
    c = CodingSegment(length=5, promoter_direction=PromoterDirection.REVERSE)
    rep = repr(c)
    assert "REVERSE" in rep
    c2 = CodingSegment(length=5, promoter_direction=PromoterDirection.FORWARD)
    assert "FORWARD" in repr(c2)


def test_segment_uuid_uniqueness():
    s1 = NonCodingSegment(length=1)
    s2 = NonCodingSegment(length=1)
    assert s1.sid != s2.sid
    s3 = CodingSegment(length=1)
    s4 = CodingSegment(length=1)
    assert s3.sid != s4.sid
