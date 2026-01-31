import pytest

from emergents.genome.coordinates import CoordinateSystem, DefaultCoordinateValidator


def validator():
    return DefaultCoordinateValidator()


@pytest.mark.parametrize(
    ("pos", "genome_length"),
    [
        (0, 1),
        (5, 6),
        (9, 10),
    ],
)
def test_validate_position_base_valid(pos, genome_length):
    v = validator()
    v.validate_position(pos, genome_length, CoordinateSystem.BASE)


@pytest.mark.parametrize(
    ("pos", "genome_length"),
    [
        (-1, 5),
        (5, 5),
        (100, 10),
    ],
)
def test_validate_position_base_out_of_bounds(pos, genome_length):
    v = validator()
    with pytest.raises(IndexError):
        v.validate_position(pos, genome_length, CoordinateSystem.BASE)


@pytest.mark.parametrize(
    ("pos", "genome_length"),
    [
        (0, 0),
        (0, 1),
        (1, 1),
        (10, 10),
    ],
)
def test_validate_position_gap_valid(pos, genome_length):
    v = validator()
    v.validate_position(pos, genome_length, CoordinateSystem.GAP)


@pytest.mark.parametrize(
    ("pos", "genome_length"),
    [
        (-1, 0),
        (-1, 5),
        (6, 5),
        (11, 10),
    ],
)
def test_validate_position_gap_out_of_bounds(pos, genome_length):
    v = validator()
    with pytest.raises(IndexError):
        v.validate_position(pos, genome_length, CoordinateSystem.GAP)


def test_validate_position_unknown_coord_sys():
    v = validator()

    class Dummy:
        pass

    with pytest.raises(ValueError, match="Unknown coordinate system"):
        v.validate_position(0, 1, Dummy())


@pytest.mark.parametrize(
    ("start", "end", "genome_length"),
    [
        (0, 1, 1),
        (0, 10, 10),
        (5, 10, 10),
    ],
)
def test_validate_base_range_valid(start, end, genome_length):
    v = validator()
    v.validate_base_range(start, end, genome_length)


@pytest.mark.parametrize(
    ("start", "end", "genome_length"),
    [
        (-1, 5, 10),
        (0, 11, 10),
        (5, 15, 10),
    ],
)
def test_validate_base_range_out_of_bounds(start, end, genome_length):
    v = validator()
    with pytest.raises(IndexError):
        v.validate_base_range(start, end, genome_length)


@pytest.mark.parametrize(
    ("start", "end", "genome_length"),
    [
        (5, 5, 10),
        (6, 5, 10),
        (10, 0, 10),
    ],
)
def test_validate_base_range_invalid_range(start, end, genome_length):
    v = validator()
    with pytest.raises(ValueError, match="Invalid range"):
        v.validate_base_range(start, end, genome_length)
