import logging

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.inversion import Inversion

# from emergents.simulation import Simulation

logger = logging.getLogger(__name__)


def main() -> None:
    segments: list[Segment] = [
        NonCodingSegment(10),
        CodingSegment(200, promoter_direction=PromoterDirection.FORWARD),
        NonCodingSegment(50),
        CodingSegment(100, promoter_direction=PromoterDirection.FORWARD),
        CodingSegment(150, promoter_direction=PromoterDirection.REVERSE),
        CodingSegment(100, promoter_direction=PromoterDirection.FORWARD),
        NonCodingSegment(100),
    ]
    genome = Genome(segments)
    logger.info(genome)

    # invalid_point_mutation = PointMutation(position=100)
    # valid_point_mutation = PointMutation(position=799)

    # logger.info(
    #     f"{invalid_point_mutation.describe()} is neutral? {invalid_point_mutation.is_neutral(genome)}"
    # )
    # logger.info(
    #     f"{valid_point_mutation.describe()} is neutral? {valid_point_mutation.is_neutral(genome)}"
    # )

    # invalid_small_insertion = SmallInsertion(position=101, length=10)
    # valid_small_insertion = SmallInsertion(position=800, length=10)

    # logger.info(
    #     f"{invalid_small_insertion.describe()} is neutral? {invalid_small_insertion.is_neutral(genome)}"
    # )
    # logger.info(
    #     f"{valid_small_insertion.describe()} is neutral? {valid_small_insertion.is_neutral(genome)}"
    # )

    # valid_small_insertion.apply(genome)
    # logger.info(genome)

    # # Test small deletion

    # invalid_small_deletion = SmallDeletion(position=1, length=10)
    # valid_small_deletion = SmallDeletion(position=0, length=10)

    # logger.info(
    #     f"{invalid_small_deletion.describe()} is neutral? {invalid_small_deletion.is_neutral(genome)}"
    # )
    # logger.info(
    #     f"{valid_small_deletion.describe()} is neutral? {valid_small_deletion.is_neutral(genome)}"
    # )

    # valid_small_deletion.apply(genome)
    # logger.info(genome)

    # # Test duplication
    # invalid_duplication = Duplication(0, 10, 309)
    # valid_duplication = Duplication(261, 508, 220)

    # logger.info(
    #     f"{invalid_duplication.describe(genome)} is neutral? {invalid_duplication.is_neutral(genome)}"
    # )
    # logger.info(
    #     f"{valid_duplication.describe(genome)} is neutral? {valid_duplication.is_neutral(genome)}"
    # )

    # valid_duplication.apply(genome)
    # logger.info(genome)

    # Test Inversion
    invalid_inversion = Inversion(5, 209)
    valid_inversion = Inversion(220, 620)

    logger.info(
        f"{invalid_inversion.describe()} is neutral? {invalid_inversion.is_neutral(genome)}"
    )
    logger.info(
        f"{valid_inversion.describe()} is neutral? {valid_inversion.is_neutral(genome)}"
    )

    valid_inversion.apply(genome)
    logger.info(genome)


if __name__ == "__main__":
    main()
