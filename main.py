import logging

from emergents.genome.genome import Genome
from emergents.genome.segments import (
    CodingSegment,
    NonCodingSegment,
    PromoterDirection,
    Segment,
)
from emergents.mutations.duplication import Duplication
from emergents.mutations.point_mutation import PointMutation
from emergents.mutations.small_deletion import SmallDeletion
from emergents.mutations.small_insertion import SmallInsertion

# from emergents.simulation import Simulation

logger = logging.getLogger(__name__)


def main():
    segments: list[Segment] = [
        NonCodingSegment(100),
        CodingSegment(200, promoter_direction=PromoterDirection.FORWARD),
        NonCodingSegment(50),
        CodingSegment(100, promoter_direction=PromoterDirection.FORWARD),
        CodingSegment(150, promoter_direction=PromoterDirection.REVERSE),
        CodingSegment(100, promoter_direction=PromoterDirection.FORWARD),
        NonCodingSegment(100),
    ]
    genome = Genome(segments)
    logger.info(genome)

    invalid_point_mutation = PointMutation(position=100)
    valid_point_mutation = PointMutation(position=99)

    logger.info(
        f"{invalid_point_mutation.describe()} is neutral? {invalid_point_mutation.is_neutral(genome)}"
    )
    logger.info(
        f"{valid_point_mutation.describe()} is neutral? {valid_point_mutation.is_neutral(genome)}"
    )

    invalid_small_insertion = SmallInsertion(position=101, length=10)
    valid_small_insertion = SmallInsertion(position=100, length=10)

    logger.info(
        f"{invalid_small_insertion.describe()} is neutral? {invalid_small_insertion.is_neutral(genome)}"
    )
    logger.info(
        f"{valid_small_insertion.describe()} is neutral? {valid_small_insertion.is_neutral(genome)}"
    )

    valid_small_insertion.apply(genome)
    logger.info(genome)
    logger.info("Coalescing")
    genome.coalesce_all()
    logger.info(genome)

    # Test small deletion
    invalid_small_deletion = SmallDeletion(position=351, length=10)
    valid_small_deletion = SmallDeletion(position=350, length=10)

    logger.info(
        f"{invalid_small_deletion.describe()} is neutral? {invalid_small_deletion.is_neutral(genome)}"
    )
    logger.info(
        f"{valid_small_deletion.describe()} is neutral? {valid_small_deletion.is_neutral(genome)}"
    )

    valid_small_deletion.apply(genome)
    logger.info(genome)
    genome.coalesce_all()

    # Test duplication
    invalid_duplication = Duplication(351, 598, 309)
    valid_duplication = Duplication(351, 598, 310)

    logger.info(
        f"{invalid_duplication.describe(genome)} is neutral? {invalid_duplication.is_neutral(genome)}"
    )
    logger.info(
        f"{valid_duplication.describe(genome)} is neutral? {valid_duplication.is_neutral(genome)}"
    )

    valid_duplication.apply(genome)
    logger.info(genome)
    logger.info("Coalescing")
    genome.coalesce_all()
    logger.info(genome)


if __name__ == "__main__":
    main()
