from typing import Optional

from emergents.genome.genome import Genome
from emergents.mutations.base import Mutation


class PointMutation(Mutation):
    def __init__(self, position: int, rng_state: Optional[int] = None):
        super().__init__(rng_state)
        self.position = position

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the mutation is neutral (i.e., does not affect the organism's fitness)."""
        segment, *_ = genome[self.position]
        return segment.is_noncoding()

    def apply(self, genome: Genome):
        """Apply the point mutation to the genome."""
        pass

    # def serialize(self) -> dict:
    #     """Serialize the mutation for logging."""
    #     return {
    #         "type": "point_mutation",
    #         "position": self.position,
    #     }

    def describe(self) -> str:
        """Return a human-readable description of the mutation."""
        return f"Point mutation at {self.position}"
