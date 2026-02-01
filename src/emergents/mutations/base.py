from emergents.genome.genome import Genome


class Mutation:
    def __init__(self, rng_state: int | None = None):
        self.rng_state = rng_state

    def is_neutral(self, genome: Genome) -> bool:
        """Check if the mutation is neutral (i.e., does not affect the organism's fitness)."""
        raise NotImplementedError("Subclasses must implement this method.")

    def apply(self, genome: Genome) -> None:
        """Apply to genome. Returns metadata needed to invert or validate."""
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize(self) -> dict:
        """For logging / replay."""
        raise NotImplementedError("Subclasses must implement this method.")

    def describe(self) -> str:
        """Human-readable summary."""
        raise NotImplementedError("Subclasses must implement this method.")
