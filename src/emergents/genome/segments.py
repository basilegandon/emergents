from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto


class PromoterDirection(Enum):
    FORWARD = auto()
    REVERSE = auto()

    def switch(self) -> PromoterDirection:
        return (
            PromoterDirection.FORWARD
            if self == PromoterDirection.REVERSE
            else PromoterDirection.REVERSE
        )


@dataclass
class Segment:
    length: int
    sid: uuid.UUID = field(default_factory=lambda: uuid.uuid4())

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Segment length must be positive.")

    def clone_with_length(self, new_length: int) -> Segment:
        """Return a new Segment of the same type but with new length and a new id."""
        raise NotImplementedError

    def is_noncoding(self) -> bool:
        return isinstance(self, NonCodingSegment)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={self.length}, id={str(self.sid)[:8]})"


@dataclass
class NonCodingSegment(Segment):
    def clone_with_length(self, new_length: int) -> NonCodingSegment:
        if new_length <= 0:
            raise ValueError("NonCodingSegment length must be positive.")
        return NonCodingSegment(length=new_length)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class CodingSegment(Segment):
    promoter_direction: PromoterDirection = PromoterDirection.FORWARD

    def clone_with_length(self, new_length: int) -> CodingSegment:
        if new_length <= 0:
            raise ValueError("CodingSegment length must be positive.")
        return CodingSegment(
            length=new_length, promoter_direction=self.promoter_direction
        )

    def __repr__(self) -> str:
        return super().__repr__().strip(")") + f", dir={self.promoter_direction.name})"
