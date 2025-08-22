from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto


class PromoterDirection(Enum):
    FORWARD = auto()
    REVERSE = auto()


@dataclass
class Segment:
    length: int
    sid: uuid.UUID = field(default_factory=lambda: uuid.uuid4())

    def clone_with_length(self, new_length: int) -> "Segment":
        """Return a new Segment of the same type but with new length and a new id."""
        raise NotImplementedError

    def is_noncoding(self) -> bool:
        return isinstance(self, NonCodingSegment)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={self.length}, id={str(self.sid)[:8]})"


@dataclass
class NonCodingSegment(Segment):
    def clone_with_length(self, new_length: int) -> "NonCodingSegment":
        return NonCodingSegment(length=new_length)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class CodingSegment(Segment):
    promoter_direction: PromoterDirection = PromoterDirection.FORWARD

    def clone_with_length(self, new_length: int) -> "CodingSegment":
        return CodingSegment(
            length=new_length, promoter_direction=self.promoter_direction
        )

    def __repr__(self) -> str:
        return super().__repr__().strip(")") + f", dir={self.promoter_direction.name})"
