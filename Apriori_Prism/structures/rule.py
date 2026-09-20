from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ItemSet:
    items: frozenset[tuple[str, bool]]
    support: int = 0


@dataclass(frozen=True, slots=True)
class Rule:
    antecedent: frozenset[tuple[str, bool]]
    consequent: frozenset[tuple[str, bool]]
    support: int
    confidence: float
    lift: float
