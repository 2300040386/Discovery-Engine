from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    product_id: str
    score: float
    category_l1: str
    in_stock: bool
    age_restricted: bool
    embedding: list[float]
