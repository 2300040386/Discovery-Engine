from __future__ import annotations

from dataclasses import dataclass

from src.guardrails.rules import (
    cap_category,
    filter_availability,
    filter_policy,
    filter_seen,
    filter_sensitive,
    mmr_diversify,
)
from src.guardrails.types import Candidate


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    dropped: dict[str, int]


def apply_all(
    candidates: list[Candidate],
    *,
    age_band: str,
    blocked_categories: set[str],
    seen_ids: set[str],
    limit: int,
    max_share: float = 0.35,
    lam: float = 0.7,
) -> tuple[list[Candidate], GuardrailReport]:
    dropped: dict[str, int] = {}
    current = candidates

    steps = [
        ("filter_availability", lambda c: filter_availability(c)),
        ("filter_policy", lambda c: filter_policy(c, age_band)),
        ("filter_sensitive", lambda c: filter_sensitive(c, blocked_categories)),
        ("filter_seen", lambda c: filter_seen(c, seen_ids)),
        ("cap_category", lambda c: cap_category(c, limit, max_share=max_share)),
        ("mmr_diversify", lambda c: mmr_diversify(c, limit, lam=lam)),
    ]
    for name, step in steps:
        before = len(current)
        current = step(current)
        dropped[name] = before - len(current)

    return current, GuardrailReport(dropped=dropped)
