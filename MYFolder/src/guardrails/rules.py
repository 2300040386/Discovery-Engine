from __future__ import annotations

import math

from src.guardrails.types import Candidate

ADULT_AGE_BANDS = frozenset({"18_25", "26_35", "36_50", "over_50"})


def filter_availability(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.in_stock]


def filter_policy(candidates: list[Candidate], age_band: str) -> list[Candidate]:
    if age_band in ADULT_AGE_BANDS:
        return list(candidates)
    return [c for c in candidates if not c.age_restricted]


def filter_sensitive(candidates: list[Candidate], blocked_categories: set[str]) -> list[Candidate]:
    return [c for c in candidates if c.category_l1 not in blocked_categories]


def filter_seen(candidates: list[Candidate], seen_ids: set[str]) -> list[Candidate]:
    return [c for c in candidates if c.product_id not in seen_ids]


def _max_pickable(available_by_category: dict[str, int], per_category_limit: int) -> int:
    return sum(min(available, per_category_limit) for available in available_by_category.values())


def cap_category(candidates: list[Candidate], limit: int, max_share: float = 0.35) -> list[Candidate]:
    if limit <= 0 or not candidates:
        return []

    available_by_category: dict[str, int] = {}
    for c in candidates:
        available_by_category[c.category_l1] = available_by_category.get(c.category_l1, 0) + 1

    # A single-item list is always 100% one category, so target=1 is exempted
    # from the feasibility search below (it is the "single-item edge case").
    target = 0
    for m in range(min(limit, len(candidates)), 0, -1):
        if m == 1:
            target = 1
            break
        per_category_limit = math.floor(max_share * m)
        if _max_pickable(available_by_category, per_category_limit) >= m:
            target = m
            break
    if target == 0:
        return []

    per_category_limit = 1 if target == 1 else math.floor(max_share * target)
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

    admitted: list[Candidate] = []
    overflow: list[Candidate] = []
    counts: dict[str, int] = {}
    for c in ranked:
        if len(admitted) >= target:
            overflow.append(c)
            continue
        if counts.get(c.category_l1, 0) < per_category_limit:
            admitted.append(c)
            counts[c.category_l1] = counts.get(c.category_l1, 0) + 1
        else:
            overflow.append(c)
    return admitted


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_diversify(candidates: list[Candidate], limit: int, lam: float = 0.7) -> list[Candidate]:
    if limit <= 0 or not candidates:
        return []

    pool = list(candidates)
    selected: list[Candidate] = []
    while pool and len(selected) < limit:
        best: Candidate | None = None
        best_mmr = float("-inf")
        for c in pool:
            max_sim = max((_cosine_similarity(c.embedding, s.embedding) for s in selected), default=0.0)
            mmr_score = lam * c.score - (1 - lam) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best = c
        assert best is not None
        selected.append(best)
        pool.remove(best)
    return selected
