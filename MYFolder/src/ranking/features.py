from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.session.intents import SessionEvent

FEATURE_NAMES = (
    "two_tower_dot",
    "intent_dot",
    "item_popularity",
    "item_ctr",
    "price_percentile_in_category",
    "category_match_count",
    "recency_same_category_minutes",
    "session_length",
    "is_new_item",
    "days_since_first_seen",
)

NEW_ITEM_DAYS_THRESHOLD = 14.0
NO_RECENT_EVENT_MINUTES = -1.0


@dataclass(frozen=True, slots=True)
class UserFeatures:
    user_id: str
    user_vector: list[float]
    intent_vectors: list[list[float]]
    intent_weights: list[float]


@dataclass(frozen=True, slots=True)
class RankingCandidate:
    article_id: int
    category_l1: str
    price: float
    item_vector: list[float]
    popularity: float
    ctr: float
    days_since_first_seen: float


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _intent_dot(user: UserFeatures, item_vector: list[float]) -> float:
    return sum(weight * _dot(vector, item_vector) for weight, vector in zip(user.intent_weights, user.intent_vectors))


def _price_percentiles(candidates: list[RankingCandidate]) -> dict[int, float]:
    by_category: dict[str, list[RankingCandidate]] = defaultdict(list)
    for c in candidates:
        by_category[c.category_l1].append(c)

    percentile_by_id: dict[int, float] = {}
    for group in by_category.values():
        prices = sorted(c.price for c in group)
        n = len(group)
        for c in group:
            if n == 1:
                percentile_by_id[c.article_id] = 0.5
            else:
                below = sum(1 for p in prices if p < c.price)
                percentile_by_id[c.article_id] = below / (n - 1)
    return percentile_by_id


def _category_match_counts(session: list[SessionEvent]) -> Counter[str]:
    return Counter(e.category_l1 for e in session)


def _last_category_event_time(session: list[SessionEvent]) -> dict[str, datetime]:
    last_seen: dict[str, datetime] = {}
    for e in session:
        current = last_seen.get(e.category_l1)
        if current is None or e.timestamp > current:
            last_seen[e.category_l1] = e.timestamp
    return last_seen


def build_features(
    user: UserFeatures,
    candidates: list[RankingCandidate],
    session: list[SessionEvent],
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    percentile_by_id = _price_percentiles(candidates)
    category_counts = _category_match_counts(session)
    last_category_time = _last_category_event_time(session)
    now = max((e.timestamp for e in session), default=None)
    session_length = float(len(session))

    rows = np.empty((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(candidates):
        last_time = last_category_time.get(c.category_l1)
        recency = (
            (now - last_time).total_seconds() / 60.0 if now is not None and last_time is not None else NO_RECENT_EVENT_MINUTES
        )

        rows[i] = (
            _dot(user.user_vector, c.item_vector),
            _intent_dot(user, c.item_vector),
            c.popularity,
            c.ctr,
            percentile_by_id[c.article_id],
            float(category_counts.get(c.category_l1, 0)),
            recency,
            session_length,
            1.0 if c.days_since_first_seen < NEW_ITEM_DAYS_THRESHOLD else 0.0,
            c.days_since_first_seen,
        )
    return rows
