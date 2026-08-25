from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.session.encoder import EMBED_DIM, SESSION_LENGTH, SessionEncoder

ARTIFACTS_DIR = Path("artifacts")
SESSION_ENCODER_PATH = ARTIFACTS_DIR / "session_encoder.pt"
ITEM_VECTORS_PATH = ARTIFACTS_DIR / "item_vectors.npy"
ID_MAP_PATH = ARTIFACTS_DIR / "id_map.json"

LABELS = ("urgent_replacement", "seasonal_browsing", "complementary", "bargain_hunting")
COLD_START_LABEL = "cold_start"


@dataclass(frozen=True, slots=True)
class SessionEvent:
    article_id: int
    timestamp: datetime
    price: float
    category_l1: str
    discounted: bool = False


@dataclass(frozen=True, slots=True)
class Intent:
    vector: list[float]
    weight: float
    label: str


_encoder: SessionEncoder | None = None
_item_vectors: np.ndarray | None = None
_article_id_to_index: dict[int, int] | None = None


def _load_defaults() -> tuple[SessionEncoder, np.ndarray, dict[int, int]]:
    global _encoder, _item_vectors, _article_id_to_index
    if _encoder is None:
        checkpoint = torch.load(SESSION_ENCODER_PATH, map_location="cpu", weights_only=False)
        _encoder = SessionEncoder(**checkpoint["config"])
        _encoder.load_state_dict(checkpoint["encoder_state"])
        _encoder.eval()
    if _item_vectors is None:
        _item_vectors = np.load(ITEM_VECTORS_PATH)
    if _article_id_to_index is None:
        with ID_MAP_PATH.open() as f:
            id_map = json.load(f)
        _article_id_to_index = {int(k): v for k, v in id_map["article_id_to_index"].items()}
    return _encoder, _item_vectors, _article_id_to_index


def _cold_start() -> list[Intent]:
    return [Intent(vector=[0.0] * EMBED_DIM, weight=1.0, label=COLD_START_LABEL)]


def _label_scores(events: list[SessionEvent]) -> dict[str, float]:
    n = len(events)
    ordered = sorted(events, key=lambda e: e.timestamp)
    gaps_min = [(b.timestamp - a.timestamp).total_seconds() / 60.0 for a, b in zip(ordered, ordered[1:])]
    mean_gap_min = sum(gaps_min) / len(gaps_min) if gaps_min else 0.0

    categories = {e.category_l1 for e in events}
    category_spread = len(categories) / n

    prices = [e.price for e in events]
    price_mean = sum(prices) / n
    price_var = sum((p - price_mean) ** 2 for p in prices) / n

    discount_ratio = sum(e.discounted for e in events) / n

    return {
        "urgent_replacement": (1.0 / (1.0 + mean_gap_min)) * (1.0 - category_spread),
        "seasonal_browsing": category_spread * min(1.0, mean_gap_min / 10.0 + 0.3),
        "complementary": category_spread * (1.0 - min(1.0, mean_gap_min / 10.0)),
        "bargain_hunting": 0.6 * discount_ratio + 0.4 * min(1.0, price_var**0.5 / (price_mean + 1e-6)),
    }


def _pad_embeddings(embeddings: np.ndarray, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, dim = embeddings.shape
    padded = np.zeros((length, dim), dtype=np.float32)
    used = min(n, length)
    padded[:used] = embeddings[:used]
    mask = np.zeros(length, dtype=np.float32)
    mask[:used] = 1.0
    return torch.tensor(padded).unsqueeze(0), torch.tensor(mask).unsqueeze(0)


def infer_intents(
    events: list[SessionEvent],
    encoder: SessionEncoder | None = None,
    item_vectors: np.ndarray | None = None,
    article_id_to_index: dict[int, int] | None = None,
) -> list[Intent]:
    if not events:
        return _cold_start()

    if encoder is None or item_vectors is None or article_id_to_index is None:
        encoder, item_vectors, article_id_to_index = _load_defaults()

    recent = sorted(events, key=lambda e: e.timestamp)[-SESSION_LENGTH:]
    idxs = [article_id_to_index[e.article_id] for e in recent if e.article_id in article_id_to_index]
    if not idxs:
        return _cold_start()

    padded_embeddings, mask = _pad_embeddings(item_vectors[idxs], SESSION_LENGTH)

    encoder.eval()
    with torch.no_grad():
        intents, weights = encoder(padded_embeddings, mask)

    scores = _label_scores(events)
    ranked_labels = sorted(LABELS, key=lambda label: scores[label], reverse=True)
    order = torch.argsort(weights[0], descending=True).tolist()

    return [
        Intent(vector=intents[0, head_idx].tolist(), weight=float(weights[0, head_idx]), label=ranked_labels[rank])
        for rank, head_idx in enumerate(order)
    ]
