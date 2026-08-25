from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.compliance.erasure import erase_user
from src.guardrails.pipeline import GuardrailReport, apply_all
from src.guardrails.types import Candidate
from src.ranking.features import RankingCandidate, UserFeatures, build_features
from src.session.encoder import EMBED_DIM, SessionEncoder
from src.session.intents import COLD_START_LABEL, Intent, SessionEvent, infer_intents
from src.shared import telemetry
from src.shared.config import Settings, get_settings
from src.shared.db import get_pg_connection, get_redis_client
from src.shared.telemetry import Timer, get_logger
from trainer.train_towers import UserTower

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
ITEMS_PATH = PROCESSED_DIR / "items.parquet"

ARTIFACTS_DIR = Path("artifacts")
ITEM_VECTORS_PATH = ARTIFACTS_DIR / "item_vectors.npy"
ID_MAP_PATH = ARTIFACTS_DIR / "id_map.json"
TOWERS_PATH = ARTIFACTS_DIR / "towers.pt"
SESSION_ENCODER_PATH = ARTIFACTS_DIR / "session_encoder.pt"

OBJECTIVES = ("click", "cart", "purchase", "wishlist")
COMBINED_WEIGHTS = {"click": 0.2, "cart": 0.3, "purchase": 0.5}
CTR_SMOOTHING = 20.0

SESSION_LENGTH = 20
SESSION_TTL_SECONDS = 1800
QUERY_CACHE_TTL_SECONDS = 3600
RETRIEVAL_TIMEOUT_S = 0.04


# --- request/response models ---


class FeedContext(BaseModel):
    device: str | None = None
    region: str | None = None
    age_band: str | None = None


class FeedRequest(BaseModel):
    user_id: str | None = None
    session_id: str
    limit: int = 20
    context: FeedContext | None = None


class BundlesRequest(BaseModel):
    seed_product_id: int
    user_id: str | None = None
    limit: int = 10


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    session_id: str | None = None
    limit: int = 20


class EventRequest(BaseModel):
    user_id: str | None = None
    session_id: str
    article_id: int
    event_type: str = "view"
    price: float | None = None
    category_l1: str | None = None
    discounted: bool = False
    timestamp: datetime | None = None


class ConsentRequest(BaseModel):
    user_id: str
    purpose: str
    granted: bool


class IntentOut(BaseModel):
    label: str
    confidence: float
    slots: int


class ReasonOut(BaseModel):
    code: str
    text: str


class ItemOut(BaseModel):
    product_id: str
    score: float
    intent: str | None = None
    reason: ReasonOut


class GuardrailsOut(BaseModel):
    diversity_cap_applied: bool
    items_dropped: int
    filters_fired: dict[str, int]


class FeedResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str
    model_version: str
    detected_intents: list[IntentOut]
    items: list[ItemOut]
    guardrails: GuardrailsOut
    degraded: str | None = None
    latency_ms: int


class BundlesResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str
    model_version: str
    seed_product_id: int
    items: list[ItemOut]
    guardrails: GuardrailsOut
    degraded: str | None = None
    latency_ms: int


class SearchResponseBody(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str
    model_version: str
    query: str
    parsed_intent: dict
    items: list[ItemOut]
    guardrails: GuardrailsOut
    degraded: str | None = None
    latency_ms: int


# --- item metadata (materialized once at startup, mirrors trainer/train_ranker.py) ---


@dataclass(frozen=True, slots=True)
class ItemMeta:
    category_l1: str
    popularity: float
    ctr: float
    avg_price: float
    days_since_first_seen: float


DEFAULT_ITEM_META = ItemMeta(category_l1="unknown", popularity=0.0, ctr=0.0, avg_price=0.0, days_since_first_seen=9999.0)


def _build_item_meta(items: pl.DataFrame, train: pl.DataFrame) -> dict[int, ItemMeta]:
    max_n = items["n_interactions"].max() or 1
    category_by_item = dict(zip(items["article_id"].to_list(), items["category_l1"].to_list()))
    n_by_item = dict(zip(items["article_id"].to_list(), items["n_interactions"].to_list()))

    stats = train.group_by("article_id").agg(
        pl.col("t_dat").min().alias("first_seen"), pl.col("price").mean().alias("avg_price")
    )
    first_seen_by_item = dict(zip(stats["article_id"].to_list(), stats["first_seen"].to_list()))
    avg_price_by_item = dict(zip(stats["article_id"].to_list(), stats["avg_price"].to_list()))
    reference_date = train["t_dat"].max()

    meta: dict[int, ItemMeta] = {}
    for article_id in items["article_id"].to_list():
        n = n_by_item.get(article_id, 0)
        first_seen = first_seen_by_item.get(article_id, reference_date)
        meta[article_id] = ItemMeta(
            category_l1=category_by_item.get(article_id, "unknown"),
            popularity=n / max_n,
            ctr=n / (n + CTR_SMOOTHING),
            avg_price=avg_price_by_item.get(article_id, 0.0),
            days_since_first_seen=max(0.0, float((reference_date - first_seen).days)),
        )
    return meta


def _build_category_centroids(
    items: pl.DataFrame, item_vectors: np.ndarray, article_id_to_index: dict[int, int]
) -> dict[str, list[float]]:
    vectors_by_category: dict[str, list[np.ndarray]] = defaultdict(list)
    for article_id, category in zip(items["article_id"].to_list(), items["category_l1"].to_list()):
        idx = article_id_to_index.get(article_id)
        if idx is not None:
            vectors_by_category[category].append(item_vectors[idx])
    return {category: np.mean(vecs, axis=0).tolist() for category, vecs in vectors_by_category.items()}


# --- app state ---


class AppState:
    settings: Settings | None = None
    item_vectors: np.ndarray | None = None
    article_id_to_index: dict[int, int] | None = None
    index_to_article_id: dict[int, int] | None = None
    user_tower: UserTower | None = None
    age_band_vocab: dict[str, int] | None = None
    region_vocab: dict[str, int] | None = None
    session_encoder: SessionEncoder | None = None
    boosters: dict[str, lgb.Booster] | None = None
    item_meta: dict[int, ItemMeta] | None = None
    category_centroid: dict[str, list[float]] | None = None
    global_centroid: np.ndarray | None = None
    category_vocab: list[str] | None = None
    http_client: httpx.Client | None = None
    redis_client: object | None = None
    pg_conn_factory: Callable[[], object] | None = None


state = AppState()


def _load_session_encoder(path: Path) -> SessionEncoder | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = SessionEncoder(**checkpoint["config"])
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()
    return encoder


def _populate_state(target: AppState, settings: Settings) -> None:
    target.settings = settings

    target.item_vectors = np.load(ITEM_VECTORS_PATH)
    with ID_MAP_PATH.open() as f:
        id_map = json.load(f)
    target.article_id_to_index = {int(k): v for k, v in id_map["article_id_to_index"].items()}
    target.index_to_article_id = {int(k): v for k, v in id_map["index_to_article_id"].items()}

    checkpoint = torch.load(TOWERS_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    target.user_tower = UserTower(
        len(checkpoint["age_band_vocab"]), len(checkpoint["region_vocab"]), embed_dim=config["embedding_dim"]
    )
    target.user_tower.load_state_dict(checkpoint["user_tower_state"])
    target.user_tower.eval()
    target.age_band_vocab = checkpoint["age_band_vocab"]
    target.region_vocab = checkpoint["region_vocab"]

    target.session_encoder = _load_session_encoder(SESSION_ENCODER_PATH)
    target.boosters = {obj: lgb.Booster(model_file=str(ARTIFACTS_DIR / f"ranker_{obj}.txt")) for obj in OBJECTIVES}

    items = pl.read_parquet(ITEMS_PATH)
    train = pl.read_parquet(TRAIN_PATH)
    target.item_meta = _build_item_meta(items, train)
    target.category_centroid = _build_category_centroids(items, target.item_vectors, target.article_id_to_index)
    target.global_centroid = target.item_vectors.mean(axis=0)
    target.category_vocab = sorted(set(items["category_l1"].to_list()))

    target.http_client = httpx.Client()
    target.redis_client = get_redis_client(settings)
    target.pg_conn_factory = lambda: get_pg_connection(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _populate_state(state, get_settings())
    except Exception:
        logger.exception("failed to load gateway state at startup")
    yield
    if state.http_client is not None:
        state.http_client.close()


app = FastAPI(lifespan=lifespan)


class GuardrailsUnavailable(Exception):
    pass


@app.exception_handler(GuardrailsUnavailable)
def _handle_guardrails_unavailable(request: Request, exc: GuardrailsUnavailable) -> JSONResponse:
    return _error_response("GUARDRAILS_UNAVAILABLE", "guardrails could not be evaluated", str(exc), retryable=True, status_code=503)


class _NullIndexHandle:
    # No per-user vectors are persisted anywhere (user embeddings are computed
    # on the fly from session history), so there is nothing to remove.
    def remove(self, user_id: str) -> None:
        return None


_NULL_INDEX_HANDLE = _NullIndexHandle()


def _error_response(code: str, message: str, request_id: str, retryable: bool, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id, "retryable": retryable}},
    )


# --- session helpers ---


def _session_key(user_id: str | None, session_id: str | None) -> str:
    return f"sess:{user_id}" if user_id else f"sess:{session_id or 'anonymous'}"


def _load_session_events(redis_client: object, key: str) -> list[SessionEvent]:
    try:
        raw = redis_client.lrange(key, 0, SESSION_LENGTH - 1)
    except Exception:
        logger.exception("failed to load session from redis")
        return []

    events: list[SessionEvent] = []
    for item in raw:
        try:
            data = json.loads(item)
            events.append(
                SessionEvent(
                    article_id=int(data["article_id"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    price=float(data.get("price", 0.0)),
                    category_l1=str(data.get("category_l1", "unknown")),
                    discounted=bool(data.get("discounted", False)),
                )
            )
        except Exception:
            continue
    return events


def _infer_intents_safe(
    events: list[SessionEvent],
    encoder: SessionEncoder | None,
    item_vectors: np.ndarray | None,
    article_id_to_index: dict[int, int] | None,
) -> list[Intent]:
    if encoder is None or item_vectors is None or article_id_to_index is None:
        return [Intent(vector=[0.0] * EMBED_DIM, weight=1.0, label=COLD_START_LABEL)]
    return infer_intents(events, encoder=encoder, item_vectors=item_vectors, article_id_to_index=article_id_to_index)


def _context_region_age(context: FeedContext | None) -> tuple[str, str]:
    region = (context.region if context else None) or "unknown"
    age_band = (context.age_band if context else None) or "unknown"
    return region, age_band


def _compute_user_vector(
    user_tower: UserTower,
    events: list[SessionEvent],
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
    age_band_idx: int,
    region_idx: int,
) -> list[float]:
    idxs = [article_id_to_index[e.article_id] for e in events[-SESSION_LENGTH:] if e.article_id in article_id_to_index]
    if idxs:
        history_emb = torch.tensor(item_vectors[idxs], dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, len(idxs))
    else:
        history_emb = torch.zeros(1, 1, item_vectors.shape[1])
        mask = torch.zeros(1, 1)
    with torch.no_grad():
        user_emb = user_tower(history_emb, mask, torch.tensor([age_band_idx]), torch.tensor([region_idx]))
    return user_emb[0].tolist()


# --- retrieval ---


def _call_retrieval(
    client: httpx.Client, base_url: str, vectors: list[list[float]], k: int, exclude: list[int]
) -> list[list[dict]]:
    response = client.post(f"{base_url}/search", json={"vectors": vectors, "k": k, "exclude": exclude}, timeout=RETRIEVAL_TIMEOUT_S)
    response.raise_for_status()
    return response.json()["results"]


def _merge_candidates(
    results_per_query: list[list[dict]], labels: list[str], merge_size: int
) -> tuple[list[int], dict[int, float], dict[int, str]]:
    best_score: dict[int, float] = {}
    source_label: dict[int, str] = {}
    for label, row in zip(labels, results_per_query):
        for entry in row:
            article_id, score = int(entry["id"]), float(entry["score"])
            if article_id not in best_score or score > best_score[article_id]:
                best_score[article_id] = score
            source_label.setdefault(article_id, label)
    ranked = sorted(best_score, key=lambda a: best_score[a], reverse=True)[:merge_size]
    return ranked, best_score, source_label


def _popularity_fallback(redis_client: object, region: str, age_band: str, limit: int) -> list[int]:
    key = f"pop:{region}:{age_band}"
    try:
        raw = redis_client.zrevrange(key, 0, limit - 1)
    except Exception:
        logger.exception("popularity fallback lookup failed")
        return []
    return [int(a) for a in raw]


def _get_feed_candidates(
    state: AppState, intents: list[Intent], seen_ids: list[int], region: str, age_band: str, merge_size: int
) -> tuple[list[int], dict[int, float], dict[int, str], str | None]:
    is_cold_start = len(intents) == 1 and intents[0].label == COLD_START_LABEL
    if is_cold_start:
        pop_ids = _popularity_fallback(state.redis_client, region, age_band, merge_size)
        if pop_ids:
            return pop_ids, {a: 1.0 for a in pop_ids}, {a: COLD_START_LABEL for a in pop_ids}, None

    vectors = [i.vector for i in intents]
    labels = [i.label for i in intents]
    try:
        results = _call_retrieval(state.http_client, state.settings.retrieval_service_url, vectors, state.settings.retrieval_k_per_query, seen_ids)
        ids, scores, source = _merge_candidates(results, labels, merge_size)
        return ids, scores, source, None
    except Exception:
        logger.exception("retrieval call failed, falling back to popularity")
        pop_ids = _popularity_fallback(state.redis_client, region, age_band, merge_size)
        source = {a: "popularity_fallback" for a in pop_ids}
        return pop_ids, {a: 1.0 for a in pop_ids}, source, "retrieval"


# --- ranking ---


def _make_ranking_candidate(
    article_id: int, item_vectors: np.ndarray, article_id_to_index: dict[int, int], item_meta: dict[int, ItemMeta]
) -> RankingCandidate | None:
    idx = article_id_to_index.get(article_id)
    if idx is None:
        return None
    meta = item_meta.get(article_id, DEFAULT_ITEM_META)
    return RankingCandidate(
        article_id=article_id,
        category_l1=meta.category_l1,
        price=meta.avg_price,
        item_vector=item_vectors[idx].tolist(),
        popularity=meta.popularity,
        ctr=meta.ctr,
        days_since_first_seen=meta.days_since_first_seen,
    )


def _ranking_candidates(article_ids: list[int], state: AppState) -> tuple[list[RankingCandidate], list[int]]:
    candidates: list[RankingCandidate] = []
    valid_ids: list[int] = []
    for article_id in article_ids:
        c = _make_ranking_candidate(article_id, state.item_vectors, state.article_id_to_index, state.item_meta)
        if c is not None:
            candidates.append(c)
            valid_ids.append(article_id)
    return candidates, valid_ids


def _combined_score(boosters: dict[str, lgb.Booster], X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros(0)
    return sum(weight * boosters[obj].predict(X) for obj, weight in COMBINED_WEIGHTS.items())


def _rank(
    user_features: UserFeatures,
    article_ids: list[int],
    session_events: list[SessionEvent],
    state: AppState,
    fallback_scores: dict[int, float],
) -> tuple[list[int], dict[int, float], str | None]:
    try:
        candidates, valid_ids = _ranking_candidates(article_ids, state)
        if not candidates:
            return article_ids, fallback_scores, None
        X = build_features(user_features, candidates, session_events)
        combined = _combined_score(state.boosters, X)
        order = sorted(range(len(valid_ids)), key=lambda i: combined[i], reverse=True)
        ranked_ids = [valid_ids[i] for i in order]
        scores = {valid_ids[i]: float(combined[i]) for i in range(len(valid_ids))}
        return ranked_ids, scores, None
    except Exception:
        logger.exception("ranking failed, serving retrieval order")
        return article_ids, fallback_scores, "ranking"


# --- guardrails ---


def _guardrail_candidates(article_ids: list[int], scores: dict[int, float], state: AppState) -> list[Candidate]:
    out: list[Candidate] = []
    for article_id in article_ids:
        idx = state.article_id_to_index.get(article_id) if state.article_id_to_index else None
        meta = (state.item_meta or {}).get(article_id, DEFAULT_ITEM_META)
        embedding = state.item_vectors[idx].tolist() if idx is not None and state.item_vectors is not None else []
        out.append(
            Candidate(
                product_id=str(article_id),
                score=scores.get(article_id, 0.0),
                category_l1=meta.category_l1,
                in_stock=True,
                age_restricted=False,
                embedding=embedding,
            )
        )
    return out


def _apply_guardrails(
    candidates: list[Candidate], age_band: str, seen_ids: set[str], limit: int, request_id: str
) -> tuple[list[Candidate], GuardrailReport]:
    try:
        return apply_all(candidates, age_band=age_band, blocked_categories=set(), seen_ids=seen_ids, limit=limit)
    except Exception as exc:
        raise GuardrailsUnavailable(request_id) from exc


# --- reason codes ---


def _reason_for(intent_label: str, category_match_count: int) -> ReasonOut:
    if intent_label == COLD_START_LABEL:
        return ReasonOut(code="POPULAR", text="Trending with shoppers like you")
    if category_match_count > 0:
        return ReasonOut(code="SESSION_AFFINITY", text=f"Because you interacted with {category_match_count} similar items")
    return ReasonOut(code="INTENT_MATCH", text=f"Matches your {intent_label.replace('_', ' ')} intent")


# --- audit log (background task) ---


def _log_recommendation(
    state: AppState,
    request_id: str,
    user_id: str | None,
    session_id: str | None,
    surface: str,
    model_version: str,
    rows: list[tuple[str, float, str]],
) -> None:
    if state.pg_conn_factory is None:
        return
    try:
        pg_conn = state.pg_conn_factory()
        cursor = pg_conn.cursor()
        for position, (product_id, score, reason_code) in enumerate(rows):
            cursor.execute(
                "INSERT INTO recommendation_log "
                "(request_id, customer_id, session_id, article_id, rank_position, score, reason_code, model_version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (request_id, user_id, session_id, int(product_id), position, score, reason_code, model_version),
            )
        pg_conn.commit()
    except Exception:
        logger.exception("failed to record recommendation_log for surface=%s", surface)


# --- search query parsing ---


def _parse_query(redis_client: object, query: str, category_vocab: list[str]) -> dict:
    cache_key = f"qcache:{hashlib.sha1(query.strip().lower().encode()).hexdigest()}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        logger.exception("query cache lookup failed")

    tokens = re.findall(r"[a-z0-9]+", query.lower())
    matched_category = next(
        (c for c in category_vocab if any(tok in c.lower() or c.lower() in tok for tok in tokens if len(tok) > 2)),
        None,
    )
    parsed = {"keywords": tokens, "category_l1": matched_category}

    try:
        redis_client.setex(cache_key, QUERY_CACHE_TTL_SECONDS, json.dumps(parsed))
    except Exception:
        logger.exception("query cache write failed")
    return parsed


# --- complements (bundles) ---


def _complement_edges(state: AppState, seed_id: int) -> dict[int, float]:
    if state.pg_conn_factory is None:
        return {}
    try:
        pg_conn = state.pg_conn_factory()
        cursor = pg_conn.cursor()
        cursor.execute(
            "SELECT complement_article_id, score FROM complements WHERE article_id = %s ORDER BY score DESC LIMIT 50",
            (seed_id,),
        )
        rows = cursor.fetchall()
        return {int(r[0]): float(r[1]) for r in rows}
    except Exception:
        logger.exception("complement edge lookup failed")
        return {}


def _iso(value: object) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


# --- endpoints ---


@app.post("/v1/feed", response_model=FeedResponse)
def feed(request: FeedRequest, background_tasks: BackgroundTasks) -> FeedResponse:
    request_id = str(uuid.uuid4())
    region, age_band = _context_region_age(request.context)
    session_key = _session_key(request.user_id, request.session_id)

    with Timer("feed") as timer:
        events = _load_session_events(state.redis_client, session_key)
        intents = _infer_intents_safe(events, state.session_encoder, state.item_vectors, state.article_id_to_index)
        seen_ids = [e.article_id for e in events]

        candidate_ids, candidate_scores, source_intent, degraded = _get_feed_candidates(
            state, intents, seen_ids, region, age_band, state.settings.retrieval_merge_size
        )

        age_band_idx = (state.age_band_vocab or {}).get(age_band, 0)
        region_idx = (state.region_vocab or {}).get(region, 0)
        user_vector = _compute_user_vector(state.user_tower, events, state.item_vectors, state.article_id_to_index, age_band_idx, region_idx)
        user_features = UserFeatures(
            user_id=request.user_id or request.session_id,
            user_vector=user_vector,
            intent_vectors=[i.vector for i in intents],
            intent_weights=[i.weight for i in intents],
        )

        ranked_ids, scores, rank_degraded = _rank(user_features, candidate_ids, events, state, candidate_scores)
        degraded = degraded or rank_degraded

        guardrail_candidates = _guardrail_candidates(ranked_ids, scores, state)
        filtered, report = _apply_guardrails(
            guardrail_candidates, age_band, {str(a) for a in seen_ids}, request.limit, request_id
        )

        session_categories = Counter(e.category_l1 for e in events)
        items_out: list[ItemOut] = []
        for c in filtered:
            article_id = int(c.product_id)
            intent_label = source_intent.get(article_id, intents[0].label)
            category = (state.item_meta or {}).get(article_id, DEFAULT_ITEM_META).category_l1
            reason = _reason_for(intent_label, session_categories.get(category, 0))
            items_out.append(ItemOut(product_id=c.product_id, score=c.score, intent=intent_label, reason=reason))

        slot_counts = Counter(source_intent.get(int(c.product_id), intents[0].label) for c in filtered)
        detected_intents_out = [
            IntentOut(label=i.label, confidence=round(i.weight, 4), slots=slot_counts.get(i.label, 0)) for i in intents
        ]
        guardrails_out = GuardrailsOut(
            diversity_cap_applied=report.dropped.get("cap_category", 0) > 0,
            items_dropped=sum(report.dropped.values()),
            filters_fired={name: count for name, count in report.dropped.items() if count > 0},
        )

    response = FeedResponse(
        request_id=request_id,
        model_version=state.settings.model_version,
        detected_intents=detected_intents_out,
        items=items_out,
        guardrails=guardrails_out,
        degraded=degraded,
        latency_ms=round(timer.elapsed_ms),
    )
    background_tasks.add_task(
        _log_recommendation,
        state,
        request_id,
        request.user_id,
        request.session_id,
        "feed",
        state.settings.model_version,
        [(item.product_id, item.score, item.reason.code) for item in items_out],
    )
    return response


@app.post("/v1/bundles", response_model=BundlesResponse)
def bundles(request: BundlesRequest, background_tasks: BackgroundTasks) -> BundlesResponse:
    request_id = str(uuid.uuid4())
    seed_id = request.seed_product_id
    degraded: str | None = None

    with Timer("bundles") as timer:
        complement_scores = _complement_edges(state, seed_id)

        embedding_scores: dict[int, float] = {}
        seed_idx = (state.article_id_to_index or {}).get(seed_id)
        if seed_idx is not None:
            try:
                seed_vector = state.item_vectors[seed_idx].tolist()
                results = _call_retrieval(
                    state.http_client, state.settings.retrieval_service_url, [seed_vector], request.limit * 5, [seed_id]
                )
                _, embedding_scores, _ = _merge_candidates(results, ["similar"], state.settings.retrieval_merge_size)
            except Exception:
                logger.exception("bundles embedding-neighbour retrieval failed")
                degraded = "retrieval"

        blended: dict[int, float] = {}
        source: dict[int, str] = {}
        for article_id, weight in complement_scores.items():
            blended[article_id] = blended.get(article_id, 0.0) + 0.6 * weight
            source[article_id] = "COMPLEMENT"
        if embedding_scores:
            max_sim = max(embedding_scores.values()) or 1.0
            for article_id, sim in embedding_scores.items():
                blended[article_id] = blended.get(article_id, 0.0) + 0.4 * (sim / max_sim)
                source.setdefault(article_id, "SIMILAR_STYLE")

        ranked_ids = sorted(blended, key=lambda a: blended[a], reverse=True)
        guardrail_candidates = _guardrail_candidates(ranked_ids, blended, state)
        filtered, report = _apply_guardrails(guardrail_candidates, "unknown", {str(seed_id)}, request.limit, request_id)

        items_out = [
            ItemOut(
                product_id=c.product_id,
                score=c.score,
                intent=None,
                reason=ReasonOut(
                    code=source.get(int(c.product_id), "SIMILAR_STYLE"),
                    text="Frequently bought with this item" if source.get(int(c.product_id)) == "COMPLEMENT" else "Similar style",
                ),
            )
            for c in filtered
        ]
        guardrails_out = GuardrailsOut(
            diversity_cap_applied=report.dropped.get("cap_category", 0) > 0,
            items_dropped=sum(report.dropped.values()),
            filters_fired={name: count for name, count in report.dropped.items() if count > 0},
        )

    response = BundlesResponse(
        request_id=request_id,
        model_version=state.settings.model_version,
        seed_product_id=seed_id,
        items=items_out,
        guardrails=guardrails_out,
        degraded=degraded,
        latency_ms=round(timer.elapsed_ms),
    )
    background_tasks.add_task(
        _log_recommendation,
        state,
        request_id,
        request.user_id,
        None,
        "bundles",
        state.settings.model_version,
        [(item.product_id, item.score, item.reason.code) for item in items_out],
    )
    return response


@app.post("/v1/search", response_model=SearchResponseBody)
def search(request: SearchRequest, background_tasks: BackgroundTasks) -> SearchResponseBody:
    request_id = str(uuid.uuid4())
    degraded: str | None = None

    with Timer("search") as timer:
        parsed = _parse_query(state.redis_client, request.query, state.category_vocab or [])
        category = parsed.get("category_l1")
        if category and state.category_centroid and category in state.category_centroid:
            query_vector = state.category_centroid[category]
        else:
            query_vector = state.global_centroid.tolist()

        events: list[SessionEvent] = []
        if request.user_id or request.session_id:
            events = _load_session_events(state.redis_client, _session_key(request.user_id, request.session_id))
        seen_ids = [e.article_id for e in events]

        try:
            results = _call_retrieval(
                state.http_client, state.settings.retrieval_service_url, [query_vector], request.limit * 5, seen_ids
            )
            candidate_ids, candidate_scores, _ = _merge_candidates(results, ["search"], state.settings.retrieval_merge_size)
        except Exception:
            logger.exception("search retrieval failed, falling back to popularity")
            degraded = "retrieval"
            candidate_ids = _popularity_fallback(state.redis_client, "unknown", "unknown", request.limit * 3)
            candidate_scores = {a: 0.0 for a in candidate_ids}

        user_features = UserFeatures(
            user_id=request.user_id or request.session_id or "anonymous",
            user_vector=query_vector,
            intent_vectors=[query_vector],
            intent_weights=[1.0],
        )
        ranked_ids, scores, rank_degraded = _rank(user_features, candidate_ids, events, state, candidate_scores)
        degraded = degraded or rank_degraded

        guardrail_candidates = _guardrail_candidates(ranked_ids, scores, state)
        filtered, report = _apply_guardrails(guardrail_candidates, "unknown", {str(a) for a in seen_ids}, request.limit, request_id)

        items_out = [
            ItemOut(
                product_id=c.product_id,
                score=c.score,
                intent="search",
                reason=ReasonOut(code="SEARCH_MATCH", text=f"Matches your search for '{request.query}'"),
            )
            for c in filtered
        ]
        guardrails_out = GuardrailsOut(
            diversity_cap_applied=report.dropped.get("cap_category", 0) > 0,
            items_dropped=sum(report.dropped.values()),
            filters_fired={name: count for name, count in report.dropped.items() if count > 0},
        )

    response = SearchResponseBody(
        request_id=request_id,
        model_version=state.settings.model_version,
        query=request.query,
        parsed_intent=parsed,
        items=items_out,
        guardrails=guardrails_out,
        degraded=degraded,
        latency_ms=round(timer.elapsed_ms),
    )
    background_tasks.add_task(
        _log_recommendation,
        state,
        request_id,
        request.user_id,
        request.session_id,
        "search",
        state.settings.model_version,
        [(item.product_id, item.score, item.reason.code) for item in items_out],
    )
    return response


@app.post("/v1/events", status_code=202)
def events(request: EventRequest) -> Response:
    key = _session_key(request.user_id, request.session_id)
    event = {
        "article_id": request.article_id,
        "timestamp": (request.timestamp or datetime.utcnow()).isoformat(),
        "price": request.price or 0.0,
        "category_l1": request.category_l1 or "unknown",
        "discounted": request.discounted,
    }
    try:
        state.redis_client.lpush(key, json.dumps(event))
        state.redis_client.ltrim(key, 0, SESSION_LENGTH - 1)
        state.redis_client.expire(key, SESSION_TTL_SECONDS)
    except Exception:
        logger.exception("failed to record event")
    return Response(status_code=202)


@app.delete("/v1/users/{user_id}", status_code=204)
def delete_user(user_id: str) -> Response:
    if state.pg_conn_factory is None:
        return _error_response("CATALOG_UNAVAILABLE", "postgres unavailable, erasure could not be completed", str(uuid.uuid4()), True, 503)
    try:
        pg_conn = state.pg_conn_factory()
    except Exception:
        logger.exception("postgres unavailable for erasure")
        return _error_response("CATALOG_UNAVAILABLE", "postgres unavailable, erasure could not be completed", str(uuid.uuid4()), True, 503)

    receipt = erase_user(user_id, pg_conn, state.redis_client, _NULL_INDEX_HANDLE)
    response = Response(status_code=204)
    response.headers["X-Deletion-Receipt-Id"] = receipt.receipt_id
    return response


@app.get("/v1/users/{user_id}/data")
def get_user_data(user_id: str) -> dict:
    session_events: list[dict] = []
    try:
        raw = state.redis_client.lrange(_session_key(user_id, None), 0, -1)
        session_events = [json.loads(r) for r in raw]
    except Exception:
        logger.exception("failed to load session for data export")

    consent_rows: list[dict] = []
    recommendation_rows: list[dict] = []
    if state.pg_conn_factory is not None:
        try:
            pg_conn = state.pg_conn_factory()
            cursor = pg_conn.cursor()
            cursor.execute(
                "SELECT consent_type, granted, granted_at, revoked_at FROM consent WHERE customer_id = %s", (user_id,)
            )
            consent_rows = [
                {"purpose": r[0], "granted": r[1], "granted_at": _iso(r[2]), "revoked_at": _iso(r[3])}
                for r in cursor.fetchall()
            ]
            cursor.execute(
                "SELECT article_id, rank_position, reason_code, model_version, served_at FROM recommendation_log "
                "WHERE customer_id = %s ORDER BY served_at DESC LIMIT 100",
                (user_id,),
            )
            recommendation_rows = [
                {"article_id": r[0], "rank_position": r[1], "reason_code": r[2], "model_version": r[3], "served_at": _iso(r[4])}
                for r in cursor.fetchall()
            ]
        except Exception:
            logger.exception("failed to load postgres data for export")

    return {"user_id": user_id, "session_events": session_events, "consent": consent_rows, "recommendation_log": recommendation_rows}


@app.post("/v1/consent")
def consent(request: ConsentRequest) -> dict:
    if state.pg_conn_factory is None:
        return _error_response("CATALOG_UNAVAILABLE", "postgres unavailable", str(uuid.uuid4()), True, 503)
    try:
        pg_conn = state.pg_conn_factory()
        cursor = pg_conn.cursor()
        cursor.execute(
            "UPDATE consent SET revoked_at = now() WHERE customer_id = %s AND consent_type = %s AND revoked_at IS NULL",
            (request.user_id, request.purpose),
        )
        cursor.execute(
            "INSERT INTO consent (customer_id, consent_type, granted, granted_at, source) VALUES (%s, %s, %s, now(), %s)",
            (request.user_id, request.purpose, request.granted, "api"),
        )
        pg_conn.commit()
    except Exception:
        logger.exception("failed to record consent")
        return _error_response("CATALOG_UNAVAILABLE", "consent could not be recorded", str(uuid.uuid4()), True, 503)
    return {"user_id": request.user_id, "purpose": request.purpose, "granted": request.granted}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    lines = [
        "# HELP discovery_engine_latency_ms Latency per pipeline stage in milliseconds.",
        "# TYPE discovery_engine_latency_ms_count counter",
        "# TYPE discovery_engine_latency_ms_sum counter",
    ]
    for name, samples in telemetry.LATENCY.items():
        if not samples:
            continue
        lines.append(f'discovery_engine_latency_ms_count{{stage="{name}"}} {len(samples)}')
        lines.append(f'discovery_engine_latency_ms_sum{{stage="{name}"}} {sum(samples):.3f}')
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
