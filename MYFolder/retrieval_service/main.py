from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
ID_MAP_PATH = ARTIFACTS_DIR / "id_map.json"


class SearchRequest(BaseModel):
    vectors: list[list[float]]
    k: int
    exclude: list[int] = Field(default_factory=list)


class SearchResult(BaseModel):
    id: int
    score: float


class SearchResponse(BaseModel):
    results: list[list[SearchResult]]
    took_ms: float


class AppState:
    index: faiss.Index | None = None
    index_to_article_id: dict[int, int] | None = None


state = AppState()


def load_index(index_path: Path = INDEX_PATH, id_map_path: Path = ID_MAP_PATH) -> tuple[faiss.Index, dict[int, int]]:
    index = faiss.read_index(str(index_path))
    with id_map_path.open() as f:
        raw = json.load(f)
    index_to_article_id = {int(k): v for k, v in raw["index_to_article_id"].items()}
    return index, index_to_article_id


def search_batch(
    index: faiss.Index,
    index_to_article_id: dict[int, int],
    vectors: np.ndarray,
    k: int,
    exclude: set[int],
) -> list[list[SearchResult]]:
    fetch_k = min(index.ntotal, k + len(exclude))
    scores, indices = index.search(np.ascontiguousarray(vectors, dtype=np.float32), fetch_k)

    results: list[list[SearchResult]] = []
    for row_scores, row_indices in zip(scores, indices):
        row_results: list[SearchResult] = []
        for score, idx in zip(row_scores, row_indices):
            if idx < 0:
                continue
            article_id = index_to_article_id[int(idx)]
            if article_id in exclude:
                continue
            row_results.append(SearchResult(id=article_id, score=float(score)))
            if len(row_results) >= k:
                break
        results.append(row_results)
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    if INDEX_PATH.exists() and ID_MAP_PATH.exists():
        state.index, state.index_to_article_id = load_index()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    if state.index is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return {"status": "ready"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if state.index is None or state.index_to_article_id is None:
        raise HTTPException(status_code=503, detail="index not loaded")

    vectors = np.array(request.vectors, dtype=np.float32)
    start = time.perf_counter()
    results = search_batch(state.index, state.index_to_article_id, vectors, request.k, set(request.exclude))
    took_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(results=results, took_ms=took_ms)
