"""Filename search and cosine ranking, grouped by file."""

import fnmatch
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semfs.file_manager import FileManager


@dataclass
class SearchResult:
    path: Path
    score: float
    passage: str
    location: str


def filename_search(manager: FileManager, query: str) -> list[Path]:
    query = query.strip().casefold()
    if not query:
        return []
    errors: list[OSError] = []
    paths = [
        path
        for path in manager.walk_files(errors.append)
        if query in path.name.casefold() or fnmatch.fnmatchcase(path.name.casefold(), query)
    ]
    return sorted(paths, key=lambda path: str(path.relative_to(manager.root)).casefold())


def rank_chunks(
    chunks: list, query: np.ndarray, limit: int = 30, exclude: Path | None = None
) -> list[SearchResult]:
    """Take each file's best chunk score, not a sum biased toward long files."""
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not norm:
        raise ValueError("Query embedding is invalid")
    best: dict[Path, SearchResult] = {}
    for chunk in chunks:
        if chunk.path == exclude:
            continue
        embedding = chunk.embedding
        if embedding.shape != vector.shape or not np.isfinite(embedding).all():
            continue
        denominator = np.linalg.norm(embedding) * norm
        if not denominator:
            continue
        score = float(np.clip(np.dot(vector, embedding) / denominator, -1, 1))
        result = SearchResult(chunk.path, score, chunk.text, chunk.location)
        if chunk.path not in best or score > best[chunk.path].score:
            best[chunk.path] = result
    return sorted(best.values(), key=lambda item: (-item.score, str(item.path)))[:limit]


def semantic_search(database, embedder, query: str, limit: int = 30) -> list[SearchResult]:
    if not query.strip():
        return []
    chunks = database.chunks(embedder.model_id)
    if not chunks:
        return []
    return rank_chunks(chunks, embedder.query(query), limit)


def similar_files(database, embedder, path: Path, limit: int = 30) -> list[SearchResult]:
    """Compare cached document vectors in the same space, without a query prefix."""
    path = database.manager.safe(path)
    chunks = database.chunks(embedder.model_id)
    source = [chunk.embedding for chunk in chunks if chunk.path == path]
    if not source:
        raise ValueError("This file is not indexed or has changed. Press i to index it.")
    # A normalized centroid represents all chunks, rather than just the first page.
    vector = np.mean(source, axis=0)
    return rank_chunks(chunks, vector, limit, exclude=path)
