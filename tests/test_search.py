import numpy as np
import pytest

from semfs.database import CachedChunk
from semfs.file_manager import FileManager
from semfs.indexer import index_files
from semfs.search import (
    MIN_SCORE,
    filename_search,
    rank_chunks,
    semantic_search,
    similar_files,
)


class FakeEmbedder:
    model_id = "test-model"

    def __init__(self):
        self.calls = 0

    def documents(self, texts, title):
        self.calls += 1
        # Only a test double: production never uses this keyword-based representation.
        return np.array([self.query(text) for text in texts], dtype=np.float32)

    def query(self, text):
        text = text.casefold()
        vector = np.array(
            [
                text.count("deadlock"),
                sum(text.count(word) for word in ["mutex", "semaphore"]),
                sum(text.count(word) for word in ["tomato", "garden", "soil"]),
                0.01,
            ],
            dtype=np.float32,
        )
        return vector / np.linalg.norm(vector)

    def image(self, path):
        self.calls += 1
        return np.array([[0, 0, 0, 1]], dtype=np.float32)


def test_filename_search(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "OS_notes.PDF").write_text("notes")
    (tmp_path / "dijkstra.py").write_text("code")
    manager = FileManager(tmp_path)
    assert filename_search(manager, "notes") == [tmp_path / "nested" / "OS_notes.PDF"]
    assert filename_search(manager, ".py") == [tmp_path / "dijkstra.py"]
    assert filename_search(manager, "*.pdf") == [tmp_path / "nested" / "OS_notes.PDF"]
    assert filename_search(manager, "") == []


def test_cosine_scores_and_chunk_grouping(tmp_path):
    file = tmp_path / "a.txt"
    chunks = [
        CachedChunk(file, "first", "Page 1", np.array([0.5, 0.5], dtype=np.float32)),
        CachedChunk(file, "best", "Page 2", np.array([4, 0], dtype=np.float32)),
        CachedChunk(tmp_path / "b.txt", "other", "Line 1", np.array([0, 3], dtype=np.float32)),
    ]
    results = rank_chunks(chunks, np.array([2, 0], dtype=np.float32))
    assert len(results) == 2
    assert results[0].score == pytest.approx(1.0)
    assert results[0].passage == "best"
    assert results[0].location == "Page 2"
    assert results[1].score == pytest.approx(0.0)


def test_semantic_retrieval_and_similar_files(indexed):
    manager, database = indexed
    (manager.root / "os.txt").write_text("deadlock prevention with mutex and semaphore")
    (manager.root / "locks.txt").write_text("mutex and semaphore synchronization")
    (manager.root / "garden.txt").write_text("tomato garden soil")
    embedder = FakeEmbedder()
    index_files(manager, database, embedder)
    results = semantic_search(database, embedder, "deadlock mutex semaphore")
    # garden.txt scores 0 against this query, below the AI search threshold.
    assert [result.path.name for result in results] == ["os.txt", "locks.txt"]
    assert all(result.score >= MIN_SCORE for result in results)
    # "deadlock" alone scores about 0.45 against os.txt.
    assert semantic_search(database, embedder, "deadlock") == []
    assert semantic_search(database, embedder, "deadlock", min_score=0.4)[0].path.name == "os.txt"
    related = similar_files(database, embedder, manager.root / "os.txt")
    assert related[0].path.name == "locks.txt"
    assert all(result.path.name != "os.txt" for result in related)


def test_empty_and_invalid_queries(indexed):
    _, database = indexed
    assert semantic_search(database, FakeEmbedder(), "notes") == []
    with pytest.raises(ValueError):
        rank_chunks([], np.zeros(3))
    with pytest.raises(ValueError):
        rank_chunks([], np.array([np.nan]))
