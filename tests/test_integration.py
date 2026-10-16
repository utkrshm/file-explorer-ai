"""Opt-in test: downloads the real checkpoint and performs local multimodal inference."""

import os

import pytest

from semfs.embeddings import Embedder
from semfs.indexer import index_files
from semfs.search import semantic_search


@pytest.fixture(scope="module")
def real_embedder():
    return Embedder()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SEMFS_REAL_MODEL") != "1",
    reason="Set SEMFS_REAL_MODEL=1 to download and run the real model",
)
def test_real_embeddinggemma_text_and_image_retrieval(indexed, real_embedder):
    manager, database = indexed
    (manager.root / "os.txt").write_text(
        "A deadlock occurs when processes wait indefinitely for each other's resources. "
        "Mutexes and semaphores synchronize threads. Lock ordering prevents circular wait."
    )
    (manager.root / "garden.txt").write_text(
        "Tomatoes grow best in sunny gardens with fertile soil and regular watering."
    )
    from PIL import Image, ImageDraw

    for number, color in enumerate(["red", "blue"]):
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 236, 236), fill=color)
        image.save(manager.root / f"scene_{number}.png")
    model = real_embedder
    report = index_files(manager, database, model)
    assert not report.errors, report.errors
    assert report.indexed == 4
    results = semantic_search(database, model, "processes stuck waiting for locks")
    assert results[0].path.name == "os.txt"
    images = semantic_search(database, model, "a red square on a white background")
    image_paths = [item.path.name for item in images if item.path.suffix == ".png"]
    assert image_paths[0] == "scene_0.png"
    assert all(-1 <= item.score <= 1 for item in results)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SEMFS_REAL_MODEL") != "1"
    or not all(
        os.environ.get(key)
        for key in ("SEMFS_TEST_ROOT", "SEMFS_TEST_QUERY", "SEMFS_EXPECTED_PATH")
    ),
    reason="Set SEMFS_REAL_MODEL, SEMFS_TEST_ROOT, SEMFS_TEST_QUERY and SEMFS_EXPECTED_PATH",
)
def test_real_project_sample(tmp_path, monkeypatch, real_embedder):
    import json
    from pathlib import Path

    import numpy as np

    from semfs.database import Database
    from semfs.file_manager import TEXT_TYPES, FileManager

    manager = FileManager(Path(os.environ["SEMFS_TEST_ROOT"]))
    candidates = []
    for path in manager.walk_files():
        if path.suffix.lower() not in TEXT_TYPES | {".pdf", ".png", ".jpg", ".jpeg"}:
            continue
        try:
            size = path.stat().st_size
            if 0 < size <= 2_000_000:
                candidates.append(path)
        except OSError:
            continue

    readmes = sorted(
        (
            path
            for path in candidates
            if path.name.casefold() == "readme.md"
            and len(path.relative_to(manager.root).parts) == 2
        ),
        key=str,
    )[:10]
    code = sorted(
        (
            path
            for path in candidates
            if path.suffix.lower() in TEXT_TYPES - {".txt", ".md"} and path.stat().st_size <= 20_000
        ),
        key=str,
    )[:4]
    pdfs = sorted((path for path in candidates if path.suffix.lower() == ".pdf"), key=str)[:1]
    images = sorted(
        (path for path in candidates if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=str,
    )[:2]
    expected = manager.safe(manager.root / os.environ["SEMFS_EXPECTED_PATH"])
    sample = list(dict.fromkeys([expected] + readmes + code + pdfs + images))
    assert readmes and code and images, (
        "The selected real-project sample needs documents/code/images"
    )

    # Only the traversal is bounded; extraction, model inference, SQLite storage,
    # cosine retrieval, and freshness checks all use the actual application code.
    monkeypatch.setattr(manager, "walk_files", lambda on_error=None: iter(sample))
    database = Database(manager, tmp_path / "real-project.sqlite3")
    model = real_embedder
    report = index_files(
        manager,
        database,
        model,
        progress=lambda done, total, message: print(f"{done}/{total} {message}", flush=True),
    )
    assert not report.errors, report.errors
    assert report.indexed == len(sample)
    chunks = database.chunks(model.model_id)
    assert all(chunk.embedding.shape == (768,) for chunk in chunks)
    assert all(np.isfinite(chunk.embedding).all() for chunk in chunks)
    assert any(chunk.path in images for chunk in chunks)
    query = (
        "implementing denoising diffusion probabilistic models with noise schedules from scratch"
    )
    results = semantic_search(database, model, query)
    assert results[0].path == expected
    before = model._model
    reused = index_files(manager, database, model)
    assert reused.reused == len(sample)
    assert model._model is before
    reopened = Database(manager, database.path)
    repeated = semantic_search(reopened, model, query)
    assert repeated[0].path == results[0].path
    assert all(-1 <= result.score <= 1 for result in results)

    output = {
        "root": str(manager.root),
        "model": model.model_id,
        "sample_files": [str(path.relative_to(manager.root)) for path in sample],
        "chunks": len(chunks),
        "indexed": report.indexed,
        "reused": reused.reused,
        "query": query,
        "results": [
            {
                "path": str(item.path.relative_to(manager.root)),
                "score": item.score,
                "location": item.location,
            }
            for item in results[:5]
        ],
    }
    destination = Path(os.environ.get("SEMFS_REPORT_DIR", tmp_path))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "real-project-result.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)
