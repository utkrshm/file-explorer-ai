from unittest.mock import patch

import numpy as np
import pytest
from test_search import FakeEmbedder

from semfs.database import Database, cache_path
from semfs.indexer import Chunk, chunk_text, extract, index_files


def test_windows_cache_location(tmp_path, monkeypatch):
    root = tmp_path / "managed"
    root.mkdir()
    monkeypatch.setattr("semfs.database.sys.platform", "win32")
    monkeypatch.delenv("SEMFS_CACHE_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert cache_path(root).parent == tmp_path / "local" / "semfs"


def test_persistence_and_reuse(indexed):
    manager, database = indexed
    (manager.root / "os.md").write_text("deadlock and semaphore")
    model = FakeEmbedder()
    report = index_files(manager, database, model)
    assert report.indexed == 1
    assert model.calls == 1
    reopened = Database(manager, database.path)
    assert len(reopened.chunks(model.model_id)) == 1
    assert reopened.chunks(model.model_id)[0].text == "deadlock and semaphore"
    report = index_files(manager, reopened, model)
    assert report.reused == 1
    assert model.calls == 1


def test_updates_deletions_and_model_changes(indexed):
    manager, database = indexed
    file = manager.root / "os.txt"
    file.write_text("deadlock")
    model = FakeEmbedder()
    index_files(manager, database, model)
    file.write_text("tomato garden soil")
    # Stale vectors must not be searchable even before reindexing.
    assert database.chunks(model.model_id) == []
    assert index_files(manager, database, model).indexed == 1
    assert database.chunks(model.model_id)[0].text == "tomato garden soil"
    model.model_id = "different-test-model"
    assert index_files(manager, database, model).indexed == 1
    assert database.chunks("test-model") == []
    file.unlink()
    assert index_files(manager, database, model).removed == 1
    assert database.metadata() == {}


def test_overlap_and_pdf_pages(tmp_path):
    chunks = chunk_text("abcdefghijklmnopqrst", size=10, overlap=3)
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[-1].text.endswith("t")
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=10)
    import pymupdf

    pdf = tmp_path / "notes.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "Deadlock prevention")
        document.save(pdf)
    assert extract(pdf)[0].location == "Page 1"
    assert "Deadlock" in extract(pdf)[0].text


def test_native_image_indexing(indexed):
    manager, database = indexed
    from PIL import Image

    image = manager.root / "picture.png"
    Image.new("RGB", (16, 16), "red").save(image)
    model = FakeEmbedder()
    assert index_files(manager, database, model).indexed == 1
    assert model.calls == 1
    assert database.chunks(model.model_id)[0].location.startswith("Image")


def test_empty_unsupported_and_binary_files(indexed):
    manager, database = indexed
    (manager.root / "empty.txt").write_text("")
    (manager.root / "binary.txt").write_bytes(b"\0\1\2")
    (manager.root / "movie.mp4").write_bytes(b"not indexed")
    report = index_files(manager, database, FakeEmbedder())
    assert report.indexed == 1
    assert len(report.errors) == 1
    assert set(database.metadata()) == {"empty.txt"}


def test_failed_extraction_and_cancel(indexed):
    manager, database = indexed
    (manager.root / "locked.txt").write_text("deadlock")
    with patch("semfs.indexer.extract", side_effect=PermissionError("denied")):
        report = index_files(manager, database, FakeEmbedder())
    assert len(report.errors) == 1
    assert database.metadata() == {}
    report = index_files(manager, database, FakeEmbedder(), cancelled=lambda: True)
    assert report.cancelled
    assert database.metadata() == {}


def test_database_containment_and_transaction(indexed):
    manager, database = indexed
    with pytest.raises(ValueError, match="outside"):
        Database(manager, manager.root / "bad.sqlite3")
    file = manager.root / "os.txt"
    file.write_text("deadlock")
    index_files(manager, database, FakeEmbedder())
    with pytest.raises(ValueError):
        database.replace(
            file, file.stat(), "test-model", [Chunk("bad", "Line 1")], np.array([[np.nan, 1]])
        )
    assert database.chunks("test-model")[0].text == "deadlock"


def test_directory_scan_error_does_not_purge_good_cache(indexed):
    manager, database = indexed
    (manager.root / "os.txt").write_text("deadlock")
    index_files(manager, database, FakeEmbedder())

    def inaccessible(on_error):
        on_error(PermissionError("unreadable directory"))
        return iter([])

    with patch.object(manager, "walk_files", inaccessible):
        report = index_files(manager, database, FakeEmbedder())
    assert report.errors
    assert "os.txt" in database.metadata()
