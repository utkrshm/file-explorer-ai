import pytest

from semfs.database import Database
from semfs.file_manager import FileManager


@pytest.fixture(autouse=True)
def temporary_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMFS_CACHE_DIR", str(tmp_path.parent / f"{tmp_path.name}-cache"))


@pytest.fixture
def indexed(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    manager = FileManager(root)
    database = Database(manager, tmp_path / "index.sqlite3")
    return manager, database
