import pytest

from semfs.database import Database
from semfs.file_manager import FileManager


def symlink(link, target, *, target_is_directory=False):
    """Create a symlink, skipping the test where the OS does not allow it (e.g. Windows)."""
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable: {error}")


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
