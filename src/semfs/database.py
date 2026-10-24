"""A root-specific SQLite cache. Each method uses its own short-lived connection."""

import getpass
import hashlib
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semfs.file_manager import FileManager


def cache_path(root: Path) -> Path:
    root = root.resolve()
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    preferred = Path(os.environ.get("SEMFS_CACHE_DIR", str(Path(base) / "semfs"))).expanduser()
    # Managing the home directory must not put its own index inside that directory.
    candidates = [preferred, Path(tempfile.gettempdir()) / f"semfs-{getpass.getuser()}"]
    for directory in candidates:
        if not directory.resolve().is_relative_to(root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            return directory / f"{digest}.sqlite3"
    raise ValueError("No cache location outside this root. Choose a smaller root directory.")


@dataclass
class CachedChunk:
    path: Path
    text: str
    location: str
    embedding: np.ndarray


class Database:
    def __init__(self, manager: FileManager, path: Path | None = None):
        self.manager = manager
        self.path = (path or cache_path(manager.root)).resolve()
        if self.path.is_relative_to(manager.root):
            raise ValueError("The SQLite cache must be outside the managed root")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    model_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    location TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (file_path, position)
                );
            """)
            row = connection.execute("SELECT value FROM settings WHERE key='root'").fetchone()
            if row and row[0] != str(manager.root):
                raise ValueError("This database belongs to a different root directory")
            connection.execute(
                "INSERT OR IGNORE INTO settings VALUES ('root', ?)", (str(manager.root),)
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def metadata(self) -> dict[str, dict]:
        with self.connect() as connection:
            return {row["path"]: dict(row) for row in connection.execute("SELECT * FROM files")}

    def replace(
        self, path: Path, stat: os.stat_result, model_id: str, chunks: list, embeddings: np.ndarray
    ) -> None:
        path = self.manager.safe(path)
        relative = str(path.relative_to(self.manager.root))
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(chunks) != len(vectors) or not np.isfinite(vectors).all():
            raise ValueError("Model returned invalid embeddings")
        if len(vectors) and np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Model returned an empty embedding")
        with self.connect() as connection:
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
            connection.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                (relative, stat.st_size, stat.st_mtime_ns, path.suffix.lower(), model_id),
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (relative, index, chunk.text, chunk.location, vector.tobytes())
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )

    def remove(self, relative: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM files WHERE path=?", (relative,))

    def chunks(self, model_id: str) -> list[CachedChunk]:
        """Exclude changed, deleted, or unsafe files even before the next index scan."""
        valid: set[str] = set()
        for relative, info in self.metadata().items():
            if info["model_id"] != model_id:
                continue
            try:
                path = self.manager.safe(self.manager.root / relative)
                stat = path.stat()
                if path.is_file() and (stat.st_size, stat.st_mtime_ns) == (
                    info["size"],
                    info["mtime_ns"],
                ):
                    valid.add(relative)
            except (OSError, ValueError):
                continue
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT c.* FROM chunks c JOIN files f ON c.file_path=f.path "
                "WHERE f.model_id=? ORDER BY c.file_path, c.position",
                (model_id,),
            ).fetchall()
        return [
            CachedChunk(
                self.manager.root / row["file_path"],
                row["text"],
                row["location"],
                np.frombuffer(row["embedding"], dtype=np.float32).copy(),
            )
            for row in rows
            if row["file_path"] in valid
        ]
