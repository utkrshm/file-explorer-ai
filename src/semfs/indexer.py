"""Extract supported content, chunk it, and incrementally update the SQLite cache."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semfs.database import Database
from semfs.file_manager import TEXT_TYPES, FileManager

IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
SUPPORTED_TYPES = TEXT_TYPES | IMAGE_TYPES | {".pdf"}
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    text: str
    location: str


@dataclass
class IndexReport:
    indexed: int = 0
    reused: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False


def chunk_text(
    text: str, label: str = "", size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than a positive chunk size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        passage = text[start:end].strip()
        if passage:
            heading = next(
                (
                    line.lstrip("# ").strip()
                    for line in reversed(text[:start].splitlines())
                    if line.startswith("#")
                ),
                "",
            )
            location = label or (
                f"Section: {heading}" if heading else f"Line {text.count(chr(10), 0, start) + 1}"
            )
            chunks.append(Chunk(passage, location))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract(path: Path) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_TYPES:
        return [Chunk("", "Image (native vision embedding)")]
    if suffix == ".pdf":
        import pymupdf

        chunks = []
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDF")
            for number, page in enumerate(document, 1):
                chunks.extend(chunk_text(page.get_text(), f"Page {number}"))
        return chunks
    if suffix in TEXT_TYPES:
        # A binary file with a text extension must not pollute the semantic index.
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\0" in text:
            raise ValueError("Binary content in a text file")
        return chunk_text(text)
    return []


def index_files(
    manager: FileManager,
    database: Database,
    embedder,
    progress: Callable[[int, int, str], None] = lambda *_: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> IndexReport:
    report = IndexReport()
    metadata = database.metadata()
    files = [
        path
        for path in manager.walk_files(lambda error: report.errors.append(str(error)))
        if path.suffix.lower() in SUPPORTED_TYPES
    ]
    for number, path in enumerate(files, 1):
        if cancelled():
            report.cancelled = True
            break
        relative = str(path.relative_to(manager.root))
        progress(number - 1, len(files), f"Indexing {relative}")
        try:
            path = manager.safe(path)
            before = path.stat()
            old = metadata.get(relative)
            if old and (old["size"], old["mtime_ns"], old["model_id"]) == (
                before.st_size,
                before.st_mtime_ns,
                embedder.model_id,
            ):
                report.reused += 1
                progress(number, len(files), f"Reused {relative}")
                continue
            chunks = extract(path)
            if chunks:
                if path.suffix.lower() in IMAGE_TYPES:
                    vectors = embedder.image(path)
                else:
                    # Small batches keep inference memory bounded for long PDFs.
                    batches = []
                    for offset in range(0, len(chunks), 8):
                        if cancelled():
                            report.cancelled = True
                            return report
                        batches.append(
                            embedder.documents(
                                [chunk.text for chunk in chunks[offset : offset + 8]], path.name
                            )
                        )
                    vectors = np.concatenate(batches)
            else:
                vectors = np.empty((0, 768), dtype=np.float32)
            after = manager.safe(path).stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise OSError("File changed during indexing; scan again")
            if cancelled():
                report.cancelled = True
                return report
            database.replace(path, after, embedder.model_id, chunks, vectors)
            report.indexed += 1
        except (OSError, ValueError, RuntimeError) as error:
            report.errors.append(f"{relative}: {error}")
        progress(number, len(files), f"Processed {relative}")

    if not report.cancelled:
        # Only remove entries known to be gone. Unreadable directories are not deletions.
        for relative in metadata:
            path = manager.root / relative
            try:
                safe = manager.safe(path)
                missing = not safe.is_file() or safe.suffix.lower() not in SUPPORTED_TYPES
            except ValueError:
                missing = True
            except OSError:
                continue
            if missing:
                database.remove(relative)
                report.removed += 1
    progress(len(files), len(files), "Index scan finished")
    return report
