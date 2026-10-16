"""Exact duplicates: size grouping followed by streaming SHA-256."""

import hashlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from semfs.file_manager import FileManager


@dataclass
class DuplicateGroup:
    paths: list[Path]
    size: int
    digest: str

    @property
    def wasted(self) -> int:
        return self.size * (len(self.paths) - 1)


@dataclass
class DuplicateScan:
    groups: list[DuplicateGroup] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def wasted(self) -> int:
        return sum(group.wasted for group in self.groups)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def find_duplicates(
    manager: FileManager,
    progress: Callable[[str], None] = lambda _: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> DuplicateScan:
    result = DuplicateScan()
    sizes: dict[int, list[Path]] = defaultdict(list)
    inodes: set[tuple[int, int]] = set()
    for path in manager.walk_files(lambda error: result.errors.append(str(error))):
        if cancelled():
            return result
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            # Hard links share storage already, so don't call them wasted copies.
            if identity not in inodes:
                sizes[stat.st_size].append(path)
                inodes.add(identity)
        except OSError as error:
            result.errors.append(f"{path.name}: {error}")
    for size, paths in sizes.items():
        if len(paths) < 2:
            continue
        hashes: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            if cancelled():
                return result
            progress(f"Hashing {path.relative_to(manager.root)}")
            try:
                path = manager.safe(path)
                before = path.stat()
                digest = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ino,
                ):
                    raise OSError("File changed while hashing; run the scan again")
                hashes[digest].append(path)
            except (OSError, ValueError) as error:
                result.errors.append(f"{path.name}: {error}")
        result.groups.extend(
            DuplicateGroup(sorted(matches), size, digest)
            for digest, matches in hashes.items()
            if len(matches) > 1
        )
    result.groups.sort(key=lambda group: (-group.wasted, str(group.paths[0])))
    return result
