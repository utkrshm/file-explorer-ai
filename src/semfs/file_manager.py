"""Real filesystem operations, confined to one root; symlinks are never followed."""

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

TEXT_TYPES = {".txt", ".md", ".py", ".c", ".cpp", ".h", ".rs", ".java", ".js"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".cache", ".pytest_cache", ".ruff_cache"}


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return str(size)


class FileManager:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(str(self.root))
        self.current = self.root

    def safe(self, path: Path | str, *, allow_root: bool = True) -> Path:
        """Validate lexical containment and every component before touching a path."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.current / candidate
        candidate = Path(os.path.abspath(candidate))
        if not candidate.is_relative_to(self.root):
            raise ValueError("Path must stay inside the selected root")
        if candidate == self.root and not allow_root:
            raise ValueError("The selected root cannot be modified")
        part = self.root
        for name in candidate.relative_to(self.root).parts:
            part /= name
            if part.is_symlink():
                raise ValueError("Symlinks are displayed but never followed or modified")
        if not candidate.resolve().is_relative_to(self.root):
            raise ValueError("Path resolves outside the selected root")
        return candidate

    def entries(self, sort: str = "name") -> list[Path]:
        directory = self.safe(self.current)

        def key(path: Path) -> tuple:
            try:
                stat = path.lstat()
                value = {"size": stat.st_size, "modified": stat.st_mtime_ns}.get(
                    sort, path.name.casefold()
                )
                return (not (not path.is_symlink() and path.is_dir()), value, path.name)
            except OSError:
                return (True, 0 if sort != "name" else "", path.name)

        return sorted(directory.iterdir(), key=key)

    def navigate(self, path: Path | str) -> Path:
        destination = self.safe(path)
        if not destination.is_dir():
            raise NotADirectoryError(str(destination))
        self.current = destination
        return destination

    def parent(self) -> Path:
        return self.navigate(self.current.parent if self.current != self.root else self.root)

    def create(self, name: str, *, directory: bool = False) -> Path:
        if not name.strip():
            raise ValueError("Enter a name")
        path = self.safe(name, allow_root=False)
        if directory:
            path.mkdir()
        else:
            with path.open("x"):
                pass
        return path

    def rename(self, path: Path, name: str) -> Path:
        source = self.safe(path, allow_root=False)
        if not name.strip() or name in {".", ".."} or Path(name).name != name:
            raise ValueError("Enter a single file or folder name")
        target = self.safe(source.with_name(name), allow_root=False)
        if target.exists():
            raise FileExistsError(str(target))
        return source.rename(target)

    def transfer(self, path: Path, destination: str | Path, *, move: bool = False) -> Path:
        source = self.safe(path, allow_root=False)
        target = self.safe(destination)
        if target.is_dir():
            target = self.safe(target / source.name, allow_root=False)
        if target == self.root or target.exists():
            raise FileExistsError(f"Destination already exists: {target}")
        if target.is_relative_to(source):
            raise ValueError("Cannot copy or move a directory inside itself")
        if source.is_dir():
            # copytree otherwise follows links nested inside the source directory.
            for directory, dirs, files in os.walk(source, followlinks=False, onerror=_raise):
                for name in dirs + files:
                    self.safe(Path(directory) / name)
        if move:
            shutil.move(str(source), str(target))
        elif source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return target

    def delete(self, path: Path) -> None:
        path = self.safe(path, allow_root=False)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def preview(self, path: Path, limit: int = 64_000) -> str:
        path = self.safe(path)
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if b"\0" in data:
            return "Binary file - use o to open with the default application."
        text = data[:limit].decode("utf-8", errors="replace")
        return text + ("\n\n[Preview truncated]" if len(data) > limit else "")

    def open_default(self, path: Path) -> None:
        path = self.safe(path)
        if sys.platform == "win32":
            os.startfile(path)
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen(
            [opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def editor_command(self, path: Path) -> list[str]:
        path = self.safe(path)
        if not path.is_file():
            raise ValueError("Select a text file to edit")
        with path.open("rb") as stream:
            data = stream.read(8192)
        if b"\0" in data:
            raise ValueError("Select a text file to edit")
        data.decode("utf-8")
        windows = sys.platform == "win32"
        value = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        value = value or ("notepad" if windows else "nano")
        command = shlex.split(value, posix=not windows)
        if windows:
            # Non-POSIX splitting keeps backslashes in paths but also keeps the quotes.
            command = [part.strip('"') for part in command]
        if not command:
            raise ValueError("EDITOR is empty")
        return command + [str(path)]

    def walk_files(self, on_error: Callable[[OSError], None] | None = None) -> Iterator[Path]:
        """Yield regular files without following links; report unreadable directories."""
        for directory, dirs, files in os.walk(self.root, followlinks=False, onerror=on_error):
            dirs[:] = [
                name
                for name in dirs
                if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink()
            ]
            for name in files:
                path = Path(directory) / name
                try:
                    self.safe(path)
                    if path.is_file():
                        yield path
                except ValueError:
                    continue
                except OSError as error:
                    if on_error:
                        on_error(error)


def _raise(error: OSError) -> None:
    raise error
