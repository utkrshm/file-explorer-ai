import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import symlink

from semfs.file_manager import FileManager


@pytest.fixture
def fm(tmp_path):
    return FileManager(tmp_path)


def test_create_navigate_and_parent(fm):
    file = fm.create("note.txt")
    assert file.read_text() == ""
    folder = fm.create("notes", directory=True)
    assert fm.navigate(folder) == folder
    assert fm.create("nested.md").parent == folder
    assert fm.parent() == fm.root
    assert fm.parent() == fm.root
    with pytest.raises(FileExistsError):
        fm.create("note.txt")


def test_rename_copy_move_delete(fm):
    original = fm.create("one.txt")
    original.write_text("important content")
    renamed = fm.rename(original, "two.txt")
    assert not original.exists()
    copied = fm.transfer(renamed, "copy.txt")
    assert copied.read_text() == "important content"
    target = fm.create("target", directory=True)
    moved = fm.transfer(copied, target, move=True)
    assert moved == target / "copy.txt"
    assert not copied.exists()
    fm.delete(renamed)
    fm.delete(target)
    assert not renamed.exists()
    assert not target.exists()


def test_copy_and_move_directories(fm):
    folder = fm.create("folder", directory=True)
    (folder / "note.txt").write_text("hello")
    copied = fm.transfer(folder, "other")
    assert (copied / "note.txt").read_text() == "hello"
    moved = fm.transfer(copied, "moved", move=True)
    assert moved.is_dir()
    with pytest.raises(ValueError, match="inside itself"):
        fm.transfer(folder, folder / "nested")
    with pytest.raises(FileExistsError):
        fm.transfer(folder, fm.root)


@pytest.mark.parametrize("name", ["../escape.txt", "/tmp/escape.txt", "../../outside"])
def test_root_containment(fm, name):
    with pytest.raises(ValueError):
        fm.create(name)
    with pytest.raises(ValueError):
        fm.navigate(name)


def test_root_cannot_be_modified(fm):
    for operation in (
        lambda: fm.delete(fm.root),
        lambda: fm.rename(fm.root, "oops"),
        lambda: fm.transfer(fm.root, fm.root.parent / "oops"),
    ):
        with pytest.raises(ValueError):
            operation()


def test_symlink_containment(fm, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    link = fm.root / "link"
    symlink(link, outside, target_is_directory=True)
    dangling = fm.root / "broken"
    symlink(dangling, outside / "missing")
    for path in (link, link / "secret.txt", dangling):
        with pytest.raises(ValueError):
            fm.safe(path)
    assert list(fm.walk_files()) == []
    with pytest.raises(ValueError):
        fm.create("broken")
    with pytest.raises(ValueError):
        fm.navigate(link)


def test_nested_symlink_copy_blocked(fm, tmp_path):
    folder = fm.create("folder", directory=True)
    outside = tmp_path.parent / f"{tmp_path.name}-target.txt"
    outside.write_text("secret")
    symlink(folder / "link", outside)
    with pytest.raises(ValueError):
        fm.transfer(folder, "copy")
    assert not (fm.root / "copy").exists()


def test_preview_sort_and_editor(fm, monkeypatch):
    small = fm.create("a.txt")
    small.write_text("one")
    large = fm.create("z.md")
    large.write_text("longer document")
    assert fm.entries("size") == [small, large]
    assert fm.preview(small) == "one"
    assert "truncated" in fm.preview(large, limit=4)
    monkeypatch.setenv("EDITOR", "vim -u NONE")
    monkeypatch.delenv("VISUAL", raising=False)
    assert fm.editor_command(small) == ["vim", "-u", "NONE", str(small)]
    if sys.platform == "win32":
        with patch("semfs.file_manager.os.startfile") as startfile:
            fm.open_default(small)
            startfile.assert_called_once_with(small)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        with patch("semfs.file_manager.subprocess.Popen") as popen:
            fm.open_default(small)
            assert popen.call_args.args[0] == [opener, str(small)]


def test_windows_editor_default_and_paths(fm, monkeypatch):
    file = fm.create("a.txt")
    monkeypatch.setattr("semfs.file_manager.sys.platform", "win32")
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert fm.editor_command(file) == ["notepad", str(file)]
    monkeypatch.setenv("EDITOR", r'"C:\Program Files\Editor\edit.exe" --wait')
    assert fm.editor_command(file) == [r"C:\Program Files\Editor\edit.exe", "--wait", str(file)]


def test_inaccessible_file_reports_error(fm):
    file = fm.create("locked.txt")
    with patch.object(Path, "open", side_effect=PermissionError("permission denied")):
        with pytest.raises(PermissionError):
            fm.preview(file)


def test_reject_binary_editor_and_rename_paths(fm):
    binary = fm.create("fake.txt")
    binary.write_bytes(b"\0binary")
    with pytest.raises(ValueError):
        fm.editor_command(binary)
    with pytest.raises(ValueError):
        fm.rename(binary, "../outside.txt")
