from pathlib import Path
from unittest.mock import patch

from semfs.duplicates import find_duplicates
from semfs.file_manager import FileManager


def test_exact_duplicates_and_wasted_storage(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"identical")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.bin").write_bytes(b"identical")
    (tmp_path / "c.txt").write_bytes(b"identical")
    (tmp_path / "unique.txt").write_text("different")
    scan = find_duplicates(FileManager(tmp_path))
    assert len(scan.groups) == 1
    assert len(scan.groups[0].paths) == 3
    assert scan.wasted == 18
    assert len(scan.groups[0].digest) == 64


def test_same_size_different_contents(tmp_path):
    (tmp_path / "a").write_text("abc")
    (tmp_path / "b").write_text("xyz")
    assert find_duplicates(FileManager(tmp_path)).groups == []


def test_hard_links_and_symlinks(tmp_path):
    file = tmp_path / "a"
    file.write_text("same")
    (tmp_path / "hard").hardlink_to(file)
    (tmp_path / "sym").symlink_to(file)
    assert find_duplicates(FileManager(tmp_path)).groups == []


def test_unreadable_files_are_reported(tmp_path):
    (tmp_path / "a").write_text("same")
    (tmp_path / "b").write_text("same")
    with patch.object(Path, "open", side_effect=PermissionError("denied")):
        scan = find_duplicates(FileManager(tmp_path))
    assert not scan.groups
    assert len(scan.errors) == 2


def test_duplicate_scan_can_be_cancelled(tmp_path):
    (tmp_path / "a").write_text("same")
    (tmp_path / "b").write_text("same")
    assert find_duplicates(FileManager(tmp_path), cancelled=lambda: True).groups == []
