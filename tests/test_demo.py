import pytest

from semfs.demo import generate
from semfs.duplicates import find_duplicates
from semfs.file_manager import FileManager


def test_demo_dataset(tmp_path):
    root = generate(tmp_path / "demo")
    assert (root / "notes/os_handbook.pdf").is_file()
    assert (root / "images/scene_01.png").is_file()
    assert (root / "code/graphs/dijkstra.py").is_file()
    groups = find_duplicates(FileManager(root)).groups
    assert sorted(len(group.paths) for group in groups) == [2, 3]
    with pytest.raises(FileExistsError):
        generate(root)
