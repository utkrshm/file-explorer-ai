from textual.widgets import DataTable, Input

from semfs.app import Help, Prompt, SemFS


async def test_keyboard_browse_and_create(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    app = SemFS(tmp_path)
    async with app.run_test(size=(110, 35)) as pilot:
        await pilot.press("enter")
        assert app.fm.current == folder
        await pilot.press("backspace", "n")
        assert isinstance(app.screen, Prompt)
        await pilot.press(*"new.txt", "enter")
        await pilot.pause()
        assert (tmp_path / "new.txt").exists()
        await pilot.press("N")
        await pilot.press(*"created", "enter")
        await pilot.pause()
        assert (tmp_path / "created").is_dir()


async def test_keyboard_navigation_help_and_focus(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    app = SemFS(tmp_path)
    async with app.run_test(size=(100, 30)) as pilot:
        assert app.query_one(DataTable).cursor_row == 0
        await pilot.press("j")
        assert app.selected().name == "b.txt"
        await pilot.press("k")
        assert app.selected().name == "a.txt"
        await pilot.press("/")
        assert isinstance(app.focused, Input)
        await pilot.press(*"notes")
        assert app.query_one("#search", Input).value == "notes"
        await pilot.press("escape", "?")
        assert isinstance(app.screen, Help)
        await pilot.press("escape")
        assert not isinstance(app.screen, Prompt)
        await pilot.press("tab")
        assert app.focused is not app.query_one(DataTable)


async def test_keyboard_mutations_and_duplicate_delete(tmp_path):
    (tmp_path / "original.txt").write_text("same content")
    app = SemFS(tmp_path)
    async with app.run_test(size=(110, 35)) as pilot:
        await pilot.press("r")
        app.screen.query_one(Input).value = "renamed.txt"
        await pilot.press("enter")
        await pilot.pause()
        assert not (tmp_path / "original.txt").exists()
        assert (tmp_path / "renamed.txt").exists()

        await pilot.press("c")
        app.screen.query_one(Input).value = "copy.txt"
        await pilot.press("enter")
        await pilot.pause()
        assert (tmp_path / "copy.txt").read_text() == "same content"

        app.query_one(DataTable).move_cursor(row=app.paths.index(tmp_path / "copy.txt"))
        await pilot.press("m")
        app.screen.query_one(Input).value = "moved.txt"
        await pilot.press("enter")
        await pilot.pause()
        assert not (tmp_path / "copy.txt").exists()
        assert (tmp_path / "moved.txt").exists()

        await pilot.press("d")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.mode == "duplicates"
        assert len(app.paths) == 2
        chosen = app.selected()
        await pilot.press("delete")
        assert isinstance(app.screen, Prompt)
        await pilot.press("enter")  # Cancel has initial focus.
        assert chosen.exists()
        await pilot.press("delete")
        await pilot.click("#accept")
        await pilot.pause()
        assert not chosen.exists()


async def test_keyboard_search_index_and_similar(tmp_path):
    from test_search import FakeEmbedder

    (tmp_path / "os.txt").write_text("deadlock prevention mutex")
    (tmp_path / "locks.md").write_text("mutex and semaphore")
    (tmp_path / "garden.txt").write_text("tomato garden")
    app = SemFS(tmp_path)
    app.embedder = FakeEmbedder()
    async with app.run_test(size=(110, 35)) as pilot:
        await pilot.press("/", *".txt", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.paths) == 2
        assert app.mode == "filename"

        await pilot.press("escape", "i")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.database.metadata()) == 3

        await pilot.press("s")
        app.query_one("#search", Input).value = "deadlock"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.mode == "semantic"
        assert app.selected().name == "os.txt"

        await pilot.press("f")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.mode == "similar"
        assert app.selected().name == "locks.md"


async def test_background_scan_allows_navigation_and_cancel(tmp_path, monkeypatch):
    import time

    from semfs.indexer import IndexReport

    folder = tmp_path / "folder"
    folder.mkdir()

    def slow_index(manager, database, embedder, progress, cancelled):
        while not cancelled():
            time.sleep(0.01)
        return IndexReport(cancelled=True)

    monkeypatch.setattr("semfs.app.index_files", slow_index)
    app = SemFS(tmp_path)
    async with app.run_test(size=(90, 28)) as pilot:
        await pilot.press("i")
        await pilot.pause()
        assert app.busy
        await pilot.press("enter")
        assert app.fm.current == folder
        await pilot.press("ctrl+x")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.busy


async def test_arrow_navigation_and_black_theme(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from textual.widgets import Footer, Header, Static

    folder = tmp_path / "folder"
    folder.mkdir()
    file = folder / "note.txt"
    file.write_text("preview")
    app = SemFS(tmp_path)
    opened = Mock()
    monkeypatch.setattr(app.fm, "open_default", opened)
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("right")
        assert app.fm.current == folder
        await pilot.press("right")
        opened.assert_called_once_with(file)

        await pilot.press("/", "p", "left", "right")
        assert app.fm.current == folder
        assert app.query_one("#search", Input).value == "p"
        opened.assert_called_once()
        await pilot.press("escape", "left")
        assert app.fm.current == tmp_path
        await pilot.press("left", "p")
        assert app.fm.current == tmp_path
        await pilot.press("right", "p")
        assert app.fm.current == tmp_path

        navigation = app.query_one("#navigation", Static)
        footer = app.query_one(Footer)
        assert navigation.region.y >= app.query_one(DataTable).region.bottom
        assert navigation.region.bottom <= footer.region.y
        assert "Parent" in str(navigation.content)
        assert "Open" in str(navigation.content)
        assert app.screen.styles.background.hex == "#000000"
        assert app.query_one(DataTable).styles.background.hex == "#000000"
        assert app.query_one(Header).styles.background.hex == "#000000"
        assert footer.styles.background.hex == "#000000"
        cursor = app.query_one(DataTable).get_component_styles("datatable--cursor")
        assert cursor.background.hex == "#1D4ED8"
