"""The Textual interface. Slow scans and inference run in workers."""

import subprocess
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, ProgressBar, Static
from textual.worker import get_current_worker

from semfs.database import Database
from semfs.duplicates import find_duplicates
from semfs.embeddings import Embedder
from semfs.file_manager import FileManager, human_size
from semfs.indexer import index_files
from semfs.search import filename_search, semantic_search, similar_files


class Prompt(ModalScreen[str | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "", *, confirm: bool = False):
        super().__init__()
        self.prompt_title, self.value, self.confirm = title, value, confirm

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.prompt_title, markup=False)
            if not self.confirm:
                yield Input(self.value, id="answer")
            with Horizontal(classes="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(
                    "Delete" if self.confirm else "Save",
                    variant="error" if self.confirm else "primary",
                    id="accept",
                )

    def on_mount(self) -> None:
        self.query_one("#cancel" if self.confirm else "#answer").focus()

    @on(Input.Submitted)
    def submitted(self) -> None:
        self.accept()

    @on(Button.Pressed, "#accept")
    def accept(self) -> None:
        self.dismiss("yes" if self.confirm else self.query_one(Input).value)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class FileTable(DataTable):
    BINDINGS = [Binding("j", "cursor_down", show=False), Binding("k", "cursor_up", show=False)]


class SemFS(App):
    TITLE = "SemFS"
    SUB_TITLE = "Your files, a thought away"
    CSS = """
    Screen { background: #0d1520; }
    Header { background: #172838; color: #63dfc5; }
    #location { height: 2; padding: 0 2; color: #a7bbcf; }
    #search { margin: 0 1 1 1; border: round #34516a; }
    #panels { height: 1fr; }
    #files { width: 3fr; border: round #34516a; margin: 0 0 0 1; }
    #detail-scroll { width: 2fr; border: round #34516a; margin: 0 1; padding: 0 1; }
    #details { height: auto; }
    #status { height: 2; padding: 0 2; color: #63dfc5; }
    #activity { height: 1; padding: 0 2; color: #a7bbcf; }
    #progress { height: 1; margin: 0 2; }
    Footer { background: #172838; }
    Help { align: center middle; background: #000000 60%; }
    Prompt { align: center middle; background: #000000 60%; }
    #dialog { width: 64; max-width: 95%; height: auto; max-height: 90%; padding: 1 2;
              border: round #63dfc5; background: #172838; }
    #dialog Static { height: auto; margin-bottom: 1; }
    .buttons { height: 3; align-horizontal: right; margin-top: 1; }
    .buttons Button { margin-left: 1; }
    """
    BINDINGS = [
        Binding("backspace", "parent", "Parent", show=False),
        Binding("n", "new_file", "New", show=False),
        Binding("N", "new_directory", "Folder", show=False),
        Binding("r", "rename", "Rename", show=False),
        Binding("c", "copy", "Copy", show=False),
        Binding("m", "move", "Move", show=False),
        Binding("delete", "delete", "Delete", show=False),
        Binding("o", "open", "Open", show=False),
        Binding("e", "edit", "Edit", show=False),
        Binding("slash", "filename", "Name search"),
        Binding("s", "semantic", "AI search"),
        Binding("i", "index", "Index"),
        Binding("d", "duplicates", "Duplicates"),
        Binding("f", "similar", "Similar", show=False),
        Binding("t", "sort", "Sort", show=False),
        Binding("escape", "browse", "Browse", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("ctrl+x", "cancel_task", "Cancel scan", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, root: Path):
        super().__init__()
        self.fm = FileManager(root)
        self.paths: list[Path] = []
        self.sort_by = "name"
        self.mode = "browse"
        self.search_mode = "filename"
        self.annotations: dict[Path, str] = {}
        self.embedder = Embedder()
        self._database: Database | None = None
        self.busy = False
        self.stop_event = threading.Event()
        self.view_generation = 0

    @property
    def database(self) -> Database:
        if self._database is None:
            self._database = Database(self.fm)
        return self._database

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="location", markup=False)
        yield Input(placeholder="/ filename search  |  s AI search  |  Escape browse", id="search")
        with Horizontal(id="panels"):
            yield FileTable(id="files", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="detail-scroll", can_focus=True):
                yield Static("Select a file to preview it.", id="details", markup=False)
        yield Static("Ready. Press ? for all shortcuts.", id="status", markup=False)
        yield Static(
            "Index: press i to scan supported files. Model loads on demand.",
            id="activity",
            markup=False,
        )
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#files").border_title = "FILES"
        self.query_one("#detail-scroll").border_title = "DETAILS & PREVIEW"
        self.query_one("#progress").display = False
        self.attempt(lambda: None)
        self.query_one("#files").focus()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if isinstance(self.screen, ModalScreen):
            return False
        if isinstance(self.focused, Input) and action != "browse":
            return False
        return True

    def status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def attempt(self, operation: Callable) -> None:
        try:
            operation()
            self.refresh_files()
        except (OSError, ValueError, UnicodeError) as error:
            self.notify(str(error), title="Operation failed", severity="error", timeout=8)
            self.status(f"Failed: {error}")

    def refresh_files(self) -> None:
        self.view_generation += 1
        self.mode = "browse"
        self.show_paths(self.fm.entries(self.sort_by))
        self.query_one("#location", Static).update(f"ROOT {self.fm.root}   /   {self.fm.current}")
        self.status(f"{len(self.paths)} entries  |  Sort: {self.sort_by}  |  ? shortcuts")

    def show_paths(self, paths: list[Path]) -> None:
        table = self.query_one("#files", DataTable)
        table.clear(columns=True)
        table.add_columns("Name / path", "Size", "Modified")
        self.paths = paths
        self.annotations = {}
        table.border_title = "FILES"
        for index, path in enumerate(paths):
            try:
                stat = path.lstat()
                base = self.fm.current if self.mode == "browse" else self.fm.root
                label = str(path.relative_to(base))
                prefix = "@ " if path.is_symlink() else "▸ " if path.is_dir() else "  "
                table.add_row(
                    Text(prefix + label),
                    human_size(stat.st_size),
                    datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    key=str(index),
                )
            except OSError:
                table.add_row(Text(path.name), "?", "Unavailable", key=str(index))
        self.show_details()

    def selected(self) -> Path | None:
        row = self.query_one("#files", DataTable).cursor_row
        return self.paths[row] if 0 <= row < len(self.paths) else None

    @on(DataTable.RowHighlighted, "#files")
    def show_details(self) -> None:
        path = self.selected()
        text = "No files to display."
        if path is not None:
            try:
                stat = path.lstat()
                kind = (
                    "Symlink (blocked)"
                    if path.is_symlink()
                    else "Folder"
                    if path.is_dir()
                    else (path.suffix or "File")
                )
                text = (
                    f"{path.name}\n\n{path.relative_to(self.fm.root)}\n\n"
                    f"Type      {kind}\nSize      {human_size(stat.st_size)}\n"
                    f"Modified  {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M:%S}"
                )
                if not path.is_symlink() and path.is_file():
                    text += "\n\n" + self.fm.preview(path)
                if path in self.annotations:
                    text = self.annotations[path] + "\n\n" + text
            except (OSError, ValueError) as error:
                text = f"Cannot preview: {error}"
        self.query_one("#details", Static).update(text)

    @on(DataTable.RowSelected, "#files")
    def enter(self) -> None:
        path = self.selected()
        if path:
            if path.is_dir():
                self.attempt(lambda: self.fm.navigate(path))
            else:
                self.action_open()

    def action_parent(self) -> None:
        self.attempt(self.fm.parent)

    def action_browse(self) -> None:
        self.query_one("#search", Input).value = ""
        self.attempt(lambda: None)
        self.query_one("#files").focus()

    def prompt_operation(self, title: str, operation: Callable[[str], object], value: str = ""):
        def done(answer: str | None) -> None:
            if answer is not None:
                self.attempt(lambda: operation(answer))

        self.push_screen(Prompt(title, value), done)

    def action_new_file(self) -> None:
        self.prompt_operation("New file (relative to current folder)", self.fm.create)

    def action_new_directory(self) -> None:
        self.prompt_operation("New directory", lambda name: self.fm.create(name, directory=True))

    def action_rename(self) -> None:
        if path := self.selected():
            self.prompt_operation("Rename", lambda name: self.fm.rename(path, name), path.name)

    def transfer(self, move: bool) -> None:
        if path := self.selected():
            self.prompt_operation(
                "Move to (current folder relative path)"
                if move
                else "Copy to (current folder relative path)",
                lambda target: self.fm.transfer(path, target, move=move),
            )

    def action_copy(self) -> None:
        self.transfer(False)

    def action_move(self) -> None:
        self.transfer(True)

    def action_delete(self) -> None:
        if path := self.selected():

            def done(answer: str | None) -> None:
                if answer == "yes":
                    self.attempt(lambda: self.fm.delete(path))

            self.push_screen(
                Prompt(
                    f"Permanently delete {path.name}?\n"
                    "Directories and their contents will be removed.",
                    confirm=True,
                ),
                done,
            )

    def action_open(self) -> None:
        if path := self.selected():
            try:
                self.fm.open_default(path)
                self.status(f"Opening {path.name} with xdg-open")
            except (OSError, ValueError) as error:
                self.notify(str(error), severity="error")

    def action_edit(self) -> None:
        if path := self.selected():

            def edit() -> None:
                command = self.fm.editor_command(path)
                with self.suspend():
                    result = subprocess.run(command, check=False)
                if result.returncode:
                    raise ValueError(f"Editor exited with code {result.returncode}")

            self.attempt(edit)

    def action_sort(self) -> None:
        sorts = ["name", "size", "modified"]
        self.sort_by = sorts[(sorts.index(self.sort_by) + 1) % len(sorts)]
        self.attempt(lambda: None)

    def focus_search(self, mode: str) -> None:
        self.search_mode = mode
        search = self.query_one("#search", Input)
        search.value = ""
        search.placeholder = (
            "AI: describe the file, then Enter"
            if mode == "semantic"
            else "Filename or extension, then Enter (recursive)"
        )
        search.focus()

    def action_filename(self) -> None:
        self.focus_search("filename")

    def action_semantic(self) -> None:
        self.focus_search("semantic")

    @on(Input.Submitted, "#search")
    def search_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        self.query_one("#files").focus()
        mode = self.search_mode
        if mode == "filename":

            def find():
                return filename_search(self.fm, query)
        else:

            def find():
                return semantic_search(self.database, self.embedder, query)

        self.start_task(
            f"{'AI' if mode == 'semantic' else 'Filename'} search: {query}",
            find,
            lambda results: self.show_search_results(results, mode, query),
            changes_view=True,
        )

    def start_task(
        self, label: str, job: Callable, finished: Callable, *, changes_view: bool = False
    ) -> None:
        if self.busy:
            self.notify("A task is running. Ctrl+X cancels between files.", severity="warning")
            return
        self.busy = True
        self.stop_event.clear()
        if changes_view:
            self.view_generation += 1
        generation = self.view_generation if changes_view else None
        self.query_one("#activity", Static).update(label)
        progress = self.query_one("#progress", ProgressBar)
        progress.display = True
        progress.update(total=None)
        self.background_job(job, finished, generation)

    @work(thread=True, exit_on_error=False)
    def background_job(self, job: Callable, finished: Callable, generation: int | None) -> None:
        worker = get_current_worker()
        result, error = None, None
        try:
            result = job()
        except Exception as caught:
            error = caught
        if not worker.is_cancelled:
            self.call_from_thread(self.finish_task, result, error, finished, generation)

    def finish_task(
        self, result, error: Exception | None, finished: Callable, generation: int | None
    ) -> None:
        self.busy = False
        self.query_one("#progress").display = False
        if error:
            self.query_one("#activity", Static).update(f"Task failed: {error}")
            self.notify(str(error), title="Task failed", severity="error", timeout=12)
        elif self.stop_event.is_set():
            self.query_one("#activity", Static).update(
                "Task cancelled. Saved files remain indexed."
            )
        elif generation is None or generation == self.view_generation:
            try:
                finished(result)
            except (OSError, ValueError) as caught:
                self.notify(str(caught), severity="error")
        else:
            self.query_one("#activity", Static).update(
                "Task finished. View changed; results dismissed."
            )

    def progress_from_thread(self, done: int, total: int, message: str) -> None:
        self.call_from_thread(self.update_progress, done, total, message)

    def update_activity(self, message: str) -> None:
        self.query_one("#activity", Static).update(message)

    def update_progress(self, done: int, total: int, message: str) -> None:
        self.query_one("#activity", Static).update(message)
        self.query_one("#progress", ProgressBar).update(total=max(total, 1), progress=done)

    def action_cancel_task(self) -> None:
        if self.busy:
            self.stop_event.set()
            self.status("Cancellation requested. The current model call may need to finish.")

    def action_quit(self) -> None:
        self.stop_event.set()
        self.exit()

    def action_index(self) -> None:
        def finished(report) -> None:
            summary = (
                f"Index: {report.indexed} updated, {report.reused} reused, "
                f"{report.removed} removed, {len(report.errors)} errors"
            )
            self.query_one("#activity", Static).update(summary)
            self.status("Index saved outside the root. Press s to search.")
            self.show_errors(report.errors)

        self.start_task(
            "Scanning files. First inference downloads EmbeddingGemma 2.",
            lambda: index_files(
                self.fm,
                self.database,
                self.embedder,
                self.progress_from_thread,
                self.stop_event.is_set,
            ),
            finished,
        )

    def show_errors(self, errors: list[str]) -> None:
        if errors:
            self.notify(
                "\n".join(errors[:3]),
                title=f"{len(errors)} files could not be processed",
                severity="warning",
                timeout=15,
            )

    def show_search_results(self, results: list, mode: str, query: str) -> None:
        self.mode = mode
        if mode == "filename":
            self.show_paths(results)
        else:
            self.show_ranked(results)
        self.query_one("#files").border_title = (
            "AI SEARCH" if mode == "semantic" else "FILENAME SEARCH"
        )
        self.query_one("#activity", Static).update(f'{len(results)} results for "{query}"')
        self.status(
            "Enter/o opens a result. Escape returns to browsing."
            if results
            else "No matches. AI search needs an up-to-date index: press i."
        )

    def show_ranked(self, results: list) -> None:
        table = self.query_one("#files", DataTable)
        table.clear(columns=True)
        table.add_columns("File", "Cosine", "Match")
        self.paths = [item.path for item in results]
        self.annotations = {
            item.path: f"SIMILARITY {item.score:.3f}  |  {item.location}\n\n{item.passage}"
            for item in results
        }
        for index, item in enumerate(results):
            table.add_row(
                Text(str(item.path.relative_to(self.fm.root))),
                f"{item.score:.3f}",
                Text(item.location),
                key=str(index),
            )
        self.show_details()

    def action_duplicates(self) -> None:
        def progress(message: str) -> None:
            self.call_from_thread(self.update_activity, message)

        self.start_task(
            "Scanning exact duplicates by size and SHA-256...",
            lambda: find_duplicates(self.fm, progress, self.stop_event.is_set),
            self.show_duplicates,
            changes_view=True,
        )

    def show_duplicates(self, scan) -> None:
        self.mode = "duplicates"
        table = self.query_one("#files", DataTable)
        table.clear(columns=True)
        table.border_title = "EXACT DUPLICATES"
        table.add_columns("Group / path", "Size", "SHA-256")
        self.paths, self.annotations = [], {}
        for group_number, group in enumerate(scan.groups, 1):
            description = (
                f"EXACT DUPLICATE GROUP {group_number}\n"
                f"SHA-256: {group.digest}\n"
                f"Each copy: {human_size(group.size)}\n"
                f"Potential wasted storage: {human_size(group.wasted)}\n\n"
                + "\n".join(str(path.relative_to(self.fm.root)) for path in group.paths)
            )
            for path in group.paths:
                self.paths.append(path)
                self.annotations[path] = description
                table.add_row(
                    Text(f"{group_number}  {path.relative_to(self.fm.root)}"),
                    human_size(group.size),
                    group.digest[:12],
                    key=str(len(self.paths) - 1),
                )
        self.show_details()
        self.query_one("#activity", Static).update(
            f"{len(scan.groups)} exact duplicate groups | "
            f"Potential wasted storage: {human_size(scan.wasted)}"
        )
        self.status("Select an unwanted copy and press Delete. Deletion requires confirmation.")
        self.show_errors(scan.errors)

    def action_similar(self) -> None:
        if path := self.selected():

            def finished(results) -> None:
                self.mode = "similar"
                self.show_ranked(results)
                self.query_one("#files").border_title = "SEMANTICALLY SIMILAR FILES"
                self.query_one("#activity", Static).update(
                    f"{len(results)} files related to {path.name}. These are not exact duplicates."
                )
                self.status("Semantic relatedness only. Review files before deciding to delete.")

            self.start_task(
                f"Finding files semantically related to {path.name}...",
                lambda: similar_files(self.database, self.embedder, path),
                finished,
                changes_view=True,
            )

    def action_help(self) -> None:
        self.push_screen(Help())


class Help(ModalScreen):
    BINDINGS = [("escape", "dismiss", "Close"), ("question_mark", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(
                "SemFS shortcuts\n\n"
                "Up/Down or j/k   Navigate       Enter   Open directory/file\n"
                "Backspace       Parent         Tab     Switch panels\n"
                "n   New file    N   New folder  r       Rename\n"
                "c   Copy        m   Move        Delete  Confirm deletion\n"
                "o   Default app e   $VISUAL/$EDITOR (default: nano)\n"
                "/   Name search s   AI search   i       Index root\n"
                "d   SHA-256 duplicates          f       Similar files\n"
                "t   Sort name/size/date         Ctrl+X  Cancel scan\n"
                "Escape   Browse/cancel          q       Quit\n\n"
                "Paths in dialogs are relative to the current directory.\n"
                "Absolute paths inside the root are also accepted.\n"
                "Existing destinations are never overwritten.\n"
                "Symlinks are visible but blocked. Text previews are automatic.\n"
                "AI scores measure cosine similarity, not probabilities.\n\n"
                "Press Escape to close.",
                markup=False,
            )
