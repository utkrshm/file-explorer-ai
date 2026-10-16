# SemFS

A compact Python terminal file manager for an Operating Systems mini-project.
Browse real files, preview text, manage folders, find exact duplicates, and search
document or image content with local **google/embeddinggemma-2** embeddings.

Python 3.12+, Textual, Sentence Transformers, NumPy, SQLite, and PyMuPDF. No model
server, cloud inference, captions, or vector database service.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from this project directory:

~~~sh
uv sync --locked
uv run semfs ~/Documents
~~~

The folder argument defaults to the current directory. The alternative module
entry point also works: **uv run python -m semfs ~/Documents**.

Linux is the primary target. Install your distribution's **xdg-utils** package
for default-application opening, and an editor such as nano, vim, or micro:

~~~sh
export EDITOR="nano"
~~~

VISUAL takes precedence over EDITOR. No editor setting defaults to nano.

Ordinary file management needs no model download. The first index or AI query
that requires inference downloads the public model checkpoint from Hugging Face.
This needs internet access and sufficient disk space. Once cached, inference
works locally and offline. No API key or login is part of the application.

The default dependency setup uses CPU PyTorch wheels to avoid installing CUDA
libraries on university lab machines. For an NVIDIA GPU, install an appropriate
CUDA build into the same environment:

~~~sh
uv pip install --reinstall torch torchvision --torch-backend=auto
uv run --no-sync semfs ~/Documents
~~~

Use **--no-sync** for subsequent CUDA launches to preserve that wheel selection.
Running uv sync restores the default CPU environment. See the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for
driver-specific wheel selection.

The application selects CUDA when PyTorch reports it available. It falls back
to CPU when CUDA is unavailable or GPU memory is insufficient during initial
loading. Force CPU inference with:

~~~sh
SEMFS_DEVICE=cpu uv run semfs ~/Documents
~~~

## Interface and shortcuts

The left panel lists directory entries or search results. The right panel shows
metadata, a text preview, and the matching passage or duplicate group.

| Key | Action |
| --- | --- |
| Up / Down or j / k | Select an entry |
| Enter | Enter a directory or open a file |
| Backspace | Parent directory, stopping at the selected root |
| n / N | Create a file / directory |
| r | Rename |
| c / m | Copy / move |
| Delete | Permanently delete after confirmation |
| o | Open with xdg-open |
| e | Edit using VISUAL, EDITOR, or nano |
| / | Filename search; type a name, extension, or glob and press Enter |
| s | AI search; describe content and press Enter |
| i | Incrementally index the selected root |
| d | Find exact duplicates |
| f | Find files similar to the selected indexed file |
| t | Cycle ascending name, size, and modification-date sorting |
| Tab / Shift+Tab | Move between panels |
| Ctrl+X | Cancel a background task between files or model batches |
| Escape | Browse or close a dialog |
| ? | Help |
| q | Quit |

Shortcuts do not intercept letters while typing. Escape returns to the file
list. Navigation and file operations remain usable during a scan. One scan or
search runs at a time. A current model call or download must finish before
cancellation can take effect.

Dialog paths are relative to the current directory. An existing destination
directory receives the original filename; a new destination names the new copy.
Absolute paths must also be inside the root. Existing entries are never
overwritten.

Symlinks are visible with @ but blocked from operations and recursive scans.
Copying or moving a directory containing links is blocked too. Deleting a
directory removes its contents, including nested links themselves, without
following their targets.

## Search and indexing

Filename search is recursive and case-insensitive, matching substrings or glob
patterns such as *.py. It needs no index. Unsupported formats remain available
for ordinary file management and filename search.

Semantic formats:

- Text: .txt, .md.
- Code: .py, .c, .cpp, .h, .rs, .java, .js.
- PDFs: .pdf, with page numbers in results.
- Images: .png, .jpg, .jpeg, through the model's native vision encoder.

The indexer extracts PDF text page by page, splitting text into overlapping
1,600-character chunks with 200-character overlap. Scanned PDFs without
extractable text need OCR first; OCR is outside this project. Password-protected
and unreadable files are reported and skipped.

Queries use the official SearchQuery prompt. Document chunks include a filename
title and content. Images have no text-task prefix. The model loads lazily and
is reused, with audio disabled. Inference uses float32 or supported CUDA
bfloat16, never float16, following the
[official model card](https://huggingface.co/google/embeddinggemma-2).

NumPy computes cosine similarity against cached chunks. Each file appears once
with its best chunk and passage. Scores range from -1 to 1 and are similarities,
not confidence percentages. Weakly related files can appear among the best
available matches. Similar-file search compares the selected file's mean
document vector against other files and excludes the source.

SQLite stores relative paths, size, nanosecond timestamps, file type, model ID,
passage locations, and float32 embeddings. Scans reuse unchanged files and
update modified ones. Deleted entries are removed on the next scan. Search
excludes changed or missing files immediately, before reindexing.

The database stays **outside the managed root**, normally under
$XDG_CACHE_HOME/semfs or ~/.cache/semfs. When that is inside the selected root
(for example, when managing your home directory), it uses a per-user directory
under the system temporary directory. That fallback survives app restarts but
can be cleaned by the OS. Set SEMFS_CACHE_DIR to choose a cache directory
outside the managed root. A filesystem-wide / root cannot have an external
cache; choose a documents or project folder.

Recursive scans skip symlinks and .git, .venv, __pycache__, .cache,
.pytest_cache, and .ruff_cache. These folders remain browsable normally.

## Duplicates and similar files

**d** groups files by size, then streams SHA-256 hashes to verify identical
contents. Groups show every copy, a hash, and potential wasted storage if one
copy is kept. Hard links already share storage and are counted once. Select an
unwanted copy and press Delete. Confirmation defaults to Cancel.

**f** finds semantic relatedness. Related files may contain different bytes.
Neither feature automatically deletes anything.

## Demonstration

Create a fresh dataset:

~~~sh
uv run semfs-demo demo-files
uv run semfs demo-files
~~~

The generator refuses to overwrite an existing directory. It creates nested
folders, OS notes, source code, a two-page PDF, two car images, exact copies,
and a pair of related synchronization documents.

1. **Browse:** select notes, press Enter, then Backspace.
2. **Create and rename:** press n, type scratch.txt, and save. Select it, press r,
   replace the name with lab_notes.txt, and save. N creates a folder.
3. **Open:** select the PDF and press o. Select text and press e to edit.
   Text also previews in the right panel.
4. **Index:** press i and wait for the updated/reused summary. The first run
   downloads and loads the model.
5. **AI search:** press s, enter "notes about deadlocks and semaphores", and
   press Enter. Select a result to inspect the passage or PDF page.
6. **Code and images:** try "Python implementation of shortest paths in a graph"
   and "a red car". Image filenames do not contain their colors.
7. **Exact duplicates:** Escape, then d. Inspect the OS notes and Dijkstra groups.
8. **Similar files:** Escape, browse to notes/synchronization.md, and press f.
   Inspect concurrency_summary.txt among the related content.
9. **Delete a copy:** press d, select an unwanted file under copies/, press
   Delete, and confirm with the Delete button. Press d again to rescan.
10. **Reuse:** q to quit, relaunch the same root, and press i. Unchanged files are
    reused. AI search can use the saved SQLite index directly after restart.

Use generated files for the classroom deletion demonstration.

## Development

~~~sh
uv sync --locked
uv run pytest -m "not integration"
uv run ruff check .
uv run ruff format --check .
~~~

Ordinary tests mock inference and use temporary files and databases. Coverage
includes filesystem operations, root containment, filename search, exact hashes,
persistence, index updates, semantic ranking, extraction, and keyboard flows.

The opt-in integration test downloads and runs the **real EmbeddingGemma 2**
model on text and images:

~~~sh
SEMFS_REAL_MODEL=1 uv run pytest -m integration -v
~~~

To check a read-only sample of an existing project collection, provide a query
and its expected top result. The sample includes that file, up to ten project
READMEs, four source files, one PDF, and two images:

~~~sh
SEMFS_REAL_MODEL=1 \
SEMFS_TEST_ROOT=~/active_projects \
SEMFS_TEST_QUERY="implementing denoising diffusion probabilistic models with noise schedules from scratch" \
SEMFS_EXPECTED_PATH="diffusion-from-scratch/README.md" \
uv run pytest tests/test_integration.py::test_real_project_sample -v -s
~~~

Use the standard HF_HUB_CACHE environment variable for a custom model cache.
For CUDA testing after replacing the wheel, use **uv run --no-sync pytest**.
Scores and run times depend on hardware and content; no fixed performance
claim is made.

## Code layout

| Module | Responsibility |
| --- | --- |
| app.py | Textual widgets, dialogs, bindings, workers |
| file_manager.py | Root-confined filesystem operations |
| duplicates.py | Size grouping and SHA-256 |
| database.py | SQLite cache and freshness checks |
| embeddings.py | Lazy model loading and embedding conventions |
| indexer.py | Extraction, chunking, incremental indexing |
| search.py | Filename matches, cosine ranking, similar files |
| demo.py | Reproducible demo dataset |

This is a small local OS project. It handles ordinary filesystem errors and
avoids following symlinks; it is not a security sandbox against another process
maliciously replacing paths during an operation.
