"""Command line entry point."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SemFS - local semantic file manager")
    parser.add_argument("folder", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        from semfs.app import SemFS

        app = SemFS(args.folder.expanduser())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    app.run()


if __name__ == "__main__":
    main()
