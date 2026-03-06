from __future__ import annotations

from pathlib import Path


def main() -> None:
    app_path = Path("src/app.txt")
    current = app_path.read_text(encoding="utf-8")
    updated = current.replace("version=1", "version=2")
    app_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
