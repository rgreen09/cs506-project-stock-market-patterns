from pathlib import Path
from typing import List


class MLLogger:
    """
    Minimal logger that accumulates lines and can write to disk.
    """

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.lines: List[str] = []

    def log(self, message: str = "") -> None:
        self.lines.append(message)
        print(message)

    def section(self, title: str) -> None:
        self.log("")
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)

    def save(self) -> None:
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\nLog saved to {path}")

