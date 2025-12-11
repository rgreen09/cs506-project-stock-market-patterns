import csv
import os
from pathlib import Path
from typing import Iterable, List, Sequence


def ensure_parent_dir(path: str) -> None:
    """
    Ensure the parent directory for a path exists.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_header(path: str, columns: Sequence[str]) -> None:
    """
    Create or overwrite a CSV file with the given header.
    """
    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def append_rows(path: str, columns: Sequence[str], rows: Iterable[dict]) -> int:
    """
    Append rows (dicts) to a CSV file, respecting the provided column order.
    Returns number of rows written.
    """
    ensure_parent_dir(path)
    count = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def resolve_output_path(base_dir: str, filename: str) -> str:
    """
    Join base directory and filename, ensuring the base exists.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(base_dir, filename)

