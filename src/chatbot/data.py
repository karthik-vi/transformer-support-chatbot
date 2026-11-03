"""Banking77 (Casanueva et al., 2020): 13,083 online-banking customer queries, 77 intents. CC BY 4.0.
Source: https://github.com/PolyAI-LDN/task-specific-datasets (banking_data/train.csv, test.csv). NOT Jocata data."""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "test.csv"):
        if not (dest / f).exists():
            with urllib.request.urlopen(BASE + f, timeout=120) as r:
                (dest / f).write_bytes(r.read())
    return dest


def load_split(path: Path) -> tuple[list[str], list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]
