"""Helper to copy Phase 1 results into the paper directory.

Called by the training script at the end of each save_every step, and
by the post-eval script. Keeps `paper/tables/` and `paper/figures/`
in sync with `results/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def sync_results_to_paper(
    results_dir: Path,
    paper_dir: Path,
    prefix: str,
) -> None:
    """Copy result files matching `<prefix>_*` to paper/tables/ + paper/figures/.

    - `<prefix>_summary.json` → `paper/tables/<prefix>_summary.json`
    - `<prefix>_reliability.txt` → `paper/tables/<prefix>_reliability.txt`
    - `<prefix>_traces.jsonl` → `paper/data/<prefix>_traces.jsonl` (gitignored in results/, kept in paper)
    - `<prefix>_<timestamp>.log` → `paper/logs/<prefix>_<timestamp>.log`
    """
    results_dir = Path(results_dir)
    paper_dir = Path(paper_dir)
    tables = paper_dir / "tables"
    figures = paper_dir / "figures"
    paper_data = paper_dir / "data"
    paper_logs = paper_dir / "logs"
    for d in (tables, figures, paper_data, paper_logs):
        d.mkdir(parents=True, exist_ok=True)

    for src in results_dir.glob(f"{prefix}*"):
        if src.is_file():
            if src.suffix in (".txt", ".json", ".csv"):
                dst = tables / src.name
            elif src.suffix == ".jsonl":
                dst = paper_data / src.name
            elif src.suffix == ".log":
                dst = paper_logs / src.name
            else:
                continue
            shutil.copy2(src, dst)
            print(f"  synced: {src.name} → {dst.relative_to(paper_dir)}")
