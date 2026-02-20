from __future__ import annotations

import json
from pathlib import Path


def append_jsonl(path: str, event: dict) -> None:
    """Append a single event as one JSON line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
