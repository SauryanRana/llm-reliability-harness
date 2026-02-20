from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = {"id", "input_text", "expected"}


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL rows from ``path`` and validate required keys."""
    rows: list[dict] = []
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no}: expected a JSON object")

            missing = REQUIRED_KEYS.difference(item.keys())
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Line {line_no}: missing required keys: {missing_list}")

            rows.append(item)

    return rows
