from __future__ import annotations

from llm_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["report", "--results", "logs/results.jsonl"]))
