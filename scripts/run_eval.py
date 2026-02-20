from __future__ import annotations

from llm_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main([
            "run",
            "--dataset",
            "data/tickets_eval.jsonl",
            "--provider",
            "dummy",
            "--model",
            "dummy",
        ])
    )
