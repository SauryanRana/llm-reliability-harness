from __future__ import annotations

from pathlib import Path

from llm_harness.eval.dataset import load_jsonl


def test_load_jsonl_reads_two_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        '{"id":"1","input_text":"hello","expected":"world"}\n'
        '{"id":"2","input_text":"foo","expected":"bar"}\n',
        encoding="utf-8",
    )

    rows = load_jsonl(str(dataset))

    assert len(rows) == 2
    assert rows[0]["id"] == "1"
    assert rows[1]["id"] == "2"
