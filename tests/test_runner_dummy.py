from __future__ import annotations

import json
from pathlib import Path

from llm_harness.eval import runner as runner_module
from llm_harness.eval.runner import run_eval
from llm_harness.providers.base import ProviderResult


def test_run_eval_dummy_writes_two_results(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_path = tmp_path / "logs" / "results.jsonl"
    events_path = tmp_path / "logs" / "events.jsonl"

    rows = [
        {
            "id": "1",
            "input_text": "Router keeps dropping every 10 minutes.",
            "expected": {
                "category": "Connectivity",
                "priority": "P2",
                "device": "Router",
                "needs_clarification": False,
                "missing_fields": [],
                "summary": "Router drops connection intermittently.",
            },
        },
        {
            "id": "2",
            "input_text": "Laptop battery drains quickly.",
            "expected": {
                "category": "Hardware",
                "priority": "P3",
                "device": "Laptop",
                "needs_clarification": False,
                "missing_fields": [],
                "summary": "Battery life is shorter than expected.",
            },
        },
    ]

    with dataset_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    run_summary = run_eval(
        dataset_path=str(dataset_path),
        provider="dummy",
        model="dummy",
        out_results_path=str(results_path),
        out_events_path=str(events_path),
    )

    assert run_summary["total_cases"] == 2
    assert results_path.exists()

    result_lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(result_lines) == 2

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(event_lines) == 2


def test_run_eval_builds_allowed_missing_fields_from_dataset(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_path = tmp_path / "logs" / "results.jsonl"
    events_path = tmp_path / "logs" / "events.jsonl"

    row = {
        "id": "emp-1",
        "input_text": "Can not access payroll portal.",
        "expected": {
            "category": "Access",
            "priority": "P2",
            "device": "Unknown",
            "needs_clarification": True,
            "missing_fields": ["employee_email", "employee_id"],
            "summary": "Need employee identity details.",
        },
    }

    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class StaticProvider:
        def generate(self, item: dict, model: str) -> ProviderResult:
            del model
            expected = item["expected"]
            return ProviderResult(actual=expected, latency_ms=7.0, status="ok")

    monkeypatch.setattr(runner_module, "get_provider", lambda *args, **kwargs: StaticProvider())

    run_eval(
        dataset_path=str(dataset_path),
        provider="dummy",
        model="dummy",
        out_results_path=str(results_path),
        out_events_path=str(events_path),
    )

    lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["unknown_missing_fields"] == []
    assert result["warnings"] == []
    assert result["overall_pass"] is True


def test_run_eval_ollama_errors_do_not_abort_run(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_path = tmp_path / "logs" / "results.jsonl"
    events_path = tmp_path / "logs" / "events.jsonl"

    rows = [
        {
            "id": "err-1",
            "input_text": "VPN issue",
            "expected": {
                "category": "VPN",
                "priority": "P2",
                "device": "Windows",
                "needs_clarification": False,
                "missing_fields": [],
                "summary": "VPN issue.",
            },
        },
        {
            "id": "err-2",
            "input_text": "Email issue",
            "expected": {
                "category": "Email",
                "priority": "P3",
                "device": "Mac",
                "needs_clarification": False,
                "missing_fields": [],
                "summary": "Email issue.",
            },
        },
    ]

    with dataset_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    class FailingProvider:
        def generate(self, item: dict, model: str) -> ProviderResult:
            del item, model
            raise RuntimeError("ollama boom")

    monkeypatch.setattr(runner_module, "get_provider", lambda *args, **kwargs: FailingProvider())

    run_summary = run_eval(
        dataset_path=str(dataset_path),
        provider="ollama",
        model="qwen2.5:4b",
        out_results_path=str(results_path),
        out_events_path=str(events_path),
        show_progress=False,
    )

    assert run_summary["total_cases"] == 2
    result_lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(result_lines) == 2
    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(event_lines) == 2
