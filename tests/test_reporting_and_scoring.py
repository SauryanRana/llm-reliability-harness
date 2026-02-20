from __future__ import annotations

import json
from pathlib import Path

from llm_harness.eval.scoring import score_case
from llm_harness.reporting.summarize import evaluate_gates, summarize_results, write_report_md


def test_score_case_flags_hallucination_and_extraction_failure() -> None:
    expected = {
        "category": "Hardware",
        "priority": "P2",
        "device": "Laptop",
        "needs_clarification": False,
        "missing_fields": [],
        "summary": "Battery issue",
    }
    actual = {
        "category": "Hardware",
        "priority": "P2",
        "device": "Unknown",
        "needs_clarification": True,
        "missing_fields": [],
        "summary": "Battery issue",
    }

    score = score_case(expected, actual, input_text="Windows laptop battery drains in one hour")

    assert score["hallucination"] is True
    assert "clarification_without_missing_fields" in score["failure_reasons"]
    assert score["extraction_failure_device_unknown"] is True
    assert "extraction_failure_device_unknown" in score["failure_reasons"]


def test_score_case_warning_only_keeps_overall_pass_true() -> None:
    expected = {
        "category": "Access",
        "priority": "P3",
        "device": "Laptop",
        "needs_clarification": True,
        "missing_fields": ["employee_email"],
        "summary": "Need email to continue.",
    }
    actual = {
        "category": "Access",
        "priority": "P3",
        "device": "Laptop",
        "needs_clarification": True,
        "missing_fields": ["invented_field_name"],
        "summary": "Need email to continue.",
    }

    score = score_case(expected, actual, input_text="Please help", allowed_missing_fields={"employee_email"})

    assert score["hallucination"] is False
    assert score["unknown_missing_fields"] == ["invented_field_name"]
    assert score["warnings"] == ["unknown_missing_fields"]
    assert "unknown_missing_fields" not in score["failure_reasons"]
    assert score["overall_pass"] is True


def test_report_contains_failures_section(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    events_path = tmp_path / "events.jsonl"
    report_path = tmp_path / "report.md"

    failing_row = {
        "id": "case-1",
        "overall_pass": False,
        "json_valid": True,
        "schema_valid": True,
        "category_correct": True,
        "priority_correct": False,
        "device_correct": True,
        "needs_clarification_correct": True,
        "hallucination": False,
        "failure_reasons": ["wrong_priority"],
        "expected": {
            "category": "Connectivity",
            "priority": "P2",
            "device": "Router",
            "needs_clarification": False,
            "missing_fields": [],
            "summary": "Router issue",
        },
        "actual": {
            "category": "Connectivity",
            "priority": "P3",
            "device": "Router",
            "needs_clarification": False,
            "missing_fields": [],
            "summary": "Router issue",
        },
        "latency_ms": 12.0,
    }
    passing_row = {
        "id": "case-2",
        "overall_pass": True,
        "json_valid": True,
        "schema_valid": True,
        "category_correct": True,
        "priority_correct": True,
        "device_correct": True,
        "needs_clarification_correct": True,
        "hallucination": False,
        "unknown_missing_fields": [],
        "warnings": [],
        "expected": {
            "category": "Hardware",
            "priority": "P3",
            "device": "Laptop",
            "needs_clarification": False,
            "missing_fields": [],
            "summary": "Battery issue",
        },
        "actual": {
            "category": "Hardware",
            "priority": "P3",
            "device": "Laptop",
            "needs_clarification": False,
            "missing_fields": [],
            "summary": "Battery issue",
        },
        "latency_ms": 10.0,
    }
    warning_only_row = {
        "id": "case-3",
        "overall_pass": True,
        "json_valid": True,
        "schema_valid": True,
        "category_correct": True,
        "priority_correct": True,
        "device_correct": True,
        "needs_clarification_correct": True,
        "hallucination": False,
        "unknown_missing_fields": ["employee_email"],
        "warnings": ["unknown_missing_fields"],
        "expected": {
            "category": "Access",
            "priority": "P3",
            "device": "Laptop",
            "needs_clarification": True,
            "missing_fields": ["employee_email"],
            "summary": "Need identity details.",
        },
        "actual": {
            "category": "Access",
            "priority": "P3",
            "device": "Laptop",
            "needs_clarification": True,
            "missing_fields": ["employee_email"],
            "summary": "Need identity details.",
        },
        "latency_ms": 11.0,
    }

    with results_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(failing_row) + "\n")
        f.write(json.dumps(passing_row) + "\n")
        f.write(json.dumps(warning_only_row) + "\n")

    with events_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "case-1", "latency_ms": 12.0}) + "\n")
        f.write(json.dumps({"id": "case-2", "latency_ms": 10.0}) + "\n")
        f.write(json.dumps({"id": "case-3", "latency_ms": 11.0}) + "\n")

    summary = summarize_results(str(results_path), str(events_path))
    write_report_md(summary, str(report_path))
    report_text = report_path.read_text(encoding="utf-8")

    assert "## Failures" in report_text
    assert "ID `case-1`" in report_text
    assert "wrong_priority" in report_text
    assert "priority: expected=\"P2\", actual=\"P3\"" in report_text
    assert "unknown_missing_fields_rate" in report_text
    assert "valid_json_only_category_accuracy" in report_text
    assert "## Top Failure Causes" in report_text
    assert "| InvalidJSON |" in report_text
    assert "## Performance" in report_text
    assert "## Category Confusion Summary" in report_text
    assert "## Top Wrong-Category Examples" in report_text
    assert "## Unknown Missing Fields Examples" in report_text
    failures_section = report_text.split("## Failures", maxsplit=1)[1]
    assert "ID `case-3`" not in failures_section


def test_summary_includes_valid_json_only_accuracy_and_failure_causes(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    events_path = tmp_path / "events.jsonl"

    rows = [
        {
            "id": "ok-case",
            "overall_pass": True,
            "json_valid": True,
            "schema_valid": True,
            "category_correct": True,
            "priority_correct": True,
            "device_correct": True,
            "needs_clarification_correct": True,
            "hallucination": False,
            "warnings": [],
            "failure_reasons": [],
            "latency_ms": 10.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
        },
        {
            "id": "bad-json",
            "overall_pass": False,
            "json_valid": False,
            "schema_valid": False,
            "category_correct": False,
            "priority_correct": False,
            "device_correct": False,
            "needs_clarification_correct": False,
            "hallucination": False,
            "warnings": [],
            "failure_reasons": ["invalid_json"],
            "error_type": "ExtractionFailure",
            "raw_text": "preface text only",
            "latency_ms": 12.0,
        },
    ]

    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with events_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "ok-case", "latency_ms": 10.0}) + "\n")
        f.write(json.dumps({"id": "bad-json", "latency_ms": 12.0}) + "\n")

    summary = summarize_results(str(results_path), str(events_path))
    assert summary["valid_json_only_cases"] == 1
    assert summary["accuracy_valid_json_only"]["category"] == 1.0
    assert summary["failure_cause_counts"]["InvalidJSON"] == 1
    assert summary["failure_cause_counts"]["ExtractionFailure"] == 1
    assert summary["tokens"]["total_avg"] == 22.0


def test_evaluate_gates_reports_failures() -> None:
    summary = {
        "accuracy": {"category": 0.70},
        "schema_valid_rate": 0.95,
        "latency_ms": {"p95": 2500.0},
    }

    gates = evaluate_gates(summary)

    assert gates["passed"] is False
    failed_checks = [check["name"] for check in gates["checks"] if not check["passed"]]
    assert "category_accuracy" in failed_checks
    assert "schema_valid_rate" in failed_checks
    assert "latency_p95_ms" in failed_checks


def test_evaluate_gates_uses_ollama_default_latency_profile() -> None:
    summary = {
        "provider": "ollama",
        "accuracy": {"category": 0.90},
        "schema_valid_rate": 1.0,
        "latency_ms": {"p95": 4600.0},
    }
    gates = evaluate_gates(summary)
    assert gates["passed"] is True


def test_report_lists_all_failures_when_many_cases(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    events_path = tmp_path / "events.jsonl"
    report_path = tmp_path / "report.md"

    with results_path.open("w", encoding="utf-8") as rf, events_path.open("w", encoding="utf-8") as ef:
        for idx in range(12):
            row = {
                "id": f"fail-{idx}",
                "overall_pass": False,
                "json_valid": True,
                "schema_valid": True,
                "category_correct": False,
                "priority_correct": True,
                "device_correct": True,
                "needs_clarification_correct": True,
                "hallucination": False,
                "failure_reasons": ["wrong_category"],
                "expected": {"category": "Network"},
                "actual": {"category": "Software"},
                "latency_ms": 10.0,
            }
            rf.write(json.dumps(row) + "\n")
            ef.write(json.dumps({"id": f"fail-{idx}", "latency_ms": 10.0}) + "\n")

    summary = summarize_results(str(results_path), str(events_path))
    write_report_md(summary, str(report_path))
    report_text = report_path.read_text(encoding="utf-8")

    for idx in range(12):
        assert f"ID `fail-{idx}`" in report_text
