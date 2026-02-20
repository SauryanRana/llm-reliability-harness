from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

DEFAULT_GATES = {
    "category_accuracy_min": 0.85,
    "schema_valid_rate_min": 1.0,
    "latency_p95_ms_max": 2000.0,
}
OLLAMA_LOCAL_GATES = {
    "latency_p95_ms_max": 6000.0,
}

FAILURE_CAUSE_ORDER = [
    "InvalidJSON",
    "SchemaError",
    "EmptyOutput",
    "ExtractionFailure",
    "RuleConflict",
]

RULE_CONFLICT_REASONS = {
    "missing_fields_without_clarification",
    "clarification_without_missing_fields",
}
RULE_CONFLICT_WARNINGS = {
    "needs_clarification_without_missing_fields",
    "coerced_needs_clarification_true",
}


def summarize_results(results_path: str, events_path: str) -> dict:
    results = _load_jsonl(results_path)
    events = _load_jsonl(events_path)

    total_cases = len(results)
    json_valid_rate = _rate(results, "json_valid")
    schema_valid_rate = _rate(results, "schema_valid")
    category_accuracy = _rate(results, "category_correct")
    priority_accuracy = _rate(results, "priority_correct")
    device_accuracy = _rate(results, "device_correct")
    needs_clarification_accuracy = _rate(results, "needs_clarification_correct")
    hallucination_rate = _rate(results, "hallucination")
    unknown_missing_fields_rate = _non_empty_list_rate(results, "unknown_missing_fields")
    extraction_failure_rate = _rate(results, "extraction_failure_device_unknown")

    valid_rows = [
        row
        for row in results
        if isinstance(row, dict) and bool(row.get("json_valid")) and bool(row.get("schema_valid"))
    ]
    valid_json_only_accuracy = {
        "category": _rate(valid_rows, "category_correct"),
        "priority": _rate(valid_rows, "priority_correct"),
        "device": _rate(valid_rows, "device_correct"),
        "needs_clarification": _rate(valid_rows, "needs_clarification_correct"),
    }

    latencies = [
        float(event["latency_ms"])
        for event in events
        if isinstance(event, dict) and isinstance(event.get("latency_ms"), (int, float))
    ]
    if not latencies:
        latencies = [
            float(result["latency_ms"])
            for result in results
            if isinstance(result, dict) and isinstance(result.get("latency_ms"), (int, float))
        ]

    provider = _single_string_value(results, "provider")
    model = _single_string_value(results, "model")

    return {
        "provider": provider,
        "model": model,
        "total_cases": total_cases,
        "json_valid_rate": json_valid_rate,
        "schema_valid_rate": schema_valid_rate,
        "accuracy": {
            "category": category_accuracy,
            "priority": priority_accuracy,
            "device": device_accuracy,
            "needs_clarification": needs_clarification_accuracy,
        },
        "valid_json_only_cases": len(valid_rows),
        "accuracy_valid_json_only": valid_json_only_accuracy,
        "hallucination_rate": hallucination_rate,
        "unknown_missing_fields_rate": unknown_missing_fields_rate,
        "extraction_failure_device_unknown_rate": extraction_failure_rate,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "tokens": _token_summary(results),
        "failure_cause_counts": _failure_cause_counts(results),
        "category_confusions": _category_confusions(results, limit=5),
        "wrong_category_examples": _top_wrong_category_examples(results, limit=5),
        "remaining_misses": _remaining_category_misses(results),
        "unknown_missing_fields_examples": _unknown_missing_fields_examples(results, limit=5),
        "failures": _top_failures(results),
    }


def evaluate_gates(summary: dict, gates: dict[str, float] | None = None) -> dict:
    limits = _resolve_gate_limits(summary, gates)
    checks = [
        {
            "name": "category_accuracy",
            "passed": summary["accuracy"]["category"] >= limits["category_accuracy_min"],
            "actual": summary["accuracy"]["category"],
            "threshold": f">= {limits['category_accuracy_min']:.2f}",
        },
        {
            "name": "schema_valid_rate",
            "passed": _float_eq(summary["schema_valid_rate"], limits["schema_valid_rate_min"]),
            "actual": summary["schema_valid_rate"],
            "threshold": f"= {limits['schema_valid_rate_min']:.2f}",
        },
        {
            "name": "latency_p95_ms",
            "passed": summary["latency_ms"]["p95"] <= limits["latency_p95_ms_max"],
            "actual": summary["latency_ms"]["p95"],
            "threshold": f"<= {limits['latency_p95_ms_max']:.0f}",
        },
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def write_report_md(summary: dict, out_path: str, gates: dict | None = None) -> None:
    gate_summary = evaluate_gates(summary, gates=gates)
    latency_threshold = _resolve_gate_limits(summary, gates)["latency_p95_ms_max"]
    lines = [
        "# Eval Report",
        "",
        f"- Provider: {summary.get('provider', 'unknown')}",
        f"- Model: {summary.get('model', 'unknown')}",
        f"- Total cases: {summary['total_cases']}",
        f"- json_valid_rate: {_pct(summary['json_valid_rate'])}",
        f"- schema_valid_rate: {_pct(summary['schema_valid_rate'])}",
        f"- category_accuracy: {_pct(summary['accuracy']['category'])}",
        f"- priority_accuracy: {_pct(summary['accuracy']['priority'])}",
        f"- device_accuracy: {_pct(summary['accuracy']['device'])}",
        f"- needs_clarification_accuracy: {_pct(summary['accuracy']['needs_clarification'])}",
        f"- valid_json_only_cases: {summary['valid_json_only_cases']}",
        f"- valid_json_only_category_accuracy: {_pct(summary['accuracy_valid_json_only']['category'])}",
        f"- valid_json_only_priority_accuracy: {_pct(summary['accuracy_valid_json_only']['priority'])}",
        f"- valid_json_only_device_accuracy: {_pct(summary['accuracy_valid_json_only']['device'])}",
        (
            "- valid_json_only_needs_clarification_accuracy: "
            f"{_pct(summary['accuracy_valid_json_only']['needs_clarification'])}"
        ),
        f"- hallucination_rate: {_pct(summary['hallucination_rate'])}",
        f"- unknown_missing_fields_rate: {_pct(summary['unknown_missing_fields_rate'])}",
        f"- extraction_failure_device_unknown_rate: {_pct(summary['extraction_failure_device_unknown_rate'])}",
        "",
    ]

    lines.extend(
        [
            "## Performance",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| latency_p50_ms | {summary['latency_ms']['p50']:.2f} |",
            f"| latency_p95_ms | {summary['latency_ms']['p95']:.2f} |",
            f"| latency_p95_gate_ms | {latency_threshold:.0f} |",
            f"| prompt_tokens_avg | {_fmt_num(summary['tokens']['prompt_avg'])} |",
            f"| completion_tokens_avg | {_fmt_num(summary['tokens']['completion_avg'])} |",
            f"| total_tokens_avg | {_fmt_num(summary['tokens']['total_avg'])} |",
            "",
        ]
    )

    lines.extend(
        [
            "## Gates",
            "",
            f"- status: {'PASS' if gate_summary['passed'] else 'FAIL'}",
            "",
            "| Check | Status | Actual | Threshold |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for check in gate_summary["checks"]:
        status = "pass" if check["passed"] else "fail"
        actual_value = check["actual"]
        if isinstance(actual_value, float):
            actual_text = f"{actual_value:.3f}"
        else:
            actual_text = str(actual_value)
        lines.append(f"| {check['name']} | {status} | {actual_text} | {check['threshold']} |")

    lines.extend(["", "## Top Failure Causes", "", "| Cause | Count |", "| --- | ---: |"])
    for cause in FAILURE_CAUSE_ORDER:
        lines.append(f"| {cause} | {summary['failure_cause_counts'].get(cause, 0)} |")

    lines.extend(["", "## Category Confusion Summary", ""])
    confusions = summary.get("category_confusions", [])
    if not confusions:
        lines.append("- None")
    else:
        lines.append("| Expected | Actual | Count |")
        lines.append("| --- | --- | ---: |")
        for row in confusions:
            lines.append(f"| {row['expected']} | {row['actual']} | {row['count']} |")

    lines.extend(["", "## Top Wrong-Category Examples", ""])
    wrong_category_examples = summary.get("wrong_category_examples", [])
    if not wrong_category_examples:
        lines.append("- None")
    else:
        for example in wrong_category_examples:
            lines.append(
                f"- ID `{example['id']}` expected=`{example['expected_category']}` actual=`{example['actual_category']}`"
            )
            key_signals = example.get("key_signals")
            if isinstance(key_signals, dict) and key_signals:
                lines.append(f"  key_signals: {_render_value(key_signals)}")

    lines.extend(["", "## Remaining Misses", ""])
    remaining_misses = summary.get("remaining_misses", [])
    if not remaining_misses:
        lines.append("- None")
    else:
        for miss in remaining_misses:
            lines.append(
                f"- ID `{miss['id']}` category expected=`{miss['expected_category']}` actual=`{miss['actual_category']}`"
            )

    lines.extend(["", "## Unknown Missing Fields Examples", ""])
    unknown_examples = summary.get("unknown_missing_fields_examples", [])
    if not unknown_examples:
        lines.append("- None")
    else:
        for example in unknown_examples:
            lines.append(f"- ID `{example['id']}` unknown_missing_fields={_render_value(example['unknown_missing_fields'])}")

    lines.extend(["", "## Failures", ""])
    failures = summary.get("failures", [])
    lines.append(f"- total_failures: {len(failures)}")
    if not failures:
        lines.append("- None")
    else:
        for failure in failures:
            lines.append(f"- ID `{failure['id']}`")
            lines.append(f"  reasons: {', '.join(failure['reasons']) if failure['reasons'] else 'unknown'}")
            if not failure["differences"]:
                lines.append("  diff: none")
                continue
            diff_chunks = [
                f"{diff['field']}: expected={_render_value(diff['expected'])}, actual={_render_value(diff['actual'])}"
                for diff in failure["differences"]
            ]
            lines.append(f"  diff: {'; '.join(diff_chunks)}")

    lines.append("")

    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_gate_limits(summary: dict, gates: dict[str, float] | None) -> dict[str, float]:
    limits = dict(DEFAULT_GATES)
    provider = str(summary.get("provider", "")).lower()
    if provider == "ollama":
        limits.update(OLLAMA_LOCAL_GATES)

    env_latency = os.getenv("LLMH_LATENCY_P95_MS_THRESHOLD")
    if env_latency:
        try:
            parsed = float(env_latency)
            if parsed > 0:
                limits["latency_p95_ms_max"] = parsed
        except ValueError:
            pass

    if gates:
        limits.update(gates)
    return limits


def _single_string_value(rows: list[dict], key: str) -> str:
    values: set[str] = set()
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return "unknown"
    return "mixed"


def _category_confusions(results: list[dict], limit: int) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    for row in results:
        expected = row.get("expected")
        actual = row.get("actual")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            continue
        expected_category = expected.get("category")
        actual_category = actual.get("category")
        if not isinstance(expected_category, str) or not isinstance(actual_category, str):
            continue
        if expected_category == actual_category:
            continue
        key = (expected_category, actual_category)
        counts[key] = counts.get(key, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        {"expected": expected, "actual": actual, "count": count}
        for (expected, actual), count in ordered[:limit]
    ]


def _top_wrong_category_examples(results: list[dict], limit: int) -> list[dict]:
    out: list[dict] = []
    for row in results:
        expected = row.get("expected")
        actual = row.get("actual")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            continue
        expected_category = expected.get("category")
        actual_category = actual.get("category")
        if not isinstance(expected_category, str) or not isinstance(actual_category, str):
            continue
        if expected_category == actual_category:
            continue
        example = {
            "id": str(row.get("id", "")),
            "expected_category": expected_category,
            "actual_category": actual_category,
        }
        key_signals = row.get("key_signals")
        if isinstance(key_signals, dict):
            example["key_signals"] = key_signals
        out.append(example)
        if len(out) >= limit:
            break
    return out


def _unknown_missing_fields_examples(results: list[dict], limit: int) -> list[dict]:
    out: list[dict] = []
    for row in results:
        unknown = row.get("unknown_missing_fields")
        if not isinstance(unknown, list) or not unknown:
            continue
        out.append({"id": str(row.get("id", "")), "unknown_missing_fields": [str(value) for value in unknown]})
        if len(out) >= limit:
            break
    return out


def _remaining_category_misses(results: list[dict]) -> list[dict]:
    misses: list[dict] = []
    for row in results:
        if bool(row.get("category_correct", True)):
            continue
        expected = row.get("expected")
        actual = row.get("actual")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            continue
        expected_category = expected.get("category")
        actual_category = actual.get("category")
        if not isinstance(expected_category, str) or not isinstance(actual_category, str):
            continue
        misses.append(
            {
                "id": str(row.get("id", "")),
                "expected_category": expected_category,
                "actual_category": actual_category,
            }
        )
    return misses


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    input_path = Path(path)
    if not input_path.exists():
        return rows

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    good = sum(1 for row in rows if bool(row.get(key)))
    return good / len(rows)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(1, math.ceil((percentile / 100) * len(sorted_values)))
    return sorted_values[rank - 1]


def _non_empty_list_rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    flagged = 0
    for row in rows:
        value = row.get(key)
        if isinstance(value, list) and len(value) > 0:
            flagged += 1
    return flagged / len(rows)


def _token_summary(results: list[dict]) -> dict:
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []

    for row in results:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if isinstance(prompt, int):
            prompt_tokens.append(prompt)
        if isinstance(completion, int):
            completion_tokens.append(completion)
        if isinstance(total, int):
            total_tokens.append(total)

    return {
        "prompt_avg": _avg(prompt_tokens),
        "completion_avg": _avg(completion_tokens),
        "total_avg": _avg(total_tokens),
    }


def _avg(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _failure_cause_counts(results: list[dict]) -> dict[str, int]:
    counts = {cause: 0 for cause in FAILURE_CAUSE_ORDER}

    for row in results:
        json_valid = bool(row.get("json_valid"))
        schema_valid = bool(row.get("schema_valid"))
        error_type = str(row.get("error_type", ""))
        raw_text = row.get("raw_text")

        if not json_valid:
            counts["InvalidJSON"] += 1
        if json_valid and not schema_valid:
            counts["SchemaError"] += 1
        if error_type == "EmptyOutput" or (not json_valid and isinstance(raw_text, str) and not raw_text.strip()):
            counts["EmptyOutput"] += 1
        if error_type == "ExtractionFailure":
            counts["ExtractionFailure"] += 1
        if _is_rule_conflict(row):
            counts["RuleConflict"] += 1

    return counts


def _is_rule_conflict(row: dict) -> bool:
    reasons = row.get("failure_reasons")
    if isinstance(reasons, list):
        for reason in reasons:
            if str(reason) in RULE_CONFLICT_REASONS:
                return True

    warnings = row.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            if str(warning) in RULE_CONFLICT_WARNINGS:
                return True
    return False


def _top_failures(results: list[dict]) -> list[dict]:
    failures: list[dict] = []
    for row in results:
        if row.get("overall_pass", False):
            continue

        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
        actual = row.get("actual") if isinstance(row.get("actual"), dict) else {}
        failures.append(
            {
                "id": str(row.get("id", "")),
                "reasons": _reasons_from_row(row),
                "differences": _diff_fields(expected, actual),
            }
        )
    return failures


def _reasons_from_row(row: dict) -> list[str]:
    reasons = row.get("failure_reasons")
    if isinstance(reasons, list) and reasons:
        return [str(reason) for reason in reasons]

    fallback: list[str] = []
    if not row.get("json_valid", True):
        fallback.append("invalid_json")
    if not row.get("schema_valid", True):
        fallback.append("schema_error")
    if not row.get("category_correct", True):
        fallback.append("wrong_category")
    if not row.get("priority_correct", True):
        fallback.append("wrong_priority")
    if not row.get("device_correct", True):
        fallback.append("wrong_device")
    if not row.get("needs_clarification_correct", True):
        fallback.append("wrong_needs_clarification")
    if row.get("hallucination", False):
        fallback.append("hallucination")
    return fallback


def _diff_fields(expected: dict[str, Any], actual: dict[str, Any]) -> list[dict]:
    fields = sorted(set(expected.keys()) | set(actual.keys()))
    out: list[dict] = []
    for field in fields:
        if expected.get(field) == actual.get(field):
            continue
        out.append({"field": field, "expected": expected.get(field), "actual": actual.get(field)})
    return out


def _render_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _float_eq(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps
