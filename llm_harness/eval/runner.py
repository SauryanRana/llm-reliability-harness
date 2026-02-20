from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_harness.eval.dataset import load_jsonl
from llm_harness.eval.normalize import normalize_output
from llm_harness.eval.rules import build_output_from_signals, coerce_ticket_signals, looks_like_ticket_signals
from llm_harness.eval.scoring import score_case
from llm_harness.logging.events import append_jsonl
from llm_harness.providers import get_provider
from llm_harness.providers.base import ProviderResult


def run_eval(
    dataset_path: str,
    provider: str,
    model: str,
    out_results_path: str,
    out_events_path: str,
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 60,
    temperature: float = 0.0,
    num_predict: int = 320,
    num_ctx: int = 2048,
    json_mode: bool = False,
    show_progress: bool = True,
    progress_every: int = 3,
) -> dict:
    dataset = load_jsonl(dataset_path)
    total_cases = len(dataset)
    provider_impl = get_provider(
        provider,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
        json_mode=json_mode,
    )
    allowed_missing_fields = _build_allowed_missing_fields(dataset)

    _reset_jsonl_file(out_results_path)
    _reset_jsonl_file(out_events_path)

    if show_progress:
        print(
            f"Running {total_cases} cases with provider={provider} model={model} "
            f"num_predict={num_predict} num_ctx={num_ctx} json_mode={json_mode}"
        )

    for idx, row in enumerate(dataset, start=1):
        case_id = str(row.get("id", ""))
        input_text = str(row.get("input_text", ""))
        expected = _coerce_expected(row.get("expected"), case_id)
        try:
            provider_result = provider_impl.generate(row, model)
        except Exception as exc:
            provider_result = ProviderResult(
                actual=None,
                latency_ms=0.0,
                status="error",
                error_type=type(exc).__name__,
                error_msg=str(exc),
                prompt_chars=len(input_text),
            )

        normalized_actual = provider_result.actual
        normalization_warnings: list[str] = []
        key_signals: dict | None = None
        if isinstance(provider_result.actual, dict):
            if provider == "ollama" and looks_like_ticket_signals(provider_result.actual):
                signal_payload = coerce_ticket_signals(provider_result.actual)
                key_signals = _signal_snapshot(signal_payload)
                normalized_actual, normalization_warnings = build_output_from_signals(
                    signal_payload,
                    input_text,
                    allowed_missing_fields=allowed_missing_fields,
                )
            normalized_actual, post_warnings = normalize_output(normalized_actual, input_text)
            normalization_warnings = _dedupe_warnings(normalization_warnings + post_warnings)

        score = score_case(
            expected,
            normalized_actual,
            input_text=input_text,
            allowed_missing_fields=allowed_missing_fields,
        )
        combined_warnings = _dedupe_warnings(score.get("warnings", []) + normalization_warnings)

        result_row = {
            "id": case_id,
            "provider": provider,
            "model": model,
            "expected": expected,
            "actual": normalized_actual,
            "usage": provider_result.usage,
            "latency_ms": round(float(provider_result.latency_ms), 3),
            **score,
            "warnings": combined_warnings,
        }
        if key_signals is not None:
            result_row["key_signals"] = key_signals
        if not score["json_valid"] and provider_result.raw_text:
            result_row["raw_text"] = provider_result.raw_text[:500]
        if provider_result.status == "error":
            result_row["error_type"] = provider_result.error_type
            result_row["error_msg"] = provider_result.error_msg
        append_jsonl(out_results_path, result_row)

        event_status = provider_result.status
        event_error_type = provider_result.error_type
        event_error_msg = provider_result.error_msg
        if event_status == "ok" and (not score["json_valid"] or not score["schema_valid"]):
            event_status = "error"
            if not score["json_valid"]:
                event_error_type = "InvalidJSON"
                event_error_msg = "Output is not valid JSON object"
            elif not score["schema_valid"]:
                event_error_type = "SchemaValidationError"
                event_error_msg = "; ".join(score.get("schema_errors", []))

        event_row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "case_id": case_id,
            "latency_ms": round(float(provider_result.latency_ms), 3),
            "status": event_status,
            "json_valid": score["json_valid"],
            "schema_valid": score["schema_valid"],
            "input_chars": len(input_text),
            "response_chars": provider_result.response_chars,
        }
        if event_status == "error":
            event_row["error_type"] = event_error_type
            event_row["error_msg"] = (event_error_msg or "")[:300]
        append_jsonl(out_events_path, event_row)

        should_print = show_progress and (idx % max(1, progress_every) == 0 or idx == total_cases)
        if should_print:
            if not score["json_valid"]:
                raw_preview = (provider_result.raw_text or "")[:80].replace("\n", " ")
                print(
                    f"[{idx}/{total_cases}] {case_id} parse_failed "
                    f"latency={provider_result.latency_ms:.1f}ms raw='{raw_preview}'"
                )
            else:
                print(
                    f"[{idx}/{total_cases}] {case_id} {event_status} "
                    f"json_valid={score['json_valid']} "
                    f"schema_valid={score['schema_valid']} "
                    f"latency={provider_result.latency_ms:.1f}ms"
                )

    return {
        "total_cases": total_cases,
        "results_path": out_results_path,
        "events_path": out_events_path,
    }


def _coerce_expected(value: object, case_id: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Case {case_id}: expected field is not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Case {case_id}: expected field must be a JSON object")


def _build_allowed_missing_fields(dataset: list[dict]) -> set[str]:
    allowed: set[str] = set()
    for row in dataset:
        case_id = str(row.get("id", ""))
        expected = _coerce_expected(row.get("expected"), case_id)
        missing_fields = expected.get("missing_fields")
        if not isinstance(missing_fields, list):
            continue
        for field in missing_fields:
            if isinstance(field, str):
                allowed.add(field)
    return allowed


def _reset_jsonl_file(path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def _dedupe_warnings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _signal_snapshot(signals: dict) -> dict:
    keys = [
        "device_hint",
        "mentions_vpn",
        "mentions_email",
        "mentions_wifi_or_network",
        "mentions_printer",
        "mentions_software_app",
        "mentions_laptop_issue",
        "access_request",
        "security_incident",
        "scope",
        "urgency_words",
    ]
    snapshot: dict = {}
    for key in keys:
        if key in signals:
            snapshot[key] = signals[key]
    return snapshot
