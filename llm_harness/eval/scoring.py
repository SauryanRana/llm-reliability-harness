from __future__ import annotations

import re
from collections.abc import Collection

from .schema import validate_required_fields

DEVICE_HINT_PATTERN = re.compile(r"\b(windows|mac|iphone|android)\b", re.IGNORECASE)


def score_case(
    expected: dict,
    actual: object,
    input_text: str,
    allowed_missing_fields: Collection[str] | None = None,
) -> dict:
    json_valid = isinstance(actual, dict)
    if json_valid:
        schema_valid, schema_errors = validate_required_fields(actual)
    else:
        schema_valid, schema_errors = False, ["Output must be a JSON object"]

    category_correct = _field_correct(expected, actual, "category")
    priority_correct = _field_correct(expected, actual, "priority")
    device_correct = _field_correct(expected, actual, "device")
    needs_clarification_correct = _field_correct(expected, actual, "needs_clarification")
    hallucination, hallucination_reasons = _hallucination_checks(actual)
    extraction_failure_device_unknown = _is_device_extraction_failure(actual, input_text)
    unknown_missing_fields = _find_unknown_missing_fields(actual, allowed_missing_fields or [])

    warnings: list[str] = []
    if unknown_missing_fields:
        warnings.append("unknown_missing_fields")

    failure_reasons: list[str] = []
    if not json_valid:
        failure_reasons.append("invalid_json")
    if not schema_valid:
        failure_reasons.append("schema_error")
    if not category_correct:
        failure_reasons.append("wrong_category")
    if not priority_correct:
        failure_reasons.append("wrong_priority")
    if not device_correct:
        failure_reasons.append("wrong_device")
    if not needs_clarification_correct:
        failure_reasons.append("wrong_needs_clarification")
    if hallucination:
        failure_reasons.append("hallucination")
        failure_reasons.extend(hallucination_reasons)
    if extraction_failure_device_unknown:
        failure_reasons.append("extraction_failure_device_unknown")

    overall_pass = (
        json_valid
        and schema_valid
        and category_correct
        and priority_correct
        and device_correct
        and needs_clarification_correct
        and not hallucination
    )

    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "category_correct": category_correct,
        "priority_correct": priority_correct,
        "device_correct": device_correct,
        "needs_clarification_correct": needs_clarification_correct,
        "hallucination": hallucination,
        "unknown_missing_fields": unknown_missing_fields,
        "warnings": warnings,
        "extraction_failure_device_unknown": extraction_failure_device_unknown,
        "failure_reasons": _dedupe(failure_reasons),
        "overall_pass": overall_pass,
    }


def _field_correct(expected: dict, actual: object, field: str) -> bool:
    if not isinstance(actual, dict):
        return False
    return expected.get(field) == actual.get(field)


def _hallucination_checks(actual: object) -> tuple[bool, list[str]]:
    if not isinstance(actual, dict):
        return False, []

    reasons: list[str] = []
    needs_clarification = actual.get("needs_clarification")
    missing_fields = actual.get("missing_fields")
    if needs_clarification is False and isinstance(missing_fields, list) and len(missing_fields) > 0:
        reasons.append("missing_fields_without_clarification")

    if needs_clarification is True and isinstance(missing_fields, list) and len(missing_fields) == 0:
        reasons.append("clarification_without_missing_fields")

    return len(reasons) > 0, reasons


def _is_device_extraction_failure(actual: object, input_text: str) -> bool:
    if not isinstance(actual, dict):
        return False
    device = actual.get("device")
    if device != "Unknown":
        return False
    return bool(DEVICE_HINT_PATTERN.search(input_text))


def _find_unknown_missing_fields(actual: object, allowed_missing_fields: Collection[str]) -> list[str]:
    if not isinstance(actual, dict):
        return []

    missing_fields = actual.get("missing_fields")
    if not isinstance(missing_fields, list):
        return []

    allowed = set(allowed_missing_fields)
    unknown: list[str] = []
    for field in missing_fields:
        if isinstance(field, str) and field in allowed:
            continue
        unknown.append(str(field))
    return unknown


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
