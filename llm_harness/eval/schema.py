from __future__ import annotations

REQUIRED_OUTPUT_FIELDS: dict[str, type] = {
    "category": str,
    "priority": str,
    "device": str,
    "needs_clarification": bool,
    "missing_fields": list,
    "summary": str,
}


def validate_required_fields(output: object) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(output, dict):
        return False, ["Output must be a JSON object"]

    for field, expected_type in REQUIRED_OUTPUT_FIELDS.items():
        if field not in output:
            errors.append(f"Missing key: {field}")
            continue

        value = output[field]
        if expected_type is bool:
            if not isinstance(value, bool):
                errors.append(f"Key '{field}' must be bool")
            continue

        if not isinstance(value, expected_type):
            errors.append(f"Key '{field}' must be {expected_type.__name__}")

    return len(errors) == 0, errors
