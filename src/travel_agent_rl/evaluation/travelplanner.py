from __future__ import annotations

import json
import math
import re
from typing import Any


PLAN_FIELDS = (
    "day",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)

FIELD_ALIASES = {
    "day": "day",
    "days": "day",
    "current city": "current_city",
    "city": "current_city",
    "transportation": "transportation",
    "transport": "transportation",
    "breakfast": "breakfast",
    "attraction": "attraction",
    "attractions": "attraction",
    "lunch": "lunch",
    "dinner": "dinner",
    "accommodation": "accommodation",
    "lodging": "accommodation",
    "hotel": "accommodation",
}


def _clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().strip("|*")
    return text or "-"


def _normalise_row(row: dict[str, Any], fallback_day: int) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, value in row.items():
        canonical = FIELD_ALIASES.get(re.sub(r"[_\s]+", " ", str(key)).strip().casefold())
        if canonical:
            mapped[canonical] = _clean(value)
    mapped.setdefault("day", str(fallback_day))
    return {field: mapped.get(field, "-") for field in PLAN_FIELDS}


def _json_plan(text: str) -> list[dict[str, str]] | None:
    candidates = [text.strip()]
    candidates.extend(match.group(1) for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.I | re.S))
    array = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.S)
    if array:
        candidates.append(array.group(1))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("plan"), list):
            value = value["plan"]
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return [_normalise_row(item, index) for index, item in enumerate(value, 1)]
    return None


def _markdown_table(text: str) -> list[dict[str, str]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    for index in range(len(lines) - 2):
        headers = [cell.strip().casefold() for cell in lines[index].strip("|").split("|")]
        divider = [cell.strip() for cell in lines[index + 1].strip("|").split("|")]
        if len(headers) < 2 or len(divider) != len(headers):
            continue
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in divider):
            continue
        mapped_headers = [FIELD_ALIASES.get(re.sub(r"[_\s]+", " ", h).strip(" *")) for h in headers]
        if "day" not in mapped_headers:
            continue
        result = []
        for line in lines[index + 2 :]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(headers):
                break
            row = {key: value for key, value in zip(mapped_headers, cells) if key}
            result.append(_normalise_row(row, len(result) + 1))
        return result or None
    return None


def _labelled_day_blocks(text: str) -> list[dict[str, str]] | None:
    marker = re.compile(
        r"(?im)^(?:\s*[-*]\s*)?(?:#{1,6}\s*)?(?:\*{0,2})?"
        r"(?:day\s*(\d+)|[^\n:]{0,30}\(day\s*(\d+)\))\*{0,2}\s*[:\-]?"
    )
    matches = list(marker.finditer(text))
    if not matches:
        return None
    result = []
    label_pattern = re.compile(
        r"(?i)(current[ _]city|city|transportation|transport|breakfast|attractions?|"
        r"lunch|dinner|accommodation|lodging|hotel)\s*:\s*"
        r"(.+?)(?=(?:current[ _]city|city|transportation|transport|breakfast|attractions?|"
        r"lunch|dinner|accommodation|lodging|hotel)\s*:|$)"
    )
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        day = int(match.group(1) or match.group(2))
        row: dict[str, Any] = {"day": day}
        for field, value in label_pattern.findall(body):
            key = FIELD_ALIASES[re.sub(r"[_\s]+", " ", field).casefold()]
            row[key] = value.strip(" ;,\n")
        if not any(key != "day" for key in row):
            row["attraction"] = body
        result.append(_normalise_row(row, day))
    return result


def adapt_final_answer(answer: str | None) -> list[dict[str, str]] | None:
    """Convert only explicitly stated final-answer content to official plan JSON.

    The parser is deterministic and deliberately does not consult the query, tool
    observations, validation references, or evaluator feedback. Missing fields are
    represented as ``-`` rather than inferred.
    """
    if not answer or not answer.strip():
        return None
    for parser in (_json_plan, _markdown_table, _labelled_day_blocks):
        plan = parser(answer)
        if plan:
            return plan
    return None


def is_structurally_complete(plan: list[dict[str, Any]] | None, expected_days: int) -> bool:
    return bool(
        plan
        and len(plan) == expected_days
        and all(tuple(row) == PLAN_FIELDS for row in plan)
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    margin /= denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]
