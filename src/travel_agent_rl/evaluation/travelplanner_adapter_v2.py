from __future__ import annotations

import json
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
    "current_city": "current_city",
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

_LABELS = "|".join(sorted((re.escape(key) for key in FIELD_ALIASES), key=len, reverse=True))
_DAY_MARKER = re.compile(
    r"(?im)^(?:\s*[-*]\s*)?(?:#{1,6}\s*)?(?:\*{0,2})?"
    r"(?:day\s*(\d+)|[^\n:]{0,40}\(day\s*(\d+)\))\*{0,2}\s*[:\-]?"
)
_DAY_RANGE_BOUNDARY = re.compile(
    r"(?im)^(?:\s*[-*]\s*)?(?:#{1,6}\s*)?(?:\*{0,2})?"
    r"[^\n:]{0,40}\(days\s*\d+\s*[-–]\s*\d+\)\*{0,2}\s*[:\-]?"
)
_GLOBAL_SECTION = re.compile(
    r"(?im)^\s*(?:---\s*)?(?:#{1,6}\s*)?(?:[-*]\s*)?(?:\*{0,2})?"
    r"(?:summary|total(?:\s+estimated|\s+flight|\s+driving|\s+lodging|\s+cost)?|"
    r"budget(?:\s+summary|\s+remaining)?|cost\s+summary|cuisine\s+highlights|notes?)"
    r"(?:\*{0,2})?\s*(?::|$)"
)


def _clean(value: Any) -> str:
    if value is None:
        return "-"
    text = re.sub(r"\s+", " ", str(value).replace("*", "")).strip().strip("|")
    return text or "-"


def _canonical_key(key: Any) -> str | None:
    normalised = re.sub(r"[_\s]+", " ", str(key)).strip().casefold()
    return FIELD_ALIASES.get(normalised)


def _normalise_explicit_row(row: dict[str, Any]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, value in row.items():
        canonical = _canonical_key(key)
        if canonical:
            mapped[canonical] = _clean(value)
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
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return [_normalise_explicit_row(item) for item in value]
    return None


def _markdown_table(text: str) -> list[dict[str, str]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    for index in range(len(lines) - 2):
        headers = [_canonical_key(cell.strip(" *")) for cell in lines[index].strip("|").split("|")]
        divider = [cell.strip() for cell in lines[index + 1].strip("|").split("|")]
        if len(headers) < 2 or len(divider) != len(headers):
            continue
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in divider):
            continue
        if "day" not in headers:
            continue
        result: list[dict[str, str]] = []
        for line in lines[index + 2 :]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(headers):
                break
            row = {key: value for key, value in zip(headers, cells) if key}
            result.append(_normalise_explicit_row(row))
        return result or None
    return None


def _trim_clause(value: str) -> str:
    value = re.split(
        r"(?i)(?:\.(?=\s+(?:[A-Z*]|$))|;|\n|\bthen\b|\bafter(?:wards)?\b|\bnext\b|"
        r"\b(?:breakfast|lunch|dinner)\s+at\b|\bvisit\b|\bexplore\b|"
        r"\btour\b|\band\s+enjoy\b|\bstay\s+at\b|\bcheck\s+into\b)",
        value,
        maxsplit=1,
    )[0]
    return _clean(value.strip(" :-,;."))


def _explicit_label_values(body: str) -> dict[str, str]:
    values: dict[str, str] = {}
    pattern = re.compile(
        rf"(?is)(?<!\w)({_LABELS})\*{{0,2}}\s*:\s*(.+?)"
        rf"(?=\n\s*[-*]\s+|(?:\n\s*[-*]?\s*)?"
        rf"(?:\*{{0,2}}(?:{_LABELS})\*{{0,2}}\s*:)|$)"
    )
    for label, value in pattern.findall(body):
        canonical = _canonical_key(label)
        if canonical and canonical != "day":
            cleaned = _clean(value.strip(" ;,\n"))
            if canonical == "accommodation":
                cleaned = re.sub(r"(?i)^-?\s*option\s*:\s*", "", cleaned)
            values[canonical] = cleaned
    return values


def _append(values: dict[str, str], field: str, value: str) -> None:
    value = _clean(value)
    if value == "-":
        return
    if field in values and values[field] != "-":
        existing = [item.strip() for item in values[field].split(";")]
        if value not in existing:
            values[field] = f"{values[field].rstrip(';')}; {value}"
    else:
        values[field] = value


def _official_transport(phrase: str, origin: str, destination: str) -> str:
    if re.search(r"(?i)\b(?:fly|flight)\b", phrase):
        flight = re.search(r"(?i)\b(F\d+)\b", phrase)
        if flight:
            value = f"Flight Number: {flight.group(1)}, from {origin} to {destination}"
            times = re.search(r"\b(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})\b", phrase)
            if times:
                value += f", Departure Time: {times.group(1)}, Arrival Time: {times.group(2)}"
            cost = re.search(r"\$\s*([\d,.]+)", phrase)
            if cost:
                value += f", Cost: {cost.group(1).rstrip(',.')}"
            return value
    if re.match(r"(?i)^drive\b", phrase):
        return re.sub(r"(?i)^drive\b", "Self-driving", phrase, count=1)
    return phrase


def _extract_explicit_prose(body: str, values: dict[str, str]) -> None:
    labelled_attraction = "attraction" in values
    movement = re.search(
        r"(?i)\b((?:take\s+(?:a\s+)?(?:flight|train|bus|taxi)|fly|drive|"
        r"self[- ]driv(?:e|ing)|travel)\s+from\s+([^,.;\n]+?)\s+to\s+"
        r"([^,(.;\n]+)(?:\s*\([^.;\n]*\))?)",
        body,
    )
    if movement:
        phrase = _trim_clause(movement.group(1))
        origin = _clean(movement.group(2).strip())
        destination = _trim_clause(movement.group(3)).split("(", 1)[0].strip()
        values.setdefault("transportation", _official_transport(phrase, origin, destination))
        values.setdefault("current_city", f"from {origin} to {destination}")

    depart_parenthetical = re.search(
        r"(?i)\bdepart\s+([^,.;\n(]+?)\s*\(([^)]*)\)\s+to\s+([^,.;\n]+)", body
    )
    if not movement and depart_parenthetical:
        origin, detail, destination = (_clean(value) for value in depart_parenthetical.groups())
        phrase = f"Flight ({detail}), from {origin} to {destination}"
        values.setdefault("transportation", _official_transport(phrase, origin, destination))
        values.setdefault("current_city", f"from {origin} to {destination}")

    depart_to = re.search(
        r"(?i)\bdepart\s+([^,.;\n]+?)\s+to\s+([^,.;\n]+?)\s+via\s+"
        r"(flight|train|bus|taxi|car)(?:\s*\(([^)]*)\))?",
        body,
    )
    if not movement and not depart_parenthetical and depart_to:
        origin = _clean(depart_to.group(1))
        destination = _clean(depart_to.group(2))
        detail = f" ({depart_to.group(4)})" if depart_to.group(4) else ""
        phrase = f"{depart_to.group(3).title()}{detail}, from {origin} to {destination}"
        values.setdefault("transportation", _official_transport(phrase, origin, destination))
        values.setdefault("current_city", f"from {origin} to {destination}")

    depart_arrive = re.search(
        r"(?is)\bdepart\s+([^,.;\n]+?)\s+at\s+([^,.;\n]+?)\s*"
        r"\(flight\s+([^,;)]+)(?:,\s*\$?([^);]+))?\).*?"
        r"\barrive\s+in\s+([^,.;\n]+?)\s+at\s+([^,.;\n]+)",
        body,
    )
    if not movement and not depart_parenthetical and not depart_to and depart_arrive:
        origin, departure, flight, cost, destination, arrival = depart_arrive.groups()
        transport = (
            f"Flight Number: {_clean(flight)}, from {_clean(origin)} to {_clean(destination)}, "
            f"Departure Time: {_clean(departure)}, Arrival Time: {_clean(arrival)}"
        )
        if cost:
            transport += f", Cost: {_clean(cost)}"
        values.setdefault("transportation", transport)
        values.setdefault("current_city", f"from {_clean(origin)} to {_clean(destination)}")

    depart_drive = re.search(
        r"(?is)\bdepart\s+([^,.;\n]+)\.\s*drive\s+to\s+"
        r"([^,.;\n(]+)\s*(\([^)]*\))?",
        body,
    )
    depart_return = re.search(
        r"(?is)\bdepart\s+([^,.;\n]+)\.\s*return\s+to\s+"
        r"([^,.;\n]+?)\s+via\s+(drive|flight|train|bus)\s*(\([^)]*\))?",
        body,
    )
    if (
        not movement
        and not depart_parenthetical
        and not depart_to
        and not depart_arrive
        and (depart_drive or depart_return)
    ):
        if depart_drive:
            origin, destination, detail = depart_drive.groups()
            mode = "drive"
        else:
            assert depart_return is not None
            origin, destination, mode, detail = depart_return.groups()
        phrase = f"{mode.title()} from {_clean(origin)} to {_clean(destination)}"
        if detail:
            phrase += f" {_clean(detail)}"
        values.setdefault(
            "transportation",
            _official_transport(phrase, _clean(origin), _clean(destination)),
        )
        values.setdefault("current_city", f"from {_clean(origin)} to {_clean(destination)}")

    if "transportation" not in values:
        return_only = re.search(
            r"(?i)\b((?:fly\s+back|return)\s+to\s+.+)",
            body,
        )
        drive_back = re.search(r"(?i)\b(drive\s+back\s+to\s+.+)", body)
        explicit_return = return_only or drive_back
        if explicit_return:
            values["transportation"] = _trim_clause(explicit_return.group(1))

    clauses = re.split(r"\n|(?<=[.!?])\s+(?=[A-Z*])", body)
    for clause in clauses:
        for field in ("breakfast", "lunch", "dinner"):
            if field in values:
                continue
            match = re.search(rf"(?i)\b{field}\s+at\s+(.+)", clause)
            if match:
                _append(values, field, _trim_clause(match.group(1)))
        if "accommodation" in values:
            accommodation_patterns: tuple[str, ...] = ()
        else:
            accommodation_patterns = (
                r"(?i)\bstay\s+at\s+(.+)",
                r"(?i)\bcheck\s+into\s+(.+)",
            )
        for pattern in accommodation_patterns:
            match = re.search(pattern, clause)
            if match:
                _append(values, "accommodation", _trim_clause(match.group(1)))
        if not labelled_attraction:
            activity_patterns = (
                r"(?i)\b(?:visit|tour|see)\s+(.+)",
                r"(?i)\bexplore\s+(the\s+.+)",
                r"(?i)\bexplore\s+([^,.;\n]+\s+and\s+[^,.;\n]+)",
                r"(?i)\b(?:take\s+a\s+walk\s+around|stroll\s+through)\s+(.+)",
            )
            for pattern in activity_patterns:
                for match in re.finditer(pattern, clause):
                    value = _trim_clause(match.group(1))
                    example = re.search(r"(?i)\(e\.g\.,?\s*(.*?)\)$", value)
                    _append(values, "attraction", example.group(1) if example else value)

    if "current_city" not in values:
        explicit_city_patterns = (
            r"(?i:\barrival\s+in\s+)([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,4})",
            r"(?i:\bfull\s+day\s+in\s+)([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,4})",
            r"\b([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,4})\s+(?i:exploration)\b",
        )
        for pattern in explicit_city_patterns:
            city = re.search(pattern, body)
            if city:
                city_text = re.split(
                    r"\.\s+(?=(?:Visit|Explore|Dinner|Lunch|Stay|Fly|Drive|Return)\b)",
                    city.group(1),
                    maxsplit=1,
                )[0]
                values["current_city"] = _clean(city_text.strip(" -*"))
                break


def _day_blocks(text: str) -> list[dict[str, str]] | None:
    matches = list(_DAY_MARKER.finditer(text))
    if not matches:
        return None
    result: list[dict[str, str]] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        for boundary in _DAY_RANGE_BOUNDARY.finditer(text, match.end(), end):
            end = min(end, boundary.start())
        if position + 1 == len(matches):
            for boundary in _GLOBAL_SECTION.finditer(text, match.end(), end):
                end = min(end, boundary.start())
            trailing_total = re.search(
                r"(?i)\btotal\s+(?:estimated|flight|driving|lodging|accommodation|cost)\b",
                text[match.end() : end],
            )
            if trailing_total:
                end = min(end, match.end() + trailing_total.start())
        body = text[match.end() : end].strip()
        values = _explicit_label_values(body)
        _extract_explicit_prose(body, values)
        values["day"] = match.group(1) or match.group(2)
        result.append(_normalise_explicit_row(values))
    return result


def adapt_final_answer_v2(answer: str | None) -> list[dict[str, str]] | None:
    """Extract only information explicitly present in a frozen final answer.

    This deterministic adapter receives no query, task identifier, tool output,
    database, reference answer, or evaluator verdict. It never fills an absent
    field and never substitutes an entity.
    """
    if not answer or not answer.strip():
        return None
    for parser in (_json_plan, _markdown_table, _day_blocks):
        plan = parser(answer)
        if plan:
            return plan
    return None


def validate_official_schema(plan: list[dict[str, Any]] | None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(plan, list) or not plan:
        return False, ["plan must be a non-empty list"]
    for index, row in enumerate(plan):
        if not isinstance(row, dict):
            errors.append(f"plan[{index}] must be an object")
            continue
        if tuple(row) != PLAN_FIELDS:
            errors.append(f"plan[{index}] fields/order differ from official contract")
        for field in PLAN_FIELDS:
            if not isinstance(row.get(field), str) or not row.get(field, "").strip():
                errors.append(f"plan[{index}].{field} must be a non-empty string")
    return not errors, errors
