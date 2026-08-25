from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/analyze_travelplanner_failure_decomposition.py"
SPEC = importlib.util.spec_from_file_location("failure_decomposition", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parser_failure_is_recorded_without_inventing_plan() -> None:
    findings = MODULE.schema_findings(
        {"final_answer": "A prose itinerary", "official_plan": None},
        {"days": 3},
    )
    assert findings == [
        {
            "type": "parsing_failure",
            "evidence": "final answer present but deterministic adapter produced null plan",
        }
    ]


def test_attraction_only_plan_records_missing_fields() -> None:
    plan = [
        {
            "day": "1",
            "current_city": "-",
            "transportation": "-",
            "breakfast": "-",
            "attraction": "whole day prose",
            "lunch": "-",
            "dinner": "-",
            "accommodation": "-",
        }
    ]
    findings = MODULE.schema_findings(
        {"final_answer": "Day 1: whole day prose", "official_plan": plan},
        {"days": 1},
    )
    types = [finding["type"] for finding in findings]
    assert "missing_required_field" in types
    assert "malformed_itinerary_structure" in types


def test_sandbox_rejection_is_not_called_database_mismatch_when_schema_is_bad() -> None:
    record = {
        "official_plan": [{"attraction": "Drive to Albany, then visit a museum."}],
        "messages": [],
    }
    adapter, environment = MODULE.environment_findings(
        record,
        "The attraction Drive to Albany, then visit a museum in day 1 is invalid in the sandbox.",
        [{"type": "missing_required_field", "evidence": "current_city=1"}],
    )
    assert environment == []
    assert adapter[0]["type"] == "entity_validation_confounded_by_adapter"
