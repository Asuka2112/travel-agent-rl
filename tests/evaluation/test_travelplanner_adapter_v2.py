import json
from pathlib import Path

import pytest

from travel_agent_rl.evaluation.travelplanner_adapter_v2 import (
    PLAN_FIELDS,
    adapt_final_answer_v2,
    validate_official_schema,
)


OFFICIAL_EXAMPLE = (
    Path(__file__).parents[2] / "tests" / "fixtures" / "travelplanner_example_evaluation.jsonl"
)
PROJECT_TRAJECTORY = (
    Path(__file__).parents[2]
    / "tests"
    / "fixtures"
    / "project_non_travelplanner_trajectory.json"
)


def test_official_example_schema_round_trip() -> None:
    record = json.loads(OFFICIAL_EXAMPLE.read_text(encoding="utf-8").splitlines()[0])
    plan = adapt_final_answer_v2(json.dumps({"plan": record["plan"]}))
    assert plan is not None
    valid, errors = validate_official_schema(plan)
    assert valid, errors
    assert len(plan) == len(record["plan"])
    for source, parsed in zip(record["plan"], plan):
        assert parsed["transportation"] == source["transportation"]
        assert parsed["accommodation"] == source["accommodation"]


def test_synthetic_itinerary_extracts_all_explicit_fields() -> None:
    answer = """### Day 1
Current city: from Cedar Falls to Pine Harbor
Transportation: Train 18 from Cedar Falls to Pine Harbor
Breakfast: North Cafe, Cedar Falls
Lunch: Harbor Deli, Pine Harbor
Dinner: Juniper Table, Pine Harbor
Attraction: Maritime Museum, Pine Harbor;
Accommodation: Lantern Inn, Pine Harbor
"""
    plan = adapt_final_answer_v2(answer)
    assert plan == [
        {
            "day": "1",
            "current_city": "from Cedar Falls to Pine Harbor",
            "transportation": "Train 18 from Cedar Falls to Pine Harbor",
            "breakfast": "North Cafe, Cedar Falls",
            "attraction": "Maritime Museum, Pine Harbor",
            "lunch": "Harbor Deli, Pine Harbor",
            "dinner": "Juniper Table, Pine Harbor",
            "accommodation": "Lantern Inn, Pine Harbor",
        }
    ]
    assert tuple(plan[0]) == PLAN_FIELDS


def test_project_non_travelplanner_trajectory_parses_without_filling() -> None:
    answer = json.loads(PROJECT_TRAJECTORY.read_text(encoding="utf-8"))["final_answer"]
    plan = adapt_final_answer_v2(answer)
    assert plan is not None
    assert [row["day"] for row in plan] == ["1", "2"]
    assert all(tuple(row) == PLAN_FIELDS for row in plan)
    # The source does not explicitly provide TravelPlanner meal or hotel fields.
    assert all(row["breakfast"] == "-" for row in plan)
    assert all(row["accommodation"] == "-" for row in plan)


def test_missing_content_is_never_fabricated() -> None:
    plan = adapt_final_answer_v2("Day 1: Visit Small History Museum.")
    assert plan is not None
    assert plan[0]["attraction"] == "Small History Museum"
    assert all(
        plan[0][field] == "-"
        for field in ("current_city", "transportation", "breakfast", "lunch", "dinner", "accommodation")
    )


def test_depart_then_drive_uses_complete_explicit_cities() -> None:
    plan = adapt_final_answer_v2(
        "Day 1: Depart Alder Point. Drive to Birch Harbor (about 2.5 hours)."
    )
    assert plan is not None
    assert plan[0]["current_city"] == "from Alder Point to Birch Harbor"
    assert plan[0]["transportation"].startswith("Self-driving from Alder Point to Birch Harbor")


def test_summary_is_not_assigned_to_last_day() -> None:
    answer = """Day 1: Accommodation: No stay required.

### **Budget Summary**
Total Accommodation: $300
"""
    plan = adapt_final_answer_v2(answer)
    assert plan is not None
    assert plan[0]["accommodation"] == "No stay required."


def test_multiple_explicit_activities_are_retained() -> None:
    answer = """Day 1: Harbor Town Exploration
- Visit Maritime Museum.
- Explore the Botanical Garden.
- Stroll through the Sculpture Park.
"""
    plan = adapt_final_answer_v2(answer)
    assert plan is not None
    assert plan[0]["current_city"] == "Harbor Town"
    assert plan[0]["attraction"] == (
        "Maritime Museum; the Botanical Garden; the Sculpture Park"
    )


def test_json_missing_day_stays_absent() -> None:
    plan = adapt_final_answer_v2('{"plan":[{"attraction":"Clock Tower, Example City"}]}')
    assert plan is not None
    assert plan[0]["day"] == "-"


def test_unlabelled_plan_is_not_invented() -> None:
    assert adapt_final_answer_v2("Have a pleasant journey through several cities.") is None


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_answer_is_parser_failure(value: str | None) -> None:
    assert adapt_final_answer_v2(value) is None
