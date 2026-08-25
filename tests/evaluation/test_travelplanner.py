from travel_agent_rl.evaluation.travelplanner import (
    PLAN_FIELDS,
    adapt_final_answer,
    is_structurally_complete,
    wilson_interval,
)


def test_adapt_json_plan_without_inference() -> None:
    answer = '{"plan":[{"day":1,"city":"Boston","breakfast":"Cafe (Boston)"}]}'
    plan = adapt_final_answer(answer)
    assert plan == [
        {
            "day": "1",
            "current_city": "Boston",
            "transportation": "-",
            "breakfast": "Cafe (Boston)",
            "attraction": "-",
            "lunch": "-",
            "dinner": "-",
            "accommodation": "-",
        }
    ]


def test_adapt_markdown_table() -> None:
    answer = """| Day | Current City | Transportation | Attraction |
|---|---|---|---|
| 1 | from Boston to Albany | Self-driving from Boston to Albany | Museum (Albany); |
| 2 | from Albany to Boston | Self-driving from Albany to Boston | Park (Albany); |
"""
    plan = adapt_final_answer(answer)
    assert plan is not None
    assert len(plan) == 2
    assert tuple(plan[0]) == PLAN_FIELDS
    assert plan[1]["current_city"] == "from Albany to Boston"
    assert is_structurally_complete(plan, 2)


def test_adapt_labelled_day_blocks() -> None:
    answer = """- **Day 1:** City: Boston; Breakfast: Cafe (Boston); Attraction: Common (Boston);
- **Jan 2 (Day 2)**: City: from Boston to Albany; Transportation: Taxi from Boston to Albany
"""
    plan = adapt_final_answer(answer)
    assert plan is not None
    assert [row["day"] for row in plan] == ["1", "2"]
    assert plan[0]["breakfast"] == "Cafe (Boston)"
    assert plan[1]["transportation"] == "Taxi from Boston to Albany"


def test_unlabelled_prose_is_not_invented_into_a_plan() -> None:
    assert adapt_final_answer("Visit Boston, then Albany, and have a pleasant trip.") is None


def test_wilson_interval_known_boundary() -> None:
    low, high = wilson_interval(0, 180)
    assert low == 0.0
    assert 0.020 < high < 0.021
