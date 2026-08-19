from __future__ import annotations

from travel_agent_rl.data.select_train import constraint_types, select_queries


def record(index: int, level: str, *, constrained: bool = False) -> dict:
    return {
        "query_id": f"agent_star:{level}_{index}",
        "query": f"Plan trip number {index} from Origin {index} to Destination {index}.",
        "attributes": {
            "level": level,
            "org": f"Origin {index}",
            "dest": f"Destination {index}",
            "days": 3 + index % 3,
            "visiting_city_number": 1 + index % 2,
            "budget": 1000 + index,
            "local_constraint": {
                "house rule": "pets" if constrained else None,
                "room type": None,
                "cuisine": "Italian" if constrained else None,
                "transportation": None,
            },
        },
    }


def test_constraint_types_are_derived_without_inventing_labels() -> None:
    assert constraint_types(record(1, "medium", constrained=True)) == (
        "budget",
        "accommodation",
        "dietary",
        "multi_constraint",
    )


def test_selection_is_deterministic_and_stratified() -> None:
    candidates = [
        record(index, level, constrained=index % 2 == 0)
        for level in ("easy", "medium", "hard")
        for index in range(10)
    ]
    first, quotas = select_queries(candidates, size=12, seed=42, near_duplicate_threshold=100.0)
    second, _ = select_queries(candidates, size=12, seed=42, near_duplicate_threshold=100.0)
    assert quotas == {"easy": 4, "medium": 4, "hard": 4}
    assert [row["query_id"] for row in first] == [row["query_id"] for row in second]
