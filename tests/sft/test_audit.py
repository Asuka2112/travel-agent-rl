from travel_agent_rl.sft.audit import length_bucket, nearest_rank, select_label_audit_rows


def _row(query_id: str, difficulty: str, tool: str = "poi_search") -> dict:
    return {
        "query_id": query_id,
        "difficulty": difficulty,
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "tool_call", "content": f'{{"name":"{tool}","arguments":{{}}}}'},
            {"role": "assistant", "content": "done"},
        ],
    }


def test_length_bucket_boundaries() -> None:
    assert length_bucket(4096) == "<=4K"
    assert length_bucket(4097) == "4K-8K"
    assert length_bucket(32768) == "24K-32K"
    assert length_bucket(32769) == ">32K"


def test_nearest_rank() -> None:
    assert nearest_rank(range(1, 101), 95) == 95


def test_select_label_audit_rows_is_distinct_and_covers_requirements() -> None:
    rows = []
    for difficulty in ("easy", "medium", "hard"):
        rows.extend(_row(f"{difficulty}-{index}", difficulty) for index in range(3))
    rows.extend(_row(f"train-{index}", "hard", "train_search") for index in range(3))
    rows.extend(_row(f"web-{index}", "medium", "web_search") for index in range(3))
    selected = select_label_audit_rows(rows)
    assert len(selected) == 15
    assert len({row["query_id"] for row in selected}) == 15
