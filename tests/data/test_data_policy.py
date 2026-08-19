from pathlib import Path

import yaml


def test_official_test_split_is_disabled_for_data_pipeline() -> None:
    config = yaml.safe_load(Path("configs/data/default.yaml").read_text())
    official_test = config["sources"]["travelplanner_official_test"]
    assert official_test["enabled"] is False
    assert official_test["policy"] == "evaluation_only"
    assert config["frozen_test_minimum"] >= 80
