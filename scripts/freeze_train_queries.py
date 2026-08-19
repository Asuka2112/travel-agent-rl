#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from travel_agent_rl.data.select_train import freeze_training_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/train_selection_1835.yaml")
    )
    args = parser.parse_args()
    manifest = freeze_training_selection(args.root.resolve(), args.root / args.config)
    summary = {
        "selection_size": manifest["selection_size"],
        "difficulty_quotas": manifest["difficulty_quotas"],
        "output_sha256": manifest["output_sha256"],
        "pilot_50": manifest["pilot_50"],
        "pilot_200": manifest["pilot_200"],
        "contamination": manifest["contamination"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
