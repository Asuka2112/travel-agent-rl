#!/usr/bin/env python3
import json
from pathlib import Path

from travel_agent_rl.data.validate import validate_snapshot


if __name__ == "__main__":
    print(json.dumps(validate_snapshot(Path(".")), indent=2, sort_keys=True))
