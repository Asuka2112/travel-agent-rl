from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BLOCK_TRAINING_IMPORTS = """
import importlib.abc
import sys

class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'swift', 'peft'}:
            raise ModuleNotFoundError('blocked optional dependency: ' + fullname, name=fullname)

sys.meta_path.insert(0, RejectTrainingImports())
"""


@pytest.mark.parametrize(
    "code",
    [
        """
from travel_agent_rl.grpo import GRPOCheckpointManager, GRPOCheckpointState
assert GRPOCheckpointManager is not None and GRPOCheckpointState is not None
assert 'travel_agent_rl.grpo.scheduler' not in sys.modules
""",
        """
import importlib.util
spec = importlib.util.spec_from_file_location('pairwise', 'scripts/evaluate_pairwise_local_judge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scores = {name: 5 for name in module.PATH_DIMENSIONS}
scores.update({name: 1 for name in module.FINAL_DIMENSIONS})
assert abs(module.weighted_score(scores)['weighted'] - 0.44) < 1e-12
""",
        """
import importlib.util
spec = importlib.util.spec_from_file_location('agent_eval', 'scripts/evaluate_agent_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.build_full_replay_environment)
""",
        """
try:
    from travel_agent_rl.grpo import TravelAgentRolloutScheduler
except ModuleNotFoundError as exc:
    assert exc.name == 'swift'
else:
    raise AssertionError('scheduler must require the real ms-swift dependency')
""",
    ],
)
def test_optional_training_import_boundary(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", BLOCK_TRAINING_IMPORTS + code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
