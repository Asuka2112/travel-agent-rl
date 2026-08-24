from __future__ import annotations

import pytest

from travel_agent_rl.grpo import GRPOCheckpointManager, GRPOCheckpointState


def state(step: int) -> GRPOCheckpointState:
    return GRPOCheckpointState(
        optimizer_step=step,
        config_sha256="config",
        data_sha256="data",
        trainer_checkpoint=f"checkpoint-{step}",
        rollout_weight_version=step,
        metrics={"loss": 1.0 / (step + 1)},
    )


def test_atomic_checkpoint_resume_continues_without_repeating(tmp_path) -> None:
    path = tmp_path / "state.json"
    first_process = GRPOCheckpointManager(path, config_sha256="config", data_sha256="data")
    first_process.save(state(2))
    resumed_process = GRPOCheckpointManager(path, config_sha256="config", data_sha256="data")
    loaded = resumed_process.load()
    assert loaded is not None and loaded.optimizer_step == 2
    resumed_process.save(state(5))
    assert resumed_process.load().optimizer_step == 5


def test_checkpoint_rejects_identity_mismatch_and_step_regression(tmp_path) -> None:
    path = tmp_path / "state.json"
    manager = GRPOCheckpointManager(path, config_sha256="config", data_sha256="data")
    manager.save(state(3))
    with pytest.raises(ValueError, match="regress"):
        manager.save(state(2))
    with pytest.raises(RuntimeError, match="config hash"):
        GRPOCheckpointManager(path, config_sha256="different", data_sha256="data").load()
