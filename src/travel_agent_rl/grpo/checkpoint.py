from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GRPOCheckpointState:
    optimizer_step: int
    config_sha256: str
    data_sha256: str
    trainer_checkpoint: str
    rollout_weight_version: int
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.optimizer_step < 0 or self.rollout_weight_version < 0:
            raise ValueError("checkpoint steps cannot be negative")


class GRPOCheckpointManager:
    """Atomic metadata checkpoint with immutable experiment identity checks."""

    def __init__(self, path: Path, *, config_sha256: str, data_sha256: str) -> None:
        self.path = path
        self.config_sha256 = config_sha256
        self.data_sha256 = data_sha256

    def load(self) -> GRPOCheckpointState | None:
        if not self.path.exists():
            return None
        state = GRPOCheckpointState(**json.loads(self.path.read_text(encoding="utf-8")))
        if state.config_sha256 != self.config_sha256:
            raise RuntimeError("checkpoint config hash does not match this run")
        if state.data_sha256 != self.data_sha256:
            raise RuntimeError("checkpoint data hash does not match this run")
        return state

    def save(self, state: GRPOCheckpointState) -> None:
        if state.config_sha256 != self.config_sha256:
            raise ValueError("refusing checkpoint with a different config hash")
        if state.data_sha256 != self.data_sha256:
            raise ValueError("refusing checkpoint with a different data hash")
        previous = self.load()
        if previous and state.optimizer_step < previous.optimizer_step:
            raise ValueError("refusing to regress an existing checkpoint")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)
