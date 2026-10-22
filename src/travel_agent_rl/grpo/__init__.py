from importlib import import_module

from .checkpoint import GRPOCheckpointManager, GRPOCheckpointState

__all__ = [
    "GRPOCheckpointManager",
    "GRPOCheckpointState",
    "TravelAgentRolloutScheduler",
    "hydrate_request_tools",
    "validate_rollout_alignment",
]


def __getattr__(name: str):
    # Checkpoint metadata is dependency-light; scheduler exports require ms-swift.
    if name in __all__[2:]:
        value = getattr(import_module(".scheduler", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
