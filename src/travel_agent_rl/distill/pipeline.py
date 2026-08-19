from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter

from travel_agent_rl.agent.protocol import AgentMessage, ToolCall, parse_agent_output
from travel_agent_rl.agent.state import AgentSession, SessionStatus

from .client import ChatBackend, TokenUsage
from .models import TrajectoryMetadata, TrajectoryRecord


class TrajectoryDistiller:
    def __init__(
        self,
        backend: ChatBackend,
        session_factory: Callable[[], AgentSession],
        system_prompt: str,
        *,
        prompt_version: str = "teacher-v1",
        max_model_calls: int = 15,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        self.backend = backend
        self.session_factory = session_factory
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.max_model_calls = max_model_calls
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(self, query_id: str, query: str) -> TrajectoryRecord:
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        session = self.session_factory()
        messages = [AgentMessage(role="user", content=query)]
        tool_calls: list[ToolCall] = []
        observations: list[dict[str, object]] = []
        exceptions: list[str] = []
        final_answer: str | None = None
        input_tokens = output_tokens = total_tokens = 0
        teacher_model = self.backend.model

        for _ in range(self.max_model_calls):
            try:
                result = self.backend.complete(messages, self.system_prompt)
                teacher_model = result.model
                input_tokens += result.usage.input_tokens
                output_tokens += result.usage.output_tokens
                total_tokens += result.usage.total_tokens
                messages.append(AgentMessage(role="assistant", content=result.content))
                try:
                    parsed = parse_agent_output(result.content)
                    if isinstance(parsed, ToolCall):
                        tool_calls.append(parsed)
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

                transition = session.step(result.content)
                if transition.observation is not None:
                    messages.append(AgentMessage(role="tool", content=transition.observation))
                    try:
                        observations.append(json.loads(transition.observation.splitlines()[0]))
                    except json.JSONDecodeError:
                        observations.append({"raw": transition.observation})
                if transition.status is SessionStatus.FINISHED:
                    final_answer = transition.final_answer
                    break
            except Exception as exc:
                exceptions.append(f"{type(exc).__name__}: {exc}")
                break
        else:
            exceptions.append("model_call_limit_exceeded")

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens or input_tokens + output_tokens,
        )
        cost = self._estimate_cost(usage)
        finished_at = datetime.now(timezone.utc)
        metadata = TrajectoryMetadata(
            query_id=query_id,
            teacher_model=teacher_model,
            prompt_version=self.prompt_version,
            started_at=started_at,
            finished_at=finished_at,
            number_of_turns=session.turns_used,
            tools_used=[call.name for call in tool_calls],
            success=bool(final_answer) and not exceptions,
            exceptions=exceptions,
            token_usage=usage,
            latency_seconds=perf_counter() - started,
            estimated_api_cost_usd=cost,
        )
        return TrajectoryRecord(
            query_id=query_id,
            query=query,
            messages=messages,
            tool_calls=tool_calls,
            tool_observations=observations,
            final_answer=final_answer,
            metadata=metadata,
        )

    def _estimate_cost(self, usage: TokenUsage) -> float | None:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            usage.input_tokens * self.input_cost_per_million
            + usage.output_tokens * self.output_cost_per_million
        ) / 1_000_000


class DistillationPipeline:
    """Concurrent JSONL runner with append-only checkpoints and resume support."""

    def __init__(self, distiller: TrajectoryDistiller, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.distiller = distiller
        self.max_workers = max_workers
        self._write_lock = Lock()

    def run(
        self,
        queries: Iterable[tuple[str, str]],
        output_path: Path,
        failed_path: Path,
    ) -> dict[str, int]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        completed = self._completed_ids(output_path)
        pending = [(query_id, query) for query_id, query in queries if query_id not in completed]
        stats = {"submitted": len(pending), "resumed": len(completed), "succeeded": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.distiller.generate, query_id, query): query_id
                for query_id, query in pending
            }
            for future in as_completed(futures):
                query_id = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    self._append_jsonl(
                        failed_path,
                        {"query_id": query_id, "error": f"{type(exc).__name__}: {exc}"},
                    )
                    stats["failed"] += 1
                    continue
                self._append_jsonl(output_path, record.model_dump(mode="json"))
                if record.metadata.success:
                    stats["succeeded"] += 1
                else:
                    self._append_jsonl(
                        failed_path,
                        {"query_id": query_id, "errors": record.metadata.exceptions},
                    )
                    stats["failed"] += 1
        return stats

    @staticmethod
    def _completed_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        completed: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    completed.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"invalid checkpoint line {line_number} in {path}") from exc
        return completed

    def _append_jsonl(self, path: Path, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._write_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
