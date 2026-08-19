from __future__ import annotations

import json
import os
import re
import hashlib
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
        teacher_revision: str | None = None,
        snapshot_version: int | None = None,
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
        self.teacher_revision = teacher_revision
        self.snapshot_version = snapshot_version
        self.max_model_calls = max_model_calls
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def generate(
        self,
        query_id: str,
        query: str,
        query_metadata: dict[str, object] | None = None,
    ) -> TrajectoryRecord:
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        session = self.session_factory()
        messages = [AgentMessage(role="user", content=query)]
        recorded_messages = list(messages)
        tool_calls: list[ToolCall] = []
        observations: list[dict[str, object]] = []
        exceptions: list[str] = []
        protocol_errors: list[str] = []
        final_answer: str | None = None
        input_tokens = output_tokens = total_tokens = 0
        teacher_model = self.backend.model

        model_calls = 0
        for model_call in range(1, self.max_model_calls + 1):
            try:
                result = self.backend.complete(messages, self.system_prompt)
                model_calls = model_call
                teacher_model = result.model
                input_tokens += result.usage.input_tokens
                output_tokens += result.usage.output_tokens
                total_tokens += result.usage.total_tokens
                assistant_message = AgentMessage(role="assistant", content=result.content)
                try:
                    parsed = parse_agent_output(result.content)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    protocol_errors.append(
                        f"{type(exc).__name__}: {exc}; raw={result.content[:2000]!r}"
                    )
                    messages.append(assistant_message)
                    messages.append(
                        AgentMessage(
                            role="tool",
                            content=json.dumps(
                                {
                                    "status": "error",
                                    "error": {
                                        "code": "invalid_agent_output",
                                        "message": (
                                            "Discard that malformed response. Return exactly one "
                                            "complete <tool-call>...</tool-call> or "
                                            "<answer>...</answer> block with no outside text."
                                        ),
                                    },
                                },
                                sort_keys=True,
                            ),
                        )
                    )
                    continue
                messages.append(assistant_message)
                recorded_messages.append(assistant_message)
                if isinstance(parsed, ToolCall):
                    tool_calls.append(parsed)

                transition = session.step(result.content)
                if transition.observation is not None:
                    tool_message = AgentMessage(role="tool", content=transition.observation)
                    messages.append(tool_message)
                    recorded_messages.append(tool_message)
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
            difficulty=(
                str(query_metadata["difficulty"])
                if query_metadata and query_metadata.get("difficulty")
                else None
            ),
            teacher_model=teacher_model,
            teacher_revision=self.teacher_revision,
            prompt_version=self.prompt_version,
            snapshot_version=self.snapshot_version,
            started_at=started_at,
            finished_at=finished_at,
            number_of_turns=session.turns_used,
            number_of_model_calls=model_calls,
            tools_used=[call.name for call in tool_calls],
            tool_sequence=[call.name for call in tool_calls],
            success=bool(final_answer) and not exceptions,
            exceptions=exceptions,
            protocol_errors=protocol_errors,
            token_usage=usage,
            latency_seconds=perf_counter() - started,
            estimated_api_cost_usd=cost,
            source_query_metadata=query_metadata or {},
        )
        return TrajectoryRecord(
            query_id=query_id,
            query=query,
            messages=recorded_messages,
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
        queries: Iterable[tuple[str, str] | tuple[str, str, dict[str, object]]],
        output_path: Path,
        failed_path: Path,
        *,
        raw_dir: Path | None = None,
        index_path: Path | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = 50,
    ) -> dict[str, int]:
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        query_rows = [self._normalize_query(row) for row in queries]
        query_ids = {query_id for query_id, _, _ in query_rows}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            completed = self._completed_raw_ids(raw_dir) & query_ids
            self._rebuild_outputs(raw_dir, output_path, index_path)
        else:
            completed = self._completed_ids(output_path)
        pending = [
            (query_id, query, metadata)
            for query_id, query, metadata in query_rows
            if query_id not in completed
        ]
        stats = {"submitted": len(pending), "resumed": len(completed), "succeeded": 0, "failed": 0}
        processed_since_start = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.distiller.generate, query_id, query, metadata): query_id
                for query_id, query, metadata in pending
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
                else:
                    if record.metadata.success:
                        payload = record.model_dump(mode="json")
                        if raw_dir is None:
                            self._append_jsonl(output_path, payload)
                        else:
                            self._write_raw(raw_dir, payload)
                        stats["succeeded"] += 1
                    else:
                        self._append_jsonl(
                            failed_path,
                            record.model_dump(mode="json"),
                        )
                        stats["failed"] += 1
                processed_since_start += 1
                if checkpoint_path is not None and processed_since_start % checkpoint_every == 0:
                    if raw_dir is not None:
                        self._rebuild_outputs(raw_dir, output_path, index_path)
                    self._write_checkpoint(checkpoint_path, stats)
        if raw_dir is not None:
            self._rebuild_outputs(raw_dir, output_path, index_path)
        if checkpoint_path is not None:
            self._write_checkpoint(checkpoint_path, stats)
        return stats

    @staticmethod
    def _normalize_query(
        row: tuple[str, str] | tuple[str, str, dict[str, object]],
    ) -> tuple[str, str, dict[str, object]]:
        if len(row) == 2:
            query_id, query = row
            return query_id, query, {}
        query_id, query, metadata = row
        return query_id, query, metadata

    @staticmethod
    def _raw_filename(query_id: str) -> str:
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id).strip("._")[:80] or "query"
        digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:12]
        return f"{readable}__{digest}.json"

    @classmethod
    def _completed_raw_ids(cls, raw_dir: Path) -> set[str]:
        completed: set[str] = set()
        for path in sorted(raw_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("metadata", {}).get("success") is True:
                    completed.add(row["query_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid immutable raw trajectory: {path}") from exc
        return completed

    def _write_raw(self, raw_dir: Path, payload: dict[str, object]) -> None:
        path = raw_dir / self._raw_filename(str(payload["query_id"]))
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        with self._write_lock:
            if path.exists():
                if path.read_bytes() != encoded:
                    raise ValueError(f"refusing to overwrite immutable raw trajectory: {path}")
                return
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)

    @classmethod
    def _rebuild_outputs(
        cls,
        raw_dir: Path,
        output_path: Path,
        index_path: Path | None,
    ) -> None:
        records: list[tuple[Path, dict[str, object]]] = []
        for path in sorted(raw_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid immutable raw trajectory: {path}") from exc
            records.append((path, record))
        cls._atomic_jsonl(output_path, [record for _, record in records])
        if index_path is not None:
            rows = [
                {
                    "query_id": record["query_id"],
                    "raw_path": str(path.relative_to(raw_dir.parent)),
                    "success": record.get("metadata", {}).get("success"),
                    "difficulty": record.get("metadata", {}).get("difficulty"),
                    "teacher_model": record.get("metadata", {}).get("teacher_model"),
                    "teacher_revision": record.get("metadata", {}).get("teacher_revision"),
                    "snapshot_version": record.get("metadata", {}).get("snapshot_version"),
                    "number_of_turns": record.get("metadata", {}).get("number_of_turns"),
                    "token_usage": record.get("metadata", {}).get("token_usage", {}),
                    "exceptions": record.get("metadata", {}).get("exceptions", []),
                    "latency_seconds": record.get("metadata", {}).get("latency_seconds"),
                    "tools_used": record.get("metadata", {}).get("tools_used", []),
                }
                for path, record in records
            ]
            cls._atomic_jsonl(index_path, rows)

    @staticmethod
    def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _write_checkpoint(path: Path, stats: dict[str, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **stats,
            "processed_this_run": stats["succeeded"] + stats["failed"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

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
                    row = json.loads(line)
                    if row.get("metadata", {}).get("success") is True:
                        completed.add(row["query_id"])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"invalid checkpoint line {line_number} in {path}") from exc
        return completed

    def _append_jsonl(self, path: Path, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._write_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
