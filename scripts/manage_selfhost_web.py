#!/usr/bin/env python3
"""Manage the project-local SearXNG + Firecrawl v1.7 web stack."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
LOG_DIR = RUNTIME / "logs"
PID_DIR = RUNTIME / "pids"
STATE_DIR = RUNTIME / "state"


@dataclass(frozen=True)
class Service:
    name: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    port: int | None
    command_marker: str


def _secret() -> str:
    path = STATE_DIR / "searxng-secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
        path.chmod(0o600)
    return path.read_text(encoding="utf-8").strip()


def _services() -> list[Service]:
    node_bin = RUNTIME / "toolchains/node-v22.14.0-linux-x64/bin/node"
    firecrawl = RUNTIME / "sources/firecrawl-v1.7.0"
    api = firecrawl / "apps/api"
    playwright = firecrawl / "apps/playwright-service-ts"
    valkey = RUNTIME / "sources/valkey-8.1.3/src/valkey-server"
    searxng_python = RUNTIME / "venv-searxng311-runtime/bin/granian"
    common = {
        "REDIS_URL": "redis://127.0.0.1:3079",
        "REDIS_RATE_LIMIT_URL": "redis://127.0.0.1:3079",
        "PLAYWRIGHT_MICROSERVICE_URL": "http://127.0.0.1:3081/scrape",
        "USE_DB_AUTHENTICATION": "false",
        "ENV": "local",
        "SEARXNG_ENDPOINT": "http://127.0.0.1:3080",
        "SEARXNG_ENGINES": "bing",
        "SEARXNG_CATEGORIES": "general",
        "LOGGING_LEVEL": "info",
    }
    return [
        Service(
            "valkey",
            (
                str(valkey),
                "--bind",
                "127.0.0.1",
                "--port",
                "3079",
                "--protected-mode",
                "yes",
                "--dir",
                str(STATE_DIR / "valkey"),
                "--appendonly",
                "no",
                "--save",
                "",
            ),
            ROOT,
            {},
            3079,
            "valkey-server",
        ),
        Service(
            "searxng",
            (
                str(searxng_python),
                "--interface",
                "wsgi",
                "--host",
                "127.0.0.1",
                "--port",
                "3080",
                "searx.webapp:app",
            ),
            ROOT,
            {
                "PYTHONPATH": str(RUNTIME / "sources/searxng"),
                "SEARXNG_SETTINGS_PATH": str(ROOT / "deploy/web/searxng/settings.yml"),
                "SEARXNG_BIND_ADDRESS": "127.0.0.1",
                "SEARXNG_PORT": "3080",
                "SEARXNG_SECRET": _secret(),
            },
            3080,
            "searx.webapp:app",
        ),
        Service(
            "playwright",
            (str(node_bin), "dist/api.js"),
            playwright,
            {
                "HOST": "127.0.0.1",
                "PORT": "3081",
                "BLOCK_MEDIA": "true",
                "PLAYWRIGHT_BROWSERS_PATH": str(RUNTIME / "browsers"),
            },
            3081,
            "dist/api.js",
        ),
        Service(
            "firecrawl-api",
            (str(node_bin), "dist/src/index.js"),
            api,
            {**common, "HOST": "127.0.0.1", "PORT": "3002", "FLY_PROCESS_GROUP": "app"},
            3002,
            "dist/src/index.js",
        ),
        Service(
            "firecrawl-worker",
            (str(node_bin), "dist/src/services/queue-worker.js"),
            api,
            {**common, "FLY_PROCESS_GROUP": "worker"},
            None,
            "dist/src/services/queue-worker.js",
        ),
    ]


def _pid_path(service: Service) -> Path:
    return PID_DIR / f"{service.name}.pid"


def _read_pid(service: Service) -> int | None:
    try:
        return int(_pid_path(service).read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def _running(service: Service) -> bool:
    pid = _read_pid(service)
    return bool(pid and service.command_marker in _cmdline(pid))


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _validate_runtime(services: list[Service]) -> None:
    missing = [
        str(Path(service.command[0]))
        for service in services
        if not Path(service.command[0]).exists()
    ]
    if missing:
        raise SystemExit("missing self-host runtime; run bootstrap first:\n" + "\n".join(missing))


def start() -> None:
    services = _services()
    _validate_runtime(services)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "valkey").mkdir(parents=True, exist_ok=True)
    for service in services:
        if _running(service):
            continue
        if service.port is not None and _port_open(service.port):
            raise SystemExit(f"port {service.port} is occupied by an unmanaged process")
        environment = os.environ.copy()
        environment.update(service.environment)
        log = (LOG_DIR / f"{service.name}.log").open("ab", buffering=0)
        process = subprocess.Popen(
            service.command,
            cwd=service.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _pid_path(service).write_text(f"{process.pid}\n", encoding="utf-8")
        if service.port is not None:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not _port_open(service.port):
                if process.poll() is not None:
                    raise SystemExit(
                        f"{service.name} exited; inspect {LOG_DIR / (service.name + '.log')}"
                    )
                time.sleep(0.2)
            if not _port_open(service.port):
                raise SystemExit(f"{service.name} did not open port {service.port}")
        else:
            time.sleep(1)
            if process.poll() is not None:
                raise SystemExit(f"{service.name} exited; inspect its log")
    print(json.dumps(status(), indent=2, sort_keys=True))


def stop() -> None:
    for service in reversed(_services()):
        pid = _read_pid(service)
        if not pid:
            continue
        cmdline = _cmdline(pid)
        if service.command_marker not in cmdline:
            raise SystemExit(
                f"refusing to stop unverified pid {pid} for {service.name}: {cmdline!r}"
            )
        process_group = os.getpgid(pid)
        if process_group == pid:
            os.killpg(process_group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _cmdline(pid):
            time.sleep(0.2)
        if _cmdline(pid):
            if process_group == pid:
                os.killpg(process_group, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        _pid_path(service).unlink(missing_ok=True)
    print(json.dumps(status(), indent=2, sort_keys=True))


def status() -> dict[str, object]:
    result: dict[str, object] = {}
    for service in _services():
        pid = _read_pid(service)
        result[service.name] = {
            "pid": pid,
            "running": _running(service),
            "loopback_port_ready": service.port is None or _port_open(service.port),
            "port": service.port,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    args = parser.parse_args()
    if args.action == "start":
        start()
    elif args.action == "stop":
        stop()
    else:
        print(json.dumps(status(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
