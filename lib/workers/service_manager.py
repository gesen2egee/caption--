# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from lib.workers.base import WorkerInput, WorkerOutput
from lib.workers.errors import (
    WorkerServiceRemoteError,
    WorkerServiceTimeoutError,
    WorkerServiceUnavailableError,
)
from lib.workers.serialization import deserialize_worker_output, serialize_worker_input


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


class WorkerServiceClient:
    def __init__(
        self,
        category: str,
        worker_name: str,
        *,
        python_executable: Optional[str] = None,
    ) -> None:
        self.category = str(category or "").strip().upper()
        self.worker_name = str(worker_name or "").strip()
        self.python_executable = str(python_executable or sys.executable or "python").strip()
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._stderr_lines: Deque[str] = deque(maxlen=120)
        self._stderr_thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._request_count = 0

    def _spawn(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [self.python_executable, "-u", "-m", "lib.workers.service_process"],
            cwd=_repo_root(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._started_at = time.time()
        self._request_count = 0
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True, name=f"worker-stderr-{self.worker_name}")
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_lines.append(str(line).rstrip())

    def _communicate(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        self._spawn()
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise WorkerServiceUnavailableError(self.category, self.worker_name)
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

        deadline = time.time() + max(float(timeout or 0.0), 1.0)
        while time.time() < deadline:
            line = process.stdout.readline()
            if line:
                response = json.loads(line)
                return dict(response or {})
            if process.poll() is not None:
                break
            time.sleep(0.01)
        raise WorkerServiceTimeoutError(
            self.category,
            self.worker_name,
            timeout=timeout,
            stderr_tail=list(self._stderr_lines)[-8:],
        )

    def process(self, config: Dict[str, Any], worker_input: WorkerInput, *, timeout: float) -> WorkerOutput:
        payload = {
            "id": str(uuid.uuid4()),
            "action": "process",
            "category": self.category,
            "worker_name": self.worker_name,
            "config": dict(config or {}),
            "worker_input": serialize_worker_input(worker_input),
        }
        with self._lock:
            try:
                response = self._communicate(payload, timeout)
            except Exception:
                self.stop()
                self._spawn()
                response = self._communicate(payload, timeout)
            self._request_count += 1
        if not response.get("ok"):
            raise WorkerServiceRemoteError(
                self.category,
                self.worker_name,
                str(response.get("error") or f"worker service failed: {self.category}/{self.worker_name}"),
                error_info=dict(response.get("error_info", {}) or {}) or None,
                stderr_tail=list(self._stderr_lines)[-8:],
            )
        return deserialize_worker_output(dict(response.get("result", {}) or {}))

    def reload(self) -> Dict[str, Any]:
        with self._lock:
            response = self._communicate(
                {
                    "id": str(uuid.uuid4()),
                    "action": "reload",
                },
                timeout=30.0,
            )
        return {
            "ok": bool(response.get("ok", False)),
            "category": self.category,
            "worker_name": self.worker_name,
            "pid": self.pid,
        }

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.stdout is not None:
                process.stdin.write(json.dumps({"id": str(uuid.uuid4()), "action": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.wait(timeout=2.0)
        except Exception:
            process.kill()
        self._started_at = None

    @property
    def pid(self) -> Optional[int]:
        if self._process is None or self._process.poll() is not None:
            return None
        return int(self._process.pid)

    def status(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "worker_name": self.worker_name,
            "pid": self.pid,
            "alive": bool(self.pid),
            "started_at": self._started_at,
            "uptime_seconds": max(time.time() - self._started_at, 0.0) if self._started_at else 0.0,
            "request_count": self._request_count,
            "stderr_tail": list(self._stderr_lines)[-8:],
            "python_executable": self.python_executable,
        }


class WorkerServiceManager:
    def __init__(self) -> None:
        self._services: Dict[Tuple[str, str], WorkerServiceClient] = {}
        self._lock = threading.Lock()

    def _key(self, category: str, worker_name: str) -> Tuple[str, str]:
        return (str(category or "").strip().upper(), str(worker_name or "").strip())

    def _client(self, category: str, worker_name: str, python_executable: Optional[str] = None) -> WorkerServiceClient:
        key = self._key(category, worker_name)
        with self._lock:
            client = self._services.get(key)
            if client is None:
                client = WorkerServiceClient(category, worker_name, python_executable=python_executable)
                self._services[key] = client
            return client

    def invoke(
        self,
        category: str,
        worker_name: str,
        *,
        config: Dict[str, Any],
        worker_input: WorkerInput,
        timeout: float,
        python_executable: Optional[str] = None,
    ) -> WorkerOutput:
        client = self._client(category, worker_name, python_executable=python_executable)
        return client.process(config, worker_input, timeout=timeout)

    def reload_service(
        self,
        category: Optional[str] = None,
        worker_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            items = list(self._services.items())
        reloaded = []
        for (service_category, service_worker_name), client in items:
            if category and service_category != str(category).strip().upper():
                continue
            if worker_name and service_worker_name != str(worker_name).strip():
                continue
            reloaded.append(client.reload())
        return {
            "reloaded_count": len(reloaded),
            "services": reloaded,
        }

    def stop_services(
        self,
        category: Optional[str] = None,
        worker_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            items = list(self._services.items())
        stopped = []
        for key, client in items:
            service_category, service_worker_name = key
            if category and service_category != str(category).strip().upper():
                continue
            if worker_name and service_worker_name != str(worker_name).strip():
                continue
            client.stop()
            stopped.append({"category": service_category, "worker_name": service_worker_name})
            with self._lock:
                self._services.pop(key, None)
        return {
            "stopped_count": len(stopped),
            "services": stopped,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._services.items())
        services = [client.status() for _, client in items]
        return {
            "service_count": len(services),
            "services": services,
        }


_manager = WorkerServiceManager()


def get_worker_service_manager() -> WorkerServiceManager:
    return _manager
