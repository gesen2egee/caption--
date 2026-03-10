# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import traceback
from hashlib import sha1
from typing import Any, Dict, Optional, Tuple, Type

from lib.workers.base import BaseWorker, WorkerOutput
from lib.workers.errors import (
    WorkerNotFoundError,
    WorkerUnsupportedActionError,
    build_worker_error_info,
)
from lib.workers.registry import get_registry, scan_workers
from lib.workers.serialization import deserialize_worker_input, serialize_worker_output


FALLBACK_IMPORTS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("IMAGE_PROCESS", "image_flux2_klein_gguf_local"): (
        "lib.workers.image_flux2_klein_gguf_local",
        "ImageFlux2KleinGGUFLocalWorker",
    ),
    ("RESTORE", "image_restore_raw"): (
        "lib.workers.image_restore_raw",
        "ImageRestoreRawWorker",
    ),
}


def _resolve_worker_class(category: str, worker_name: str) -> Optional[Type[BaseWorker]]:
    registry = get_registry()
    scan_workers()
    worker_cls = registry.get_worker_class(category, worker_name)
    if worker_cls is not None:
        return worker_cls
    fallback = FALLBACK_IMPORTS.get((category, worker_name))
    if fallback is None:
        return None
    module_name, class_name = fallback
    module = importlib.import_module(module_name)
    worker_cls = getattr(module, class_name, None)
    if worker_cls is None:
        return None
    registry.add_worker(category, worker_name, worker_cls)
    return worker_cls


class WorkerServiceRuntime:
    def __init__(self) -> None:
        self._worker_key: Optional[Tuple[str, str, str]] = None
        self._worker_instance: Optional[BaseWorker] = None

    @staticmethod
    def _config_hash(config: Dict[str, Any]) -> str:
        raw = json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        return sha1(raw.encode("utf-8")).hexdigest()

    def _ensure_worker(self, category: str, worker_name: str, config: Dict[str, Any]) -> BaseWorker:
        key = (category, worker_name, self._config_hash(config))
        if self._worker_instance is not None and self._worker_key == key:
            return self._worker_instance
        worker_cls = _resolve_worker_class(category, worker_name)
        if worker_cls is None:
            raise WorkerNotFoundError(category, worker_name)
        self._worker_instance = worker_cls(config)
        self._worker_key = key
        return self._worker_instance

    def process(self, category: str, worker_name: str, config: Dict[str, Any], worker_input_payload: Dict[str, Any]) -> Dict[str, Any]:
        worker_input = deserialize_worker_input(worker_input_payload)
        with contextlib.redirect_stdout(sys.stderr):
            worker = self._ensure_worker(category, worker_name, config)
            output = worker.process(worker_input)
        return serialize_worker_output(output)

    def reload(self) -> None:
        self._worker_instance = None
        self._worker_key = None


def _write_response(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    runtime = WorkerServiceRuntime()
    pid = os.getpid()

    for raw_line in sys.stdin:
        line = str(raw_line or "").strip()
        if not line:
            continue
        request: Optional[Dict[str, Any]] = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            action = str(request.get("action", "process") or "process").strip().lower()

            if action == "shutdown":
                _write_response({"id": request_id, "ok": True, "result": {"pid": pid, "shutdown": True}})
                break

            if action == "reload":
                runtime.reload()
                _write_response({"id": request_id, "ok": True, "result": {"pid": pid, "reloaded": True}})
                continue

            if action != "process":
                raise WorkerUnsupportedActionError(action)

            result = runtime.process(
                category=str(request.get("category", "") or "").strip(),
                worker_name=str(request.get("worker_name", "") or "").strip(),
                config=dict(request.get("config", {}) or {}),
                worker_input_payload=dict(request.get("worker_input", {}) or {}),
            )
            _write_response({"id": request_id, "ok": True, "result": result, "service": {"pid": pid}})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _write_response(
                {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": str(exc),
                    "error_info": build_worker_error_info(
                        exc,
                        category=(request or {}).get("category") if isinstance(request, dict) else None,
                        worker_name=(request or {}).get("worker_name") if isinstance(request, dict) else None,
                        details={"service_pid": pid},
                    ),
                    "service": {"pid": pid},
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
