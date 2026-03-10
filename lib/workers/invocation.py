# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, Optional, Tuple, Type

from lib.core.dataclasses import Settings
from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.workers.errors import (
    WorkerError,
    WorkerNotFoundError,
    build_worker_error_info,
    build_worker_failure,
    normalize_worker_output,
)
from lib.workers.registry import get_registry, scan_workers
from lib.workers.service_manager import get_worker_service_manager


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


def resolve_worker_class(category: str, worker_name: str) -> Optional[Type[BaseWorker]]:
    category_name = str(category or "").strip().upper()
    worker_name = str(worker_name or "").strip()
    registry = get_registry()
    scan_workers()
    worker_cls = registry.get_worker_class(category_name, worker_name)
    if worker_cls is not None:
        return worker_cls
    fallback = FALLBACK_IMPORTS.get((category_name, worker_name))
    if fallback is None:
        return None
    module_name, class_name = fallback
    module = importlib.import_module(module_name)
    worker_cls = getattr(module, class_name, None)
    if worker_cls is not None:
        registry.add_worker(category_name, worker_name, worker_cls)
    return worker_cls


def worker_runtime_mode(settings: Optional[Settings]) -> str:
    if settings is not None:
        configured = str(getattr(settings, "worker_runtime_mode", "") or "").strip().lower()
        if configured:
            return configured
    return str(os.getenv("CAPTION_WORKER_RUNTIME", "inprocess") or "inprocess").strip().lower()


def invoke_worker(
    category: str,
    worker_name: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    worker_input: Optional[WorkerInput] = None,
    settings: Optional[Settings] = None,
) -> WorkerOutput:
    worker_input = worker_input or WorkerInput(settings=settings)
    config = dict(config or {})
    runtime_mode = worker_runtime_mode(settings)
    if runtime_mode == "service":
        timeout = float(getattr(settings, "worker_service_request_timeout", 1800) if settings is not None else 1800)
        python_executable = str(getattr(settings, "worker_service_python_exe", "") or "").strip() if settings is not None else ""
        try:
            output = get_worker_service_manager().invoke(
                category=category,
                worker_name=worker_name,
                config=config,
                worker_input=worker_input,
                timeout=timeout,
                python_executable=python_executable or sys.executable,
            )
        except WorkerError as exc:
            return build_worker_failure(
                str(exc),
                code=str(getattr(exc, "code", None) or "worker_error"),
                source=str(getattr(exc, "source", None) or "worker.service"),
                category=category,
                worker_name=worker_name,
                details=build_worker_error_info(exc).get("details"),
            )
        except Exception as exc:
            return build_worker_failure(
                str(exc) or f"worker invoke failed: {category}/{worker_name}",
                code="worker_invocation_failed",
                source="worker.invoke",
                category=category,
                worker_name=worker_name,
                details={"exception_type": exc.__class__.__name__},
            )
        return normalize_worker_output(output, category=category, worker_name=worker_name)

    worker_cls = resolve_worker_class(category, worker_name)
    if worker_cls is None:
        return build_worker_failure(
            f"Worker '{worker_name}' not found in category '{category}'",
            code="worker_not_found",
            source="worker.registry",
            category=category,
            worker_name=worker_name,
        )
    try:
        worker = worker_cls(config)
        output = worker.process(worker_input)
    except WorkerError as exc:
        return build_worker_failure(
            str(exc),
            code=str(getattr(exc, "code", None) or "worker_error"),
            source=str(getattr(exc, "source", None) or "worker"),
            category=category,
            worker_name=worker_name,
            details=build_worker_error_info(exc).get("details"),
        )
    except Exception as exc:
        return build_worker_failure(
            str(exc) or f"worker invoke failed: {category}/{worker_name}",
            code="worker_execution_failed",
            source="worker.invoke",
            category=category,
            worker_name=worker_name,
            details={"exception_type": exc.__class__.__name__},
        )
    return normalize_worker_output(output, category=category, worker_name=worker_name)
