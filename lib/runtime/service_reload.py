# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Dict, Iterable


RELOADABLE_RUNTIME_SERVICES = {
    "selection": "lib.runtime.selection_service",
    "editor": "lib.runtime.editor_service",
    "batch": "lib.runtime.batch_service",
    "processing": "lib.runtime.processing_service",
    "command_catalog": "lib.runtime.command_catalog",
    "command_actions": "lib.runtime.command_actions",
    "control_plane": "lib.runtime.control_plane",
    "host_support": "lib.runtime.host_support",
    "pipeline_callbacks": "lib.runtime.pipeline_callbacks",
    "reload_policy": "lib.runtime.reload_policy",
    "runtime_api": "lib.runtime.runtime_api",
    "runtime_regression": "lib.runtime.runtime_regression",
    "state_adapter": "lib.runtime.state_adapter",
    "state_projection": "lib.runtime.state_projection",
    "task": "lib.runtime.task_service",
    "task_facade": "lib.runtime.task_facade",
    "task_orchestrator": "lib.runtime.task_orchestrator",
}


def list_reloadable_runtime_services() -> Dict[str, str]:
    return dict(RELOADABLE_RUNTIME_SERVICES)


def reload_runtime_services(names: Iterable[str] | None = None) -> Dict[str, Any]:
    requested_names = [str(name or "").strip().lower() for name in (names or RELOADABLE_RUNTIME_SERVICES.keys())]
    requested_names = [name for name in requested_names if name]

    reloaded: list[dict[str, str]] = []
    missing: list[str] = []
    errors: list[dict[str, str]] = []

    for name in requested_names:
        module_name = RELOADABLE_RUNTIME_SERVICES.get(name)
        if module_name is None:
            missing.append(name)
            continue
        try:
            module = importlib.import_module(module_name)
            reloaded_module: ModuleType = importlib.reload(module)
            reloaded.append(
                {
                    "name": name,
                    "module": reloaded_module.__name__,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "name": name,
                    "module": module_name,
                    "error": str(exc),
                }
            )

    return {
        "requested_names": requested_names,
        "reloaded": reloaded,
        "reloaded_count": len(reloaded),
        "missing": missing,
        "missing_count": len(missing),
        "errors": errors,
        "error_count": len(errors),
    }
