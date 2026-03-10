# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.runtime.service_reload import list_reloadable_runtime_services
from lib.workers import worker_runtime_mode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalized_rel_path(changed_path: Optional[str]) -> str:
    if not changed_path:
        return ""
    raw = str(changed_path or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    try:
        rel = candidate.relative_to(_repo_root()).as_posix()
    except Exception:
        rel = candidate.as_posix()
    return rel.lower()


def _runtime_service_matchers() -> Dict[str, str]:
    matchers: Dict[str, str] = {}
    for service_name, module_name in list_reloadable_runtime_services().items():
        matchers[service_name] = module_name.replace(".", "/").lower() + ".py"
    return matchers


def build_reload_policy(settings: Any, *, changed_path: Optional[str] = None) -> Dict[str, Any]:
    runtime_mode = worker_runtime_mode(settings)
    runtime_services = list_reloadable_runtime_services()
    normalized_path = _normalized_rel_path(changed_path)
    runtime_matchers = _runtime_service_matchers()

    rules: List[Dict[str, Any]] = [
        {
            "id": "runtime-service-module",
            "scope": "lib/runtime extracted services",
            "hot_reload_supported": True,
            "idle_required": True,
            "service_mode_action": "backend.reload_services",
            "inprocess_mode_action": "backend.reload_services",
            "notes": [
                "Extracted runtime modules are safe to reload through the runtime service catalog.",
                "If a task is running, the auto-reload watcher should defer until idle.",
            ],
        },
        {
            "id": "worker-implementation",
            "scope": "lib/workers worker implementation modules",
            "hot_reload_supported": runtime_mode == "service",
            "idle_required": True,
            "service_mode_action": "workers.services_reload",
            "inprocess_mode_action": "restart_host",
            "notes": [
                "In service mode, model workers are isolated and can be reloaded without restarting the host.",
                "In inprocess mode, deep worker code changes are not guaranteed hot-swappable.",
            ],
        },
        {
            "id": "worker-runtime-core",
            "scope": "worker service manager/process/invocation/runtime core",
            "hot_reload_supported": False,
            "idle_required": True,
            "service_mode_action": "restart_host",
            "inprocess_mode_action": "restart_host",
            "notes": [
                "Changes to worker runtime plumbing affect process lifecycle and should be treated as host-level restarts.",
            ],
        },
        {
            "id": "pipeline-task",
            "scope": "lib/pipeline/tasks task classes",
            "hot_reload_supported": False,
            "idle_required": True,
            "service_mode_action": "restart_host",
            "inprocess_mode_action": "restart_host",
            "notes": [
                "Pipeline task classes are imported into the host process and are not currently part of the safe runtime reload catalog.",
            ],
        },
        {
            "id": "frontend-shell",
            "scope": "frontend shell code",
            "hot_reload_supported": True,
            "idle_required": False,
            "service_mode_action": "frontend-dev-hmr-or-rebuild",
            "inprocess_mode_action": "frontend-dev-hmr-or-rebuild",
            "notes": [
                "Vite shell handles HMR in shell mode; built web mode still requires a frontend rebuild.",
            ],
        },
        {
            "id": "default",
            "scope": "unknown or uncategorized backend changes",
            "hot_reload_supported": False,
            "idle_required": True,
            "service_mode_action": "restart_host",
            "inprocess_mode_action": "restart_host",
            "notes": [
                "If a file does not match a known safe runtime surface, prefer restarting the host to avoid stale imports.",
            ],
        },
    ]

    recommendation = {
        "rule_id": "default",
        "hot_reload_supported": False,
        "recommended_action": "restart_host",
        "why": "No matching policy rule for the changed path.",
    }

    if normalized_path:
        matched_service = None
        for service_name, matcher in runtime_matchers.items():
            if normalized_path.endswith(matcher):
                matched_service = service_name
                break

        if matched_service is not None:
            recommendation = {
                "rule_id": "runtime-service-module",
                "hot_reload_supported": True,
                "recommended_action": "backend.reload_services",
                "service_names": [matched_service],
                "why": f"Matched extracted runtime service '{matched_service}'.",
            }
        elif normalized_path.startswith("lib/workers/"):
            core_names = {
                "lib/workers/service_manager.py",
                "lib/workers/service_process.py",
                "lib/workers/invocation.py",
                "lib/workers/registry.py",
                "lib/workers/reload_policy.py",
            }
            if normalized_path in core_names:
                recommendation = {
                    "rule_id": "worker-runtime-core",
                    "hot_reload_supported": False,
                    "recommended_action": "restart_host",
                    "why": "Matched worker runtime plumbing.",
                }
            else:
                recommendation = {
                    "rule_id": "worker-implementation",
                    "hot_reload_supported": runtime_mode == "service",
                    "recommended_action": "workers.services_reload" if runtime_mode == "service" else "restart_host",
                    "why": "Matched worker implementation module.",
                }
        elif normalized_path.startswith("lib/pipeline/tasks/"):
            recommendation = {
                "rule_id": "pipeline-task",
                "hot_reload_supported": False,
                "recommended_action": "restart_host",
                "why": "Matched pipeline task module.",
            }
        elif normalized_path.startswith("frontend/"):
            recommendation = {
                "rule_id": "frontend-shell",
                "hot_reload_supported": True,
                "recommended_action": "frontend-dev-hmr-or-rebuild",
                "why": "Matched frontend shell code.",
            }

    return {
        "worker_runtime_mode": runtime_mode,
        "runtime_reloadable_services": runtime_services,
        "changed_path": normalized_path or None,
        "recommendation": recommendation,
        "rules": rules,
    }
