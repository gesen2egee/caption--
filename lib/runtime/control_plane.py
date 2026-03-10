# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from lib.runtime.agent_surface import build_agent_surface, get_command_examples
from lib.runtime.reload_policy import build_reload_policy
from lib.runtime.service_reload import list_reloadable_runtime_services, reload_runtime_services
from lib.workers import get_worker_service_manager, worker_runtime_mode
from lib.workers.registry import get_registry, scan_workers


def get_runtime_bridge_status(host: Any) -> Dict[str, Any]:
    bridge = getattr(host, "_runtime_http_bridge", None)
    if bridge is None:
        return {
            "enabled": False,
            "host": None,
            "port": None,
            "url": None,
        }
    return bridge.status().__dict__


def get_runtime_workers(host: Any) -> Dict[str, Any]:
    registry = get_registry()
    workers: Dict[str, Any] = {}
    for category in sorted(getattr(registry, "_workers", {}).keys()):
        workers[category] = registry.get_workers(category)
    return {
        "categories": workers,
        "available_categories": [name for name, items in workers.items() if items],
        "runtime_mode": worker_runtime_mode(host._get_current_settings_obj()),
        "service_manager": get_worker_services_status(host),
    }


def command_list_reloadable_runtime_services(host: Any) -> Dict[str, Any]:
    services = list_reloadable_runtime_services()
    return {
        "services": services,
        "service_names": list(services.keys()),
        "service_count": len(services),
    }


def command_reload_runtime_services(host: Any, names: Optional[List[str]] = None) -> Dict[str, Any]:
    payload = reload_runtime_services(names=names)
    payload["service_names"] = list_reloadable_runtime_services()
    host._get_runtime_event_bus().emit("backend.services.reloaded", payload)
    return payload


def command_get_reload_policy(host: Any, changed_path: Optional[str] = None) -> Dict[str, Any]:
    return build_reload_policy(
        host._get_current_settings_obj(),
        changed_path=changed_path,
    )


def get_runtime_service_watch_status(host: Any) -> Dict[str, Any]:
    watcher = getattr(host, "_runtime_service_watcher", None)
    if watcher is None:
        return {
            "enabled": False,
            "interval_seconds": None,
            "service_count": len(list_reloadable_runtime_services()),
            "services": {},
            "last_reload": None,
            "last_error": None,
        }
    return watcher.status()


def command_start_runtime_service_watch(host: Any, interval_seconds: float = 1.0) -> Dict[str, Any]:
    return host._get_runtime_service_watcher().start(interval_seconds=interval_seconds)


def command_stop_runtime_service_watch(host: Any) -> Dict[str, Any]:
    return host._stop_runtime_service_watch()


def get_worker_services_status(host: Any) -> Dict[str, Any]:
    return get_worker_service_manager().status()


def command_reload_worker_services(
    host: Any,
    category: Optional[str] = None,
    worker_name: Optional[str] = None,
) -> Dict[str, Any]:
    payload = get_worker_service_manager().reload_service(category=category, worker_name=worker_name)
    host._get_runtime_event_bus().emit("workers.services.reloaded", payload)
    return payload


def command_stop_worker_services(
    host: Any,
    category: Optional[str] = None,
    worker_name: Optional[str] = None,
) -> Dict[str, Any]:
    payload = get_worker_service_manager().stop_services(category=category, worker_name=worker_name)
    host._get_runtime_event_bus().emit("workers.services.stopped", payload)
    return payload


def get_runtime_capabilities(host: Any, mode: Optional[str] = None) -> Dict[str, Any]:
    registry = host._get_command_registry()
    visible_commands = registry.list_commands(access_mode=mode)
    command_examples = get_command_examples()
    return {
        "commands": visible_commands,
        "command_access_modes": registry.get_command_access_modes(),
        "command_metadata": registry.get_command_metadata(access_mode=mode),
        "bridge": get_runtime_bridge_status(host),
        "workers": get_runtime_workers(host),
        "settings_schema_version": host.get_runtime_settings_schema().get("version"),
        "ui_spec_version": host.get_ui_spec().get("version"),
        "ui_spec_override_active": bool(host.get_ui_spec_override()),
        "control_ids": sorted(host.get_runtime_state().get("controls", {}).keys()),
        "spec_patch_supported": True,
        "command_result_tracking": True,
        "structured_errors": True,
        "event_streaming": True,
        "event_history_limit": host._get_runtime_event_bus().history_limit(),
        "event_filtering": True,
        "event_profiles": host.get_runtime_event_profiles(),
        "agent_modes": ["safe", "development"],
        "runtime_host_backend": getattr(host, "_runtime_host_backend", "qt-main-window"),
        "runtime_app_state_backend": "central-app-state",
        "runtime_state_source_of_truth": "app-state",
        "task_runtime_backend": getattr(host, "_runtime_task_backend_name", "python-thread"),
        "task_runtime_dispatcher": getattr(host, "_runtime_task_dispatcher_name", "qt-callback-dispatcher"),
        "task_runtime_listener_api": True,
        "task_runtime_cancel_events": True,
        "worker_runtime_mode": worker_runtime_mode(host._get_current_settings_obj()),
        "worker_error_taxonomy_available": True,
        "worker_service_reload_supported": True,
        "runtime_service_reload_supported": True,
        "reload_policy_available": True,
        "runtime_reloadable_services": list_reloadable_runtime_services(),
        "runtime_service_watch_supported": True,
        "runtime_service_watch_status": get_runtime_service_watch_status(host),
        "qt_residual_components": list(getattr(host, "_qt_residual_components", ["runtime.callback_dispatcher"])),
        "command_examples": {
            name: example
            for name, example in command_examples.items()
            if name in visible_commands
        },
    }


def get_agent_manifest(host: Any, mode: Optional[str] = None) -> Dict[str, Any]:
    bridge = get_runtime_bridge_status(host)
    base_url = str(bridge.get("url") or "").rstrip("/")
    control_profile = "control"
    surface = build_agent_surface(str(mode or "development"))
    routes = {
        "capabilities": f"/capabilities?mode={surface['mode_name']}",
        "agent_manifest": f"/agent/manifest?mode={surface['mode_name']}",
        "state": "/state",
        "ui_spec": "/ui-spec",
        "ui_spec_override": "/ui-spec-override",
        "events_history": f"/events?profile={control_profile}&limit=120",
        "events_stream": f"/events/stream?profile={control_profile}",
        "commands": f"/commands/{{name}}?mode={surface['mode_name']}",
        "preview_current": "/preview/current",
    }
    capabilities = get_runtime_capabilities(host, mode=surface["mode_name"])
    command_examples = capabilities.get("command_examples", {})
    command_metadata = capabilities.get("command_metadata", {})
    all_command_metadata = host._get_command_registry().get_command_metadata()
    return {
        "agent_mode": surface["mode_name"],
        "available_modes": ["safe", "development"],
        "default_mode": "development",
        "skills_note": surface["skills_note"],
        "recommended_event_profile": control_profile,
        "event_profiles": host.get_runtime_event_profiles(),
        "recommended_commands": surface["recommended_commands"],
        "allowed_commands": surface["allowed_commands"],
        "restricted_commands": surface["restricted_commands"],
        "command_examples": {
            name: example
            for name, example in command_examples.items()
            if name in surface["allowed_commands"]
        },
        "command_metadata": {
            name: metadata
            for name, metadata in command_metadata.items()
            if name in surface["allowed_commands"]
        },
        "restricted_command_metadata": {
            name: metadata
            for name, metadata in all_command_metadata.items()
            if name in surface["restricted_commands"]
        },
        "ui_mutation_allowed": surface["ui_mutation_allowed"],
        "settings_mutation_allowed": surface["settings_mutation_allowed"],
        "destructive_file_ops_allowed": surface["destructive_file_ops_allowed"],
        "bridge": bridge,
        "routes": routes,
        "endpoints": {
            name: (f"{base_url}{path}" if base_url else None)
            for name, path in routes.items()
        },
    }


def command_start_runtime_bridge(
    host: Any,
    bind_host: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    bind_host = bind_host or str(os.getenv("CAPTION_RUNTIME_HTTP_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    if port is None:
        port_raw = str(os.getenv("CAPTION_RUNTIME_HTTP_PORT", "8765")).strip() or "8765"
        try:
            port = int(port_raw)
        except ValueError:
            port = 8765
    return host._start_runtime_http_bridge(host=bind_host, port=int(port))


def command_stop_runtime_bridge(host: Any) -> Dict[str, Any]:
    return host._stop_runtime_http_bridge()


def command_scan_workers(host: Any) -> Dict[str, Any]:
    bus = host._get_runtime_event_bus()
    bus.emit("worker.scan.started", {})
    scan_workers()
    host.check_worker_availability()
    host._sync_runtime_settings_state()
    payload = get_runtime_workers(host)
    bus.emit("worker.scan.finished", payload)
    return payload
