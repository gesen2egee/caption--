# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from lib.core.settings import save_app_settings
from lib.runtime import (
    build_legacy_ui_spec,
    build_settings_schema,
    get_runtime_event_profiles,
    patch_ui_spec_node,
    resolve_event_filters,
    strip_runtime_state,
)
import lib.runtime.task_orchestrator as task_orchestrator


def get_runtime_events(
    host: Any,
    limit: Optional[int] = None,
    profile: Optional[str] = None,
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    _, resolved_include, resolved_exclude = resolve_event_filters(
        profile=profile,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
    )
    return [
        {
            "name": event.name,
            "payload": event.payload,
            "timestamp": event.timestamp,
        }
        for event in host._get_runtime_event_bus().history(
            limit=limit,
            include_prefixes=resolved_include,
            exclude_prefixes=resolved_exclude,
        )
    ]


def get_runtime_event_profiles_api(host: Any) -> Dict[str, Any]:
    return get_runtime_event_profiles()


def get_runtime_state(host: Any) -> Dict[str, Any]:
    return host._get_runtime_app_state().snapshot()


def get_ui_spec(host: Any) -> Dict[str, Any]:
    base_spec = build_legacy_ui_spec(host.get_runtime_state())
    return host._get_runtime_ui_spec_store().apply(base_spec)


def get_ui_spec_override(host: Any) -> Dict[str, Any]:
    return host._get_runtime_ui_spec_store().override_snapshot()


def emit_ui_spec_updated(host: Any) -> Dict[str, Any]:
    payload = {
        "override": host.get_ui_spec_override(),
        "spec": host.get_ui_spec(),
    }
    host._get_runtime_event_bus().emit("ui.spec.updated", payload)
    return payload["spec"]


def update_ui_spec(host: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    host._get_runtime_ui_spec_store().merge_override(patch)
    return host._emit_ui_spec_updated()


def replace_ui_spec_override(host: Any, override: Dict[str, Any]) -> Dict[str, Any]:
    host._get_runtime_ui_spec_store().replace_override(override)
    return host._emit_ui_spec_updated()


def patch_ui_node(host: Any, node_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("node patch must be a dict")
    current_spec = host.get_ui_spec()
    if not patch_ui_spec_node(current_spec, str(node_id or "").strip(), patch):
        raise ValueError(f"ui spec node not found: {node_id}")
    host._get_runtime_ui_spec_store().replace_override(strip_runtime_state(current_spec))
    return host._emit_ui_spec_updated()


def reset_ui_spec(host: Any) -> Dict[str, Any]:
    host._get_runtime_ui_spec_store().reset_override()
    return host._emit_ui_spec_updated()


def command_shutdown_app(host: Any, confirm: bool = False) -> Dict[str, Any]:
    if not confirm:
        return {
            "scheduled": False,
            "reason": "confirmation_required",
        }

    host._get_runtime_event_bus().emit(
        "app.shutdown_requested",
        {
            "bridge": host.get_runtime_bridge_status(),
        },
    )

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    def _shutdown() -> None:
        try:
            host.close()
        finally:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    QTimer.singleShot(0, _shutdown)
    return {
        "scheduled": True,
    }


def runtime_settings_dict(host: Any) -> Dict[str, Any]:
    values = host._get_runtime_app_state().get_section("settings")
    if values:
        return values
    return dict(getattr(host, "settings", {}) or {})


def replace_runtime_settings_dict(host: Any, new_cfg: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
    normalized = dict(new_cfg or {})
    host.settings = normalized
    host._replace_runtime_state_section("settings", normalized)
    if persist:
        save_app_settings(normalized)
    return normalized


def get_runtime_settings(host: Any) -> Dict[str, Any]:
    return host._runtime_settings_dict()


def get_runtime_settings_schema(host: Any) -> Dict[str, Any]:
    return build_settings_schema(host.get_runtime_settings())


def update_runtime_settings(host: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("settings patch must be a dict")

    new_cfg = host._runtime_settings_dict()
    new_cfg.update(patch)

    if hasattr(host, "apply_runtime_settings"):
        host.apply_runtime_settings(new_cfg)
    else:
        host._replace_runtime_settings_dict(new_cfg, persist=True)

    return host._runtime_settings_dict()


def get_task_status(host: Any) -> Dict[str, Any]:
    return task_orchestrator.get_task_status(
        getattr(host, "_current_task", None),
        host.is_task_running,
    )


def execute_command(host: Any, command_name: str, *args, access_mode: str = "internal", **kwargs) -> Any:
    return host._get_command_registry().execute(
        command_name,
        *args,
        access_mode=access_mode,
        **kwargs,
    )
