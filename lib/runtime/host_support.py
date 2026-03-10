# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from lib.runtime import (
    CommandRegistry,
    EventBus,
    RuntimeAppState,
    RuntimeHttpBridge,
    RuntimeServiceWatcher,
    RuntimeStateStore,
    TaskRunner,
    RuntimeUiSpecStore,
)
from lib.runtime.agent_surface import build_command_metadata, get_command_access_modes
from lib.runtime.state_adapter import build_tag_translations


def build_initial_runtime_state(host: Any) -> Dict[str, Any]:
    folder_meta = list(getattr(host, "top_tags", []) or [])
    custom_tags = list(getattr(host, "custom_tags", []) or [])
    tagger_tags = list(getattr(host, "tagger_tags", []) or [])
    nl_tags = list(getattr(host, "nl_pages", []) or [])
    return {
        "settings": dict(getattr(host, "settings", {}) or {}),
        "selection": {
            "root_dir_path": getattr(host, "root_dir_path", ""),
            "current_image_path": getattr(host, "current_image_path", ""),
            "current_index": getattr(host, "current_index", -1),
            "image_count": len(getattr(host, "image_files", []) or []),
            "has_raw_backup": False,
            "filter_active": bool(getattr(host, "filter_active", False)),
            "filter_query": "",
            "filter_tags": True,
            "filter_text": False,
        },
        "task": {
            "running": False,
            "task_name": None,
        },
        "commands": {
            "last_started": None,
            "last_finished": None,
            "last_failed": None,
        },
        "ui": {
            "current_view_mode": getattr(host, "current_view_mode", 0),
            "temp_view_mode": getattr(host, "temp_view_mode", None),
            "current_prompt_mode": getattr(host, "current_prompt_mode", "default"),
            "nl_page_index": getattr(host, "nl_page_index", 0),
            "nl_page_count": len(getattr(host, "nl_pages", []) or []),
            "current_tab_index": 0,
        },
        "controls": {
            "current_index": getattr(host, "current_index", -1) + 1,
            "filter_query": "",
            "filter_tags": True,
            "filter_text": False,
            "view_mode": getattr(host, "current_view_mode", 0),
            "tagger_save_to_txt": True,
            "llm_save_to_txt": True,
        },
        "content": {
            "prompt_text": "",
            "image_process_prompt_text": "",
            "txt_content": "",
            "nl_latest": getattr(host, "nl_latest", ""),
        },
        "tags": {
            "folder_meta": folder_meta,
            "custom": custom_tags,
            "tagger": tagger_tags,
            "nl": nl_tags,
            "translations": build_tag_translations(
                host,
                folder_meta,
                custom_tags,
                tagger_tags,
                nl_tags,
            ),
        },
    }


def get_runtime_event_bus(host: Any) -> EventBus:
    bus = getattr(host, "_runtime_event_bus", None)
    if bus is None:
        bus = EventBus()
        host._runtime_event_bus = bus
    return bus


def get_command_registry(host: Any) -> CommandRegistry:
    registry = getattr(host, "_command_registry", None)
    if registry is None:
        registry = CommandRegistry(event_bus=get_runtime_event_bus(host))
        host._command_registry = registry
        host._register_runtime_commands()
    return registry


def get_runtime_app_state(host: Any) -> RuntimeAppState:
    state = getattr(host, "_runtime_app_state", None)
    if state is None:
        state = RuntimeAppState(initial_state=build_initial_runtime_state(host))
        host._runtime_app_state = state
    return state


def get_runtime_state_store(host: Any) -> RuntimeStateStore:
    store = getattr(host, "_runtime_state_store", None)
    if store is None:
        store = RuntimeStateStore(
            get_runtime_event_bus(host),
            initial_state=get_runtime_app_state(host).snapshot(),
        )
        host._runtime_state_store = store
        subscribe_runtime_command_state(host)
    return store


def update_runtime_state_section(host: Any, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
    get_runtime_app_state(host).update_section(section, values)
    return get_runtime_state_store(host).update_section(section, values)


def replace_runtime_state_section(host: Any, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
    get_runtime_app_state(host).replace_section(section, values)
    return get_runtime_state_store(host).replace_section(section, values)


def get_task_runner(host: Any) -> TaskRunner:
    runner = getattr(host, "_task_runner", None)
    if runner is None:
        runner = TaskRunner(get_runtime_event_bus(host))
        host._task_runner = runner
    return runner


def subscribe_runtime_command_state(host: Any) -> None:
    if getattr(host, "_runtime_command_state_subscribed", False):
        return

    bus = get_runtime_event_bus(host)

    def on_started(event) -> None:
        update_runtime_state_section(
            host,
            "commands",
            {
                "last_started": {
                    "command_name": event.payload.get("command_name"),
                    "args": event.payload.get("args"),
                    "kwargs": event.payload.get("kwargs"),
                    "timestamp": event.timestamp,
                },
            },
        )

    def on_finished(event) -> None:
        update_runtime_state_section(
            host,
            "commands",
            {
                "last_finished": {
                    "command_name": event.payload.get("command_name"),
                    "result": event.payload.get("result"),
                    "timestamp": event.timestamp,
                },
            },
        )

    def on_failed(event) -> None:
        update_runtime_state_section(
            host,
            "commands",
            {
                "last_failed": {
                    "command_name": event.payload.get("command_name"),
                    "error": event.payload.get("error"),
                    "error_info": event.payload.get("error_info"),
                    "timestamp": event.timestamp,
                },
            },
        )

    bus.subscribe("command.started", on_started)
    bus.subscribe("command.finished", on_finished)
    bus.subscribe("command.failed", on_failed)
    host._runtime_command_state_subscribed = True


def get_runtime_ui_spec_store(host: Any) -> RuntimeUiSpecStore:
    store = getattr(host, "_runtime_ui_spec_store", None)
    if store is None:
        store = RuntimeUiSpecStore()
        host._runtime_ui_spec_store = store
    return store


def get_runtime_service_watcher(host: Any) -> RuntimeServiceWatcher:
    watcher = getattr(host, "_runtime_service_watcher", None)
    if watcher is None:
        watcher = RuntimeServiceWatcher(
            get_runtime_event_bus(host),
            can_reload=lambda: not host.is_task_running(),
            context_provider=lambda: host.get_task_status(),
        )
        host._runtime_service_watcher = watcher
    return watcher


def start_runtime_http_bridge(host: Any, bind_host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
    if getattr(host, "_runtime_http_bridge", None) is not None:
        return host.get_runtime_bridge_status()

    bind_host = str(bind_host or "127.0.0.1").strip() or "127.0.0.1"
    port = int(port or 8765)

    bridge = RuntimeHttpBridge(host, host=bind_host, port=port)
    bridge.start()
    host._runtime_http_bridge = bridge
    get_runtime_event_bus(host).emit("bridge.started", host.get_runtime_bridge_status())
    return host.get_runtime_bridge_status()


def maybe_start_runtime_http_bridge(host: Any) -> None:
    raw = str(os.getenv("CAPTION_RUNTIME_HTTP", "")).strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return

    bind_host = str(os.getenv("CAPTION_RUNTIME_HTTP_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port_raw = str(os.getenv("CAPTION_RUNTIME_HTTP_PORT", "8765")).strip() or "8765"
    try:
        port = int(port_raw)
    except ValueError:
        port = 8765

    start_runtime_http_bridge(host, bind_host=bind_host, port=port)


def maybe_start_runtime_service_watch(host: Any) -> None:
    raw = str(os.getenv("CAPTION_RUNTIME_AUTO_RELOAD_SERVICES", "")).strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return

    interval_raw = str(os.getenv("CAPTION_RUNTIME_AUTO_RELOAD_INTERVAL", "1.0")).strip() or "1.0"
    try:
        interval_seconds = float(interval_raw)
    except ValueError:
        interval_seconds = 1.0

    get_runtime_service_watcher(host).start(interval_seconds=interval_seconds)


def stop_runtime_http_bridge(host: Any) -> Dict[str, Any]:
    bridge = getattr(host, "_runtime_http_bridge", None)
    if bridge is None:
        return host.get_runtime_bridge_status()
    bridge.stop()
    host._runtime_http_bridge = None
    get_runtime_event_bus(host).emit("bridge.stopped", {"enabled": False})
    return host.get_runtime_bridge_status()


def stop_runtime_service_watch(host: Any) -> Dict[str, Any]:
    watcher = getattr(host, "_runtime_service_watcher", None)
    if watcher is None:
        return host.get_runtime_service_watch_status()
    return watcher.stop()


def register_runtime_command(host: Any, name: str, handler: Any) -> None:
    host._command_registry.register(
        name,
        handler,
        access_modes=get_command_access_modes(name),
        metadata=build_command_metadata(name),
    )
