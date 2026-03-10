# -*- coding: utf-8 -*-
"""
Runtime primitives for the refactor path.

This package provides UI-agnostic building blocks that can be reused by the
legacy PyQt UI, future React runtime, and Agent-facing automation hooks.
"""

from lib.runtime.callback_dispatcher import RuntimeCallbackDispatcher
from lib.runtime.app_state import RuntimeAppState
from lib.runtime.commands import CommandAccessError, CommandRegistry
from lib.runtime.events import (
    EventBus,
    RuntimeEvent,
    get_runtime_event_profiles,
    resolve_event_filters,
)
from lib.runtime.http_bridge import RuntimeHttpBridge, RuntimeHttpBridgeStatus
from lib.runtime.settings_schema import build_settings_schema
from lib.runtime.state import RuntimeStateStore
from lib.runtime.task_runner import TaskHandle, TaskRunner
from lib.runtime.service_reload import list_reloadable_runtime_services, reload_runtime_services
from lib.runtime.service_watch import RuntimeServiceWatcher
from lib.runtime.ui_spec import (
    RuntimeUiSpecStore,
    build_legacy_ui_spec,
    merge_ui_spec,
    patch_ui_spec_node,
    strip_runtime_state,
)

__all__ = [
    "CommandRegistry",
    "CommandAccessError",
    "RuntimeCallbackDispatcher",
    "RuntimeAppState",
    "EventBus",
    "RuntimeEvent",
    "get_runtime_event_profiles",
    "resolve_event_filters",
    "RuntimeHttpBridge",
    "RuntimeHttpBridgeStatus",
    "build_settings_schema",
    "RuntimeStateStore",
    "TaskHandle",
    "TaskRunner",
    "list_reloadable_runtime_services",
    "reload_runtime_services",
    "RuntimeServiceWatcher",
    "RuntimeUiSpecStore",
    "build_legacy_ui_spec",
    "merge_ui_spec",
    "patch_ui_spec_node",
    "strip_runtime_state",
]
