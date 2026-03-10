# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any, Dict, Optional

from lib.runtime.events import EventBus


class RuntimeStateStore:
    """
    Small in-process state store for the migration path.

    It keeps a serializable snapshot that can later be consumed by a web UI,
    Agent tooling, tests, or logs without depending on widget instances.
    """

    def __init__(self, event_bus: EventBus, initial_state: Optional[Dict[str, Any]] = None):
        self._event_bus = event_bus
        self._state: Dict[str, Any] = deepcopy(initial_state or {})
        self._lock = threading.RLock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def replace(self, state: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._state = deepcopy(state)
            snapshot = deepcopy(self._state)
        payload = {
            "full_state": True,
            "sections": sorted(snapshot.keys()),
            "state": snapshot,
        }
        self._event_bus.emit("state.replaced", payload)
        return snapshot

    def update_section(self, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current = self._state.setdefault(section, {})
            if not isinstance(current, dict):
                current = {}
                self._state[section] = current
            current.update(deepcopy(values))
            snapshot = deepcopy(self._state)
            section_state = deepcopy(current)
        payload = {
            "section": section,
            "values": deepcopy(values),
            "section_state": section_state,
        }
        self._event_bus.emit(f"{section}.updated", payload)
        return snapshot

    def replace_section(self, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            normalized = deepcopy(values if isinstance(values, dict) else {})
            self._state[str(section or "").strip()] = normalized
            snapshot = deepcopy(self._state)
            section_state = deepcopy(normalized)
        payload = {
            "section": section,
            "values": deepcopy(normalized),
            "section_state": section_state,
            "full_section": True,
        }
        self._event_bus.emit(f"{section}.updated", payload)
        return snapshot

    def set_value(self, key: str, value: Any) -> Dict[str, Any]:
        with self._lock:
            self._state[key] = deepcopy(value)
            snapshot = deepcopy(self._state)
        self._event_bus.emit(
            f"{key}.updated",
            {
                "key": key,
                "value": deepcopy(value),
            },
        )
        return snapshot
