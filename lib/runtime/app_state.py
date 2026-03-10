# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any, Dict, Optional


class RuntimeAppState:
    """
    Central runtime-owned app state.

    Unlike RuntimeStateStore, this object is intended to be the in-process
    source of truth for runtime-readable state, while RuntimeStateStore remains
    the event-emitting projection consumed by the shell and debug tooling.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state: Dict[str, Any] = deepcopy(initial_state or {})
        self._lock = threading.RLock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def replace(self, state: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._state = deepcopy(state)
            return deepcopy(self._state)

    def get_section(self, section: str) -> Dict[str, Any]:
        with self._lock:
            values = self._state.get(str(section or "").strip(), {})
            return deepcopy(values) if isinstance(values, dict) else {}

    def update_section(self, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current = self._state.setdefault(section, {})
            if not isinstance(current, dict):
                current = {}
                self._state[section] = current
            current.update(deepcopy(values))
            return deepcopy(current)

    def replace_section(self, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            normalized = deepcopy(values if isinstance(values, dict) else {})
            self._state[str(section or "").strip()] = normalized
            return deepcopy(normalized)
