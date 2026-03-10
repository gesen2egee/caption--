# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional
from collections import defaultdict, deque
import threading
import traceback


@dataclass
class RuntimeEvent:
    """
    Structured runtime event.

    Events are intentionally plain Python data so they can later be bridged to
    logs, HTTP/WebSocket streams, tests, or Agent consumers.
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


RUNTIME_EVENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "all": {
        "description": "All runtime events, including low-level section sync and experimental namespaces.",
        "include_prefixes": [],
        "exclude_prefixes": [],
    },
    "control": {
        "description": "High-signal control plane events for commands, tasks, workers, UI spec changes, and user-visible side effects.",
        "include_prefixes": [
            "backend.",
            "bridge.",
            "command.",
            "editor.",
            "image.",
            "settings.",
            "task.",
            "tags.custom.",
            "ui.spec.",
            "worker.",
        ],
        "exclude_prefixes": [],
    },
    "diagnostic": {
        "description": "Verbose runtime diagnostics across the stable runtime namespaces, including section sync and telemetry.",
        "include_prefixes": [
            "backend.",
            "bridge.",
            "command.",
            "commands.",
            "content.",
            "controls.",
            "editor.",
            "image.",
            "selection.",
            "settings.",
            "tags.",
            "task.",
            "ui.",
            "worker.",
        ],
        "exclude_prefixes": [],
    },
}


def get_runtime_event_profiles() -> Dict[str, Dict[str, Any]]:
    return deepcopy(RUNTIME_EVENT_PROFILES)


def resolve_event_filters(
    profile: Optional[str] = None,
    include_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> tuple[str, List[str], List[str]]:
    profile_name = str(profile or "").strip() or "all"
    if profile_name not in RUNTIME_EVENT_PROFILES:
        raise ValueError(f"unknown event profile: {profile_name}")

    base_profile = RUNTIME_EVENT_PROFILES[profile_name]
    merged_include = EventBus._normalize_prefixes(base_profile.get("include_prefixes"))
    merged_exclude = EventBus._normalize_prefixes(base_profile.get("exclude_prefixes"))
    merged_include.extend(EventBus._normalize_prefixes(include_prefixes))
    merged_exclude.extend(EventBus._normalize_prefixes(exclude_prefixes))
    return profile_name, merged_include, merged_exclude


class EventBus:
    """
    Minimal in-process event bus.

    The current app only needs fan-out inside one process. The API is kept
    intentionally small so it can be adapted to an external transport later.
    """

    def __init__(self, max_history: int = 400):
        self._subscribers: DefaultDict[str, List[Callable[[RuntimeEvent], None]]] = defaultdict(list)
        self._max_history = max(1, int(max_history or 1))
        self._history = deque(maxlen=self._max_history)
        self._lock = threading.RLock()

    def subscribe(self, event_name: str, callback: Callable[[RuntimeEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers[event_name].append(callback)

        def unsubscribe() -> None:
            with self._lock:
                callbacks = self._subscribers.get(event_name, [])
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    @staticmethod
    def _normalize_prefixes(prefixes: Optional[Iterable[str]]) -> List[str]:
        if prefixes is None:
            return []
        items: List[str] = []
        for prefix in prefixes:
            value = str(prefix or "").strip()
            if value:
                items.append(value)
        return items

    @classmethod
    def _matches_event_name(
        cls,
        event_name: str,
        include_prefixes: Optional[Iterable[str]] = None,
        exclude_prefixes: Optional[Iterable[str]] = None,
    ) -> bool:
        include_items = cls._normalize_prefixes(include_prefixes)
        exclude_items = cls._normalize_prefixes(exclude_prefixes)

        if include_items and not any(event_name.startswith(prefix) for prefix in include_items):
            return False
        if exclude_items and any(event_name.startswith(prefix) for prefix in exclude_items):
            return False
        return True

    def emit(self, event_name: str, payload: Dict[str, Any] | None = None) -> RuntimeEvent:
        event = RuntimeEvent(name=event_name, payload=payload or {})
        with self._lock:
            self._history.append(event)
            named_callbacks = list(self._subscribers.get(event_name, []))
            wildcard_callbacks = list(self._subscribers.get("*", []))

        for callback in named_callbacks:
            try:
                callback(event)
            except Exception:
                traceback.print_exc()

        for callback in wildcard_callbacks:
            try:
                callback(event)
            except Exception:
                traceback.print_exc()

        return event

    def history(
        self,
        limit: Optional[int] = None,
        include_prefixes: Optional[Iterable[str]] = None,
        exclude_prefixes: Optional[Iterable[str]] = None,
    ) -> List[RuntimeEvent]:
        with self._lock:
            items = [
                event
                for event in self._history
                if self._matches_event_name(
                    event.name,
                    include_prefixes=include_prefixes,
                    exclude_prefixes=exclude_prefixes,
                )
            ]
        if limit is None:
            return items

        limit = max(0, int(limit))
        if limit == 0:
            return []
        return items[-limit:]

    def history_limit(self) -> int:
        return self._max_history
