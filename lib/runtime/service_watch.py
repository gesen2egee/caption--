# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from lib.runtime.events import EventBus
from lib.runtime.service_reload import (
    list_reloadable_runtime_services,
    reload_runtime_services,
)


class RuntimeServiceWatcher:
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        can_reload: Optional[Callable[[], bool]] = None,
        context_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self._event_bus = event_bus
        self._can_reload = can_reload
        self._context_provider = context_provider
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._interval_seconds = 1.0
        self._mtimes: Dict[str, float] = {}
        self._service_modules: Dict[str, str] = {}
        self._service_paths: Dict[str, str] = {}
        self._last_reload: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._pending_changes: list[str] = []

    def start(self, interval_seconds: float = 1.0) -> Dict[str, Any]:
        with self._lock:
            self._interval_seconds = max(0.25, float(interval_seconds or 1.0))
            self._refresh_paths()
            if self._thread is not None and self._thread.is_alive():
                return self.status()

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._watch_loop,
                daemon=True,
                name="RuntimeServiceWatcher",
            )
            self._thread.start()

        self._emit("backend.services.watch.started", self.status())
        return self.status()

    def stop(self) -> Dict[str, Any]:
        thread: Optional[threading.Thread]
        with self._lock:
            thread = self._thread
            if thread is None:
                return self.status()
            self._stop_event.set()

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        with self._lock:
            if self._thread is thread:
                self._thread = None

        self._emit("backend.services.watch.stopped", self.status())
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            watched = {
                name: {
                    "module": self._service_modules.get(name),
                    "path": self._service_paths.get(name),
                    "mtime": self._mtimes.get(name),
                }
                for name in sorted(self._service_modules.keys())
            }
            return {
                "enabled": bool(self._thread is not None and self._thread.is_alive()),
                "interval_seconds": self._interval_seconds,
                "service_count": len(watched),
                "services": watched,
                "pending_changes": list(self._pending_changes),
                "last_reload": self._last_reload,
                "last_error": self._last_error,
            }

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                changed_names = self._collect_changed_services()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                self._emit(
                    "backend.services.watch.failed",
                    {
                        "error": str(exc),
                    },
                )
                continue

            if not changed_names:
                continue

            pending = list(dict.fromkeys([*self._pending_changes, *changed_names]))
            if callable(self._can_reload) and not self._can_reload():
                with self._lock:
                    self._pending_changes = pending
                self._emit(
                    "backend.services.auto_reload_deferred",
                    {
                        "changed_names": pending,
                        **self._context_payload(),
                    },
                )
                continue

            reload_result = reload_runtime_services(pending)
            timestamp = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._refresh_paths()
                self._pending_changes = []
                self._last_reload = {
                    "timestamp": timestamp,
                    "changed_names": pending,
                    "result": reload_result,
                }
                self._last_error = None if reload_result.get("error_count", 0) == 0 else "reload_errors"

            event_name = (
                "backend.services.auto_reload_failed"
                if reload_result.get("error_count", 0)
                else "backend.services.auto_reloaded"
            )
            self._emit(
                event_name,
                {
                    "timestamp": timestamp,
                    "changed_names": pending,
                    **reload_result,
                },
            )

    def _collect_changed_services(self) -> list[str]:
        changed: list[str] = []
        with self._lock:
            self._refresh_paths()
            for name, path in self._service_paths.items():
                if not path or not os.path.isfile(path):
                    continue
                try:
                    current_mtime = os.path.getmtime(path)
                except OSError:
                    continue
                previous_mtime = self._mtimes.get(name)
                if previous_mtime is None:
                    self._mtimes[name] = current_mtime
                    continue
                if current_mtime > previous_mtime:
                    self._mtimes[name] = current_mtime
                    changed.append(name)
        return changed

    def _refresh_paths(self) -> None:
        services = list_reloadable_runtime_services()
        self._service_modules = services
        for name, module_name in services.items():
            try:
                module = importlib.import_module(module_name)
                module_path = str(getattr(module, "__file__", "") or "")
            except Exception:
                module_path = ""
            self._service_paths[name] = module_path
            if module_path and name not in self._mtimes and os.path.isfile(module_path):
                try:
                    self._mtimes[name] = os.path.getmtime(module_path)
                except OSError:
                    pass

    def _emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        if self._event_bus is not None:
            self._event_bus.emit(event_name, payload)

    def _context_payload(self) -> Dict[str, Any]:
        if callable(self._context_provider):
            try:
                payload = self._context_provider()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {}
        return {}
