# -*- coding: utf-8 -*-
from __future__ import annotations

import queue
import sys
import threading
import traceback
from typing import Any, Callable, Optional

_QT_PROXY_CLASS = None


def _qt_core(import_if_needed: bool = False):
    module = sys.modules.get("PyQt6.QtCore")
    if module is None and import_if_needed:
        try:
            import PyQt6.QtCore as module  # type: ignore[no-redef]
        except Exception:
            return None
    if module is None:
        return None
    return {
        "QObject": getattr(module, "QObject", object),
        "QCoreApplication": getattr(module, "QCoreApplication", None),
        "QMetaObject": getattr(module, "QMetaObject", None),
        "Qt": getattr(module, "Qt", None),
        "pyqtSlot": getattr(module, "pyqtSlot", lambda *args, **kwargs: (lambda fn: fn)),
    }


def _qt_proxy_class():
    global _QT_PROXY_CLASS
    if _QT_PROXY_CLASS is not None:
        return _QT_PROXY_CLASS

    qt_core = _qt_core(import_if_needed=False)
    if qt_core is None:
        return None

    QObject = qt_core["QObject"]
    pyqtSlot = qt_core["pyqtSlot"]

    class _QtDrainProxy(QObject):
        def __init__(self, owner, parent=None):
            super().__init__(parent)
            self._owner = owner

        @pyqtSlot()
        def drain(self) -> None:
            self._owner._drain_queue()

    _QT_PROXY_CLASS = _QtDrainProxy
    return _QT_PROXY_CLASS


class RuntimeCallbackDispatcher:
    """
    Marshal arbitrary callbacks back onto the Qt application thread.

    Task execution may happen in plain Python threads. UI mutations and any
    code that assumes the Qt main thread should go through this dispatcher.
    """

    def __init__(self, parent: Optional[Any] = None):
        self._pending: "queue.Queue[tuple[Callable[..., None], tuple[Any, ...], dict[str, Any]]]" = queue.Queue()
        self._schedule_lock = threading.Lock()
        self._drain_scheduled = False
        self._qt_proxy = None

        proxy_class = _qt_proxy_class()
        if proxy_class is not None:
            self._qt_proxy = proxy_class(self, parent=parent)

    @classmethod
    def create_default(cls) -> Optional["RuntimeCallbackDispatcher"]:
        qt_core = _qt_core(import_if_needed=False)
        if qt_core is None or qt_core["QCoreApplication"] is None:
            return None
        app = qt_core["QCoreApplication"].instance()
        if app is None:
            return None
        return cls(parent=app)

    def dispatch(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        qt_core = _qt_core(import_if_needed=False)
        app = qt_core["QCoreApplication"].instance() if qt_core is not None and qt_core["QCoreApplication"] is not None else None
        if app is None or self._qt_proxy is None or threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
            return
        self._pending.put((callback, tuple(args), dict(kwargs)))

        should_schedule = False
        with self._schedule_lock:
            if not self._drain_scheduled:
                self._drain_scheduled = True
                should_schedule = True

        if should_schedule:
            qt_meta = qt_core["QMetaObject"]
            qt_ns = qt_core["Qt"]
            qt_meta.invokeMethod(self._qt_proxy, "drain", qt_ns.ConnectionType.QueuedConnection)

    def _drain_queue(self) -> None:
        while True:
            try:
                callback, args, kwargs = self._pending.get_nowait()
            except queue.Empty:
                break

            try:
                callback(*args, **kwargs)
            except Exception:
                traceback.print_exc()

        should_reschedule = False
        with self._schedule_lock:
            self._drain_scheduled = False
            if not self._pending.empty():
                self._drain_scheduled = True
                should_reschedule = True

        if should_reschedule:
            qt_core = _qt_core(import_if_needed=False)
            if qt_core is not None and self._qt_proxy is not None:
                qt_core["QMetaObject"].invokeMethod(
                    self._qt_proxy,
                    "drain",
                    qt_core["Qt"].ConnectionType.QueuedConnection,
                )
