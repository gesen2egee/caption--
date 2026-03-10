# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, List, Optional, Type

from lib.core.dataclasses import ImageData, Settings
from lib.runtime.callback_dispatcher import RuntimeCallbackDispatcher
from lib.runtime.events import EventBus


@dataclass
class TaskHandle:
    """
    Compatibility wrapper around the current task object.

    This preserves the minimum surface the legacy UI needs while letting the
    rest of the app depend on a runner abstraction instead of raw thread objects.
    """

    name: str
    task: Any
    thread: Any = None
    stop_requested: bool = False

    def is_running(self) -> bool:
        if not self.task:
            return False
        if hasattr(self.task, "isRunning"):
            return bool(self.task.isRunning())
        if self.thread is not None and hasattr(self.thread, "is_alive"):
            return bool(self.thread.is_alive())
        return False

    def isRunning(self) -> bool:
        return self.is_running()

    def stop(self) -> None:
        self.stop_requested = True
        if self.task and hasattr(self.task, "stop"):
            self.task.stop()


class TaskRunner:
    """
    Starts legacy task classes and translates their lifecycle into runtime events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        callback_dispatcher: Optional[RuntimeCallbackDispatcher] = None,
    ):
        self.event_bus = event_bus
        self.callback_dispatcher = callback_dispatcher or RuntimeCallbackDispatcher.create_default()

    @staticmethod
    def _attach_listener(task: Any, event_name: str, handler: Callable[..., None]) -> None:
        if hasattr(task, "add_listener"):
            task.add_listener(event_name, handler)
            return

        signal = getattr(task, event_name, None)
        if signal is not None and hasattr(signal, "connect"):
            signal.connect(handler)
            return

        raise AttributeError(f"task does not expose listener or signal for '{event_name}'")

    def _dispatch(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        if self.callback_dispatcher is not None:
            self.callback_dispatcher.dispatch(callback, *args, **kwargs)
            return
        callback(*args, **kwargs)

    def start_task(
        self,
        task_class: Type[Any],
        images: List[ImageData],
        settings: Settings,
        extra: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_image_done: Optional[Callable[..., None]] = None,
        on_batch_done: Optional[Callable[[List[Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> TaskHandle:
        task = task_class(
            images=images,
            settings=settings,
            prompt=None,
            folder=None,
            extra=extra,
        )
        handle = TaskHandle(name=task.name, task=task)

        self.event_bus.emit(
            "task.started",
            {
                "task_name": handle.name,
                "image_count": len(images),
            },
        )

        def _progress(current: int, total: int, filename: str, speed: float = 0.0) -> None:
            def _deliver() -> None:
                self.event_bus.emit(
                    "task.progress",
                    {
                        "task_name": handle.name,
                        "current": current,
                        "total": total,
                        "filename": filename,
                        "speed": speed,
                    },
                )
                if on_progress is not None:
                    on_progress(current, total, filename, speed)

            self._dispatch(_deliver)

        def _image_done(image_path: str, result: Any) -> None:
            def _deliver() -> None:
                self.event_bus.emit(
                    "task.image_done",
                    {
                        "task_name": handle.name,
                        "image_path": image_path,
                        "success": bool(getattr(result, "success", False)),
                        "skipped": bool(getattr(result, "skipped", False)),
                        "error": getattr(result, "error", None),
                        "error_info": getattr(result, "error_info", None),
                    },
                )
                if on_image_done is not None:
                    on_image_done(image_path, result)

            self._dispatch(_deliver)

        def _batch_done(results: List[Any]) -> None:
            def _deliver() -> None:
                cancelled = bool(getattr(task, "was_cancelled", False) or handle.stop_requested)
                self.event_bus.emit(
                    "task.batch_done",
                    {
                        "task_name": handle.name,
                        "result_count": len(results),
                        "cancelled": cancelled,
                    },
                )
                if cancelled:
                    self.event_bus.emit(
                        "task.cancelled",
                        {
                            "task_name": handle.name,
                            "result_count": len(results),
                        },
                    )
                if on_batch_done is not None:
                    on_batch_done(results)

            self._dispatch(_deliver)

        def _error(message: str) -> None:
            def _deliver() -> None:
                self.event_bus.emit(
                    "task.failed",
                    {
                        "task_name": handle.name,
                        "error": message,
                    },
                )
                if on_error is not None:
                    on_error(message)

            self._dispatch(_deliver)

        self._attach_listener(task, "progress", _progress)
        self._attach_listener(task, "image_done", _image_done)
        self._attach_listener(task, "batch_done", _batch_done)
        self._attach_listener(task, "error", _error)

        if hasattr(task, "start"):
            task.start()
        elif hasattr(task, "run_inline"):
            worker_thread = threading.Thread(target=task.run_inline, daemon=True)
            worker_thread.start()
            handle.thread = worker_thread
        else:
            raise AttributeError("task does not expose a start() or run_inline() entry point")
        return handle
