# -*- coding: utf-8 -*-
"""
BaseTask - Integrated Task & Pipeline

Task logic plus a lightweight threading wrapper.
"""
import threading
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import Callable, DefaultDict, List, Optional, Dict, Any, Tuple

from lib.pipeline.context import TaskContext, TaskResult
from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta


class SignalProxy:
    """Small signal-like adapter for compatibility with existing connect/emit usage."""

    def __init__(self):
        self._callbacks: List[Callable[..., None]] = []
        self._lock = threading.Lock()

    def connect(self, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def disconnect(self, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(self, *args, **kwargs) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(*args, **kwargs)


class BaseTask:
    """
    BaseTask (Threaded compatibility)
    
    Contains:
    - execution loop
    - signal-like compatibility adapters
    - Abstract execute method for workers
    """

    def __init__(self, 
                 images: List[ImageData],
                 settings: Settings,
                 prompt: Optional[Prompt] = None,
                 folder: Optional[FolderMeta] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 parent=None):
        self.images = images
        self.settings = settings
        self.prompt = prompt
        self.folder = folder
        self.extra = extra or {}

        self._stop_event = False
        self._was_cancelled = False
        self._all_results: List[TaskResult] = []
        self._listeners: DefaultDict[str, List[Callable[..., None]]] = defaultdict(list)
        self._thread: Optional[threading.Thread] = None

        self.progress = SignalProxy()       # current, total, filename, speed
        self.image_done = SignalProxy()     # image_path, TaskResult
        self.batch_done = SignalProxy()     # all_results
        self.error = SignalProxy()          # error message

    @property
    def name(self) -> str:
        """Task name for logs"""
        raise NotImplementedError

    def start(self) -> None:
        if self.isRunning():
            return
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"task-{self.name}")
        self._thread.start()

    def isRunning(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def is_running(self) -> bool:
        return self.isRunning()

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
    
    def stop(self):
        """Request stop"""
        self._stop_event = True

    @property
    def was_cancelled(self) -> bool:
        return bool(self._was_cancelled)

    def add_listener(self, event_name: str, handler: Callable[..., None]) -> None:
        name = str(event_name or "").strip().lower()
        if not name or handler in self._listeners[name]:
            return
        self._listeners[name].append(handler)

    def remove_listener(self, event_name: str, handler: Callable[..., None]) -> None:
        name = str(event_name or "").strip().lower()
        handlers = self._listeners.get(name, [])
        if handler in handlers:
            handlers.remove(handler)
        if not handlers and name in self._listeners:
            self._listeners.pop(name, None)

    def _notify_listeners(self, event_name: str, *args) -> None:
        name = str(event_name or "").strip().lower()
        for handler in list(self._listeners.get(name, [])):
            handler(*args)

    def _emit_progress(self, current: int, total: int, filename: str, speed: float) -> None:
        self.progress.emit(current, total, filename, speed)
        self._notify_listeners("progress", current, total, filename, speed)

    def _emit_image_done(self, image_path: str, result: TaskResult) -> None:
        self.image_done.emit(image_path, result)
        self._notify_listeners("image_done", image_path, result)

    def _emit_batch_done(self, results: List[TaskResult]) -> None:
        copied_results = list(results)
        self.batch_done.emit(copied_results)
        self._notify_listeners("batch_done", copied_results)

    def _emit_error(self, message: str) -> None:
        self.error.emit(message)
        self._notify_listeners("error", message)
        
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """
        Determine if the image should be skipped.
        Returns (should_skip, reason)
        """
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Process a single image. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def execute_pipeline(self) -> List[TaskResult]:
        """
        Main execution loop.

        Kept separate from `run()` so future runtimes can execute the same task
        body without depending on thread-wrapper lifecycle entry points.
        """
        total = len(self.images)
        start_time = time.time()
        self._all_results = []
        self._was_cancelled = False
        
        for i, image in enumerate(self.images):
            if self._stop_event:
                self._was_cancelled = True
                break
            
            elapsed = time.time() - start_time
            speed = (i) / elapsed if elapsed > 0 and i > 0 else 0.0
            self._emit_progress(i + 1, total, image.filename, speed)
            
            task_extra = deepcopy(self.extra)
            if total == 1:
                task_extra["force_execution"] = True

            context = TaskContext(
                image=image,
                settings=self.settings,
                prompt=self.prompt,
                folder=self.folder,
                extra=task_extra,
            )
            
            result = self.execute(context)
            self._all_results.append(result)
            self._emit_image_done(image.path, result)
        
        self._emit_batch_done(self._all_results)
        return list(self._all_results)

    def run_inline(self) -> List[TaskResult]:
        return self.execute_pipeline()

    def run(self):
        """
        Main execution loop (Runs in background thread)
        """
        try:
            self.execute_pipeline()
        except Exception as e:
            traceback.print_exc()
            self._emit_error(str(e))
