# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

import lib.runtime.task_service as task_service


def get_task_status(current_task: Any, is_task_running: Callable[[], bool]) -> Dict[str, Any]:
    return {
        "running": bool(is_task_running()),
        "task_name": current_task.name if current_task else None,
    }


def start_named_runtime_task(
    *,
    current_task: Any,
    is_task_running: Callable[[], bool],
    task_runner: Any,
    task_key: str,
    image_paths: List[str],
    settings_dict: Dict[str, Any],
    extra: Optional[Dict[str, Any]],
    on_progress: Callable[..., Any],
    on_image_done: Callable[..., Any],
    on_batch_done: Callable[[List[Any]], Any],
    on_error: Callable[[str], Any],
) -> Any:
    if is_task_running():
        on_error("已有任務正在執行 (Task Running)")
        return current_task

    return task_service.start_named_task(
        task_runner=task_runner,
        task_key=task_key,
        image_paths=image_paths,
        settings_dict=settings_dict,
        extra=extra,
        on_progress=on_progress,
        on_image_done=on_image_done,
        on_batch_done=on_batch_done,
        on_error=on_error,
    )


def stop_current_task(current_task: Any, emit_event: Callable[[str, Dict[str, Any]], None]) -> bool:
    if not current_task:
        return False
    current_task.stop()
    emit_event(
        "task.cancel_requested",
        {
            "task_name": current_task.name,
        },
    )
    return True


def run_task_class(
    *,
    current_task: Any,
    is_task_running: Callable[[], bool],
    task_runner: Any,
    task_class: Type[Any],
    images: List[Any],
    settings_dict: Dict[str, Any],
    extra: Optional[Dict[str, Any]],
    on_progress: Callable[..., Any],
    on_image_done: Callable[..., Any],
    on_batch_done: Callable[[List[Any]], Any],
    on_error: Callable[[str], Any],
) -> Any:
    if is_task_running():
        on_error("已有任務正在執行 (Task Running)")
        return current_task

    return task_service.start_task_class(
        task_runner=task_runner,
        task_class=task_class,
        images=images,
        settings_dict=settings_dict,
        extra=extra,
        on_progress=on_progress,
        on_image_done=on_image_done,
        on_batch_done=on_batch_done,
        on_error=on_error,
    )

