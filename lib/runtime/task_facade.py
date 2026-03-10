# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import lib.runtime.task_orchestrator as task_orchestrator
import lib.runtime.task_service as task_service
from lib.core.dataclasses import ImageData, Settings
from lib.pipeline.context import TaskResult


def build_images_from_paths(host: Any, image_paths: List[str]) -> List[ImageData]:
    return task_service.build_task_images(image_paths)


def get_loaded_image_paths(host: Any) -> List[str]:
    image_paths = host._runtime_loaded_image_paths()
    if not image_paths:
        raise ValueError("no loaded images")
    return image_paths


def start_named_runtime_task(
    host: Any,
    task_key: str,
    image_paths: List[str],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    task_name_hint = str(task_key or "task")
    host._current_task = task_orchestrator.start_named_runtime_task(
        current_task=getattr(host, "_current_task", None),
        is_task_running=host.is_task_running,
        task_runner=host._get_task_runner(),
        task_key=task_key,
        image_paths=image_paths,
        settings_dict=host._runtime_settings_dict(),
        extra=extra,
        on_progress=host.on_pipeline_progress,
        on_image_done=host.on_pipeline_image_done,
        on_batch_done=lambda results: host._on_task_done(
            getattr(getattr(host, "_current_task", None), "name", task_name_hint),
            results,
        ),
        on_error=host.on_pipeline_error,
    )
    host._sync_runtime_task_state()


def is_task_running(host: Any) -> bool:
    return getattr(host, "_current_task", None) is not None and host._current_task.is_running()


def stop_current_task(host: Any):
    task_orchestrator.stop_current_task(
        getattr(host, "_current_task", None),
        host._get_runtime_event_bus().emit,
    )


def run_task(host: Any, TaskClass: Type[Any], images: List[ImageData], extra: Optional[Dict[str, Any]] = None):
    task_name_hint = str(getattr(TaskClass, "__name__", "task") or "task")
    host._current_task = task_orchestrator.run_task_class(
        current_task=getattr(host, "_current_task", None),
        is_task_running=host.is_task_running,
        task_runner=host._get_task_runner(),
        task_class=TaskClass,
        images=images,
        settings_dict=host._runtime_settings_dict(),
        extra=extra,
        on_progress=host.on_pipeline_progress,
        on_image_done=host.on_pipeline_image_done,
        on_batch_done=lambda results: host._on_task_done(
            getattr(getattr(host, "_current_task", None), "name", task_name_hint),
            results,
        ),
        on_error=host.on_pipeline_error,
    )
    host._sync_runtime_task_state()


def get_current_settings_obj(host: Any) -> Settings:
    return task_service.build_settings_object(host._runtime_settings_dict())


def on_task_done(host: Any, name: str, results: List[TaskResult]):
    host.on_pipeline_done(name, results)
    host._current_task = None
    host._sync_runtime_task_state()


def run_tagger(host: Any, images: List[ImageData]):
    host.run_task(task_service.resolve_task_class("tagger"), images)


def run_llm(host: Any, images: List[ImageData], user_prompt: str = None, system_prompt: str = None):
    host.run_task(
        task_service.resolve_task_class("llm"),
        images,
        extra=task_service.build_llm_task_extra(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        ),
    )


def run_unmask(host: Any, images: List[ImageData]):
    host.run_task(task_service.resolve_task_class("unmask"), images)


def run_mask_text(host: Any, images: List[ImageData]):
    host.run_task(task_service.resolve_task_class("mask_text"), images)


def run_image_process(host: Any, images: List[ImageData], edit_prompt: str = None):
    host.run_task(
        task_service.resolve_task_class("image_process"),
        images,
        extra=task_service.build_image_process_task_extra(edit_prompt=edit_prompt),
    )


def run_restore(host: Any, images: List[ImageData]):
    host.run_task(task_service.resolve_task_class("restore"), images)
