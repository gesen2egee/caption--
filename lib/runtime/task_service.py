# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Optional, Type

from lib.core.dataclasses import ImageData, Settings
from lib.runtime.task_runner import TaskHandle, TaskRunner
from lib.utils.file_ops import create_image_data_list


TASK_CLASS_SPECS: Dict[str, tuple[str, str]] = {
    "tagger": ("lib.pipeline.tasks", "TaggerTask"),
    "llm": ("lib.pipeline.tasks", "LLMTask"),
    "image_process": ("lib.pipeline.tasks", "ImageProcessTask"),
    "unmask": ("lib.pipeline.tasks", "UnmaskTask"),
    "mask_text": ("lib.pipeline.tasks", "MaskTextTask"),
    "restore": ("lib.pipeline.tasks", "RestoreTask"),
}


def normalize_image_paths(image_paths: Iterable[str]) -> List[str]:
    normalized = [str(path or "").strip() for path in image_paths or []]
    normalized = [path for path in normalized if path]
    if not normalized:
        raise ValueError("image_paths is required")
    return normalized


def build_task_images(image_paths: Iterable[str]) -> List[ImageData]:
    return create_image_data_list(normalize_image_paths(image_paths))


def build_settings_object(settings_dict: Dict[str, Any]) -> Settings:
    valid_keys = Settings.__annotations__.keys()
    clean_settings = {key: value for key, value in dict(settings_dict or {}).items() if key in valid_keys}
    return Settings(**clean_settings)


def build_llm_task_extra(
    *,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if user_prompt:
        extra["user_prompt"] = user_prompt
    if system_prompt:
        extra["system_prompt"] = system_prompt
    return extra


def build_image_process_task_extra(*, edit_prompt: Optional[str] = None) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if edit_prompt:
        extra["edit_prompt"] = edit_prompt
    return extra


def resolve_task_class(task_key: str) -> Type[Any]:
    normalized_key = str(task_key or "").strip().lower()
    module_name, class_name = TASK_CLASS_SPECS[normalized_key]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def start_task_class(
    *,
    task_runner: TaskRunner,
    task_class: Type[Any],
    images: List[ImageData],
    settings_dict: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    on_progress=None,
    on_image_done=None,
    on_batch_done=None,
    on_error=None,
) -> TaskHandle:
    settings_obj = build_settings_object(settings_dict)
    return task_runner.start_task(
        task_class=task_class,
        images=images,
        settings=settings_obj,
        extra=extra,
        on_progress=on_progress,
        on_image_done=on_image_done,
        on_batch_done=on_batch_done,
        on_error=on_error,
    )


def start_named_task(
    *,
    task_runner: TaskRunner,
    task_key: str,
    image_paths: Iterable[str],
    settings_dict: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    on_progress=None,
    on_image_done=None,
    on_batch_done=None,
    on_error=None,
) -> TaskHandle:
    return start_task_class(
        task_runner=task_runner,
        task_class=resolve_task_class(task_key),
        images=build_task_images(image_paths),
        settings_dict=settings_dict,
        extra=extra,
        on_progress=on_progress,
        on_image_done=on_image_done,
        on_batch_done=on_batch_done,
        on_error=on_error,
    )
