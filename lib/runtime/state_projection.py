# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def build_selection_state(
    *,
    root_dir_path: str,
    current_image_path: str,
    current_index: int,
    loaded_image_paths: Iterable[str],
    all_image_paths: Iterable[str],
    filtered_image_paths: Iterable[str],
    has_raw_backup: bool,
    filter_active: bool,
    filter_query: str,
    filter_tags: bool,
    filter_text: bool,
) -> Dict[str, Any]:
    loaded_paths = list(loaded_image_paths or [])
    return {
        "root_dir_path": str(root_dir_path or ""),
        "current_image_path": str(current_image_path or ""),
        "current_index": int(current_index),
        "image_count": len(loaded_paths),
        "loaded_image_paths": loaded_paths,
        "all_image_paths": list(all_image_paths or []),
        "filtered_image_paths": list(filtered_image_paths or []),
        "has_raw_backup": bool(has_raw_backup),
        "filter_active": bool(filter_active),
        "filter_query": str(filter_query or ""),
        "filter_tags": bool(filter_tags),
        "filter_text": bool(filter_text),
    }


def build_task_state(*, running: bool, task_name: str | None) -> Dict[str, Any]:
    return {
        "running": bool(running),
        "task_name": task_name,
    }


def build_ui_state(
    *,
    current_view_mode: int,
    temp_view_mode: Any,
    current_prompt_mode: str,
    nl_page_index: int,
    nl_page_count: int,
    current_tab_index: int,
) -> Dict[str, Any]:
    return {
        "current_view_mode": int(current_view_mode),
        "temp_view_mode": temp_view_mode,
        "current_prompt_mode": str(current_prompt_mode or "default"),
        "nl_page_index": int(nl_page_index),
        "nl_page_count": int(nl_page_count),
        "current_tab_index": int(current_tab_index),
    }


def build_content_state(
    *,
    prompt_text: str,
    image_process_prompt_text: str,
    txt_content: str,
    nl_latest: str,
) -> Dict[str, Any]:
    return {
        "prompt_text": str(prompt_text or ""),
        "image_process_prompt_text": str(image_process_prompt_text or ""),
        "txt_content": str(txt_content or ""),
        "nl_latest": str(nl_latest or ""),
    }


def build_controls_state(
    *,
    current_index: int,
    filter_query: str,
    filter_tags: bool,
    filter_text: bool,
    view_mode: int,
    tagger_save_to_txt: bool,
    llm_save_to_txt: bool,
) -> Dict[str, Any]:
    return {
        "current_index": int(current_index),
        "filter_query": str(filter_query or ""),
        "filter_tags": bool(filter_tags),
        "filter_text": bool(filter_text),
        "view_mode": int(view_mode),
        "tagger_save_to_txt": bool(tagger_save_to_txt),
        "llm_save_to_txt": bool(llm_save_to_txt),
    }


def build_tags_state(
    *,
    folder_meta: Iterable[str],
    custom: Iterable[str],
    tagger: Iterable[str],
    nl: Iterable[str],
    translations: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    return {
        "folder_meta": list(folder_meta or []),
        "custom": list(custom or []),
        "tagger": list(tagger or []),
        "nl": list(nl or []),
        "translations": dict(translations or {}),
    }
