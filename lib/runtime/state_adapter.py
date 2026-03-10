# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict, List

import lib.runtime.state_projection as state_projection
from lib.utils.file_ops import has_raw_backup
from lib.utils.parsing import remove_underline


UNSET = object()


def runtime_state_section(host: Any, section: str) -> Dict[str, Any]:
    return host._get_runtime_app_state().get_section(section)


def runtime_selection_value(host: Any, key: str, default: Any = None) -> Any:
    return runtime_state_section(host, "selection").get(key, default)


def runtime_controls_value(host: Any, key: str, default: Any = None) -> Any:
    return runtime_state_section(host, "controls").get(key, default)


def runtime_content_value(host: Any, key: str, default: Any = None) -> Any:
    return runtime_state_section(host, "content").get(key, default)


def runtime_current_image_path(host: Any) -> str:
    current_image_path = str(runtime_selection_value(host, "current_image_path", "") or "").strip()
    if current_image_path:
        return current_image_path
    return str(getattr(host, "current_image_path", "") or "").strip()


def runtime_loaded_image_paths(host: Any) -> List[str]:
    cached_paths = runtime_selection_value(host, "loaded_image_paths", [])
    if isinstance(cached_paths, list):
        image_paths = [str(path) for path in cached_paths if str(path or "").strip()]
        if image_paths:
            return image_paths
    image_paths = list(getattr(host, "image_files", []) or [])
    if image_paths:
        return image_paths
    return []


def runtime_all_image_paths(host: Any) -> List[str]:
    cached_paths = runtime_selection_value(host, "all_image_paths", [])
    if isinstance(cached_paths, list):
        image_paths = [str(path) for path in cached_paths if str(path or "").strip()]
        if image_paths:
            return image_paths
    image_paths = list(getattr(host, "all_image_files", []) or [])
    if image_paths:
        return image_paths
    return []


def runtime_filtered_image_paths(host: Any) -> List[str]:
    cached_paths = runtime_selection_value(host, "filtered_image_paths", [])
    if isinstance(cached_paths, list):
        image_paths = [str(path) for path in cached_paths if str(path or "").strip()]
        if image_paths:
            return image_paths
    image_paths = list(getattr(host, "filtered_image_files", []) or [])
    if image_paths:
        return image_paths
    return []


def runtime_prompt_text(host: Any) -> str:
    if hasattr(host, "prompt_edit") and host.prompt_edit is not None:
        try:
            return host.prompt_edit.toPlainText()
        except Exception:
            pass
    return str(runtime_content_value(host, "prompt_text", "") or "")


def runtime_image_process_prompt_text(host: Any) -> str:
    if hasattr(host, "img_prompt_edit") and host.img_prompt_edit is not None:
        try:
            return host.img_prompt_edit.toPlainText()
        except Exception:
            pass
    return str(runtime_content_value(host, "image_process_prompt_text", "") or "")


def runtime_txt_content(host: Any) -> str:
    if hasattr(host, "txt_edit") and host.txt_edit is not None:
        try:
            return host.txt_edit.toPlainText()
        except Exception:
            pass
    return str(runtime_content_value(host, "txt_content", "") or "")


def runtime_tagger_save_to_txt(host: Any) -> bool:
    if hasattr(host, "chk_tags_save_txt") and host.chk_tags_save_txt is not None:
        return bool(host.chk_tags_save_txt.isChecked())
    return bool(runtime_controls_value(host, "tagger_save_to_txt", True))


def runtime_llm_save_to_txt(host: Any) -> bool:
    if hasattr(host, "chk_llm_save_txt") and host.chk_llm_save_txt is not None:
        return bool(host.chk_llm_save_txt.isChecked())
    return bool(runtime_controls_value(host, "llm_save_to_txt", True))


def build_tag_translations(host: Any, *tag_groups: List[str]) -> Dict[str, str]:
    translations_csv = getattr(host, "translations_csv", {}) or {}
    translation_map: Dict[str, str] = {}

    for group in tag_groups:
        for raw_tag in group or []:
            exact_tag = str(raw_tag or "").strip()
            if not exact_tag:
                continue
            normalized_tag = remove_underline(exact_tag)
            translation = str(translations_csv.get(normalized_tag, "") or "").strip()
            if not translation:
                continue
            translation_map[normalized_tag] = translation
            translation_map[exact_tag] = translation

    return translation_map


def selection_command_result(host: Any, action: str, **extra: Any) -> Dict[str, Any]:
    result = {
        "completed": True,
        "action": action,
        "selection": runtime_state_section(host, "selection"),
    }
    result.update(extra)
    return result


def ui_command_result(host: Any, action: str, **extra: Any) -> Dict[str, Any]:
    result = {
        "completed": True,
        "action": action,
        "ui": runtime_state_section(host, "ui"),
        "controls": runtime_state_section(host, "controls"),
    }
    result.update(extra)
    return result


def content_command_result(host: Any, action: str, **extra: Any) -> Dict[str, Any]:
    result = {
        "completed": True,
        "action": action,
        "content": runtime_state_section(host, "content"),
    }
    result.update(extra)
    return result


def set_runtime_selection_values(
    host: Any,
    *,
    root_dir_path: Any = UNSET,
    current_image_path: Any = UNSET,
    current_index: Any = UNSET,
    current_folder_path: Any = UNSET,
    filter_active: Any = UNSET,
    loaded_image_paths: Any = UNSET,
    all_image_paths: Any = UNSET,
    filtered_image_paths: Any = UNSET,
    filter_query: Any = UNSET,
) -> None:
    updates: Dict[str, Any] = {}

    if root_dir_path is not UNSET:
        host.root_dir_path = str(root_dir_path or "")
        updates["root_dir_path"] = host.root_dir_path
    if current_image_path is not UNSET:
        normalized_path = str(current_image_path or "")
        host.current_image_path = normalized_path
        updates["current_image_path"] = normalized_path
        if current_folder_path is UNSET:
            current_folder_path = os.path.dirname(normalized_path) if normalized_path else ""
    if current_index is not UNSET:
        host.current_index = int(current_index)
        updates["current_index"] = host.current_index
    if current_folder_path is not UNSET:
        host.current_folder_path = str(current_folder_path or "")
    if filter_active is not UNSET:
        host.filter_active = bool(filter_active)
        updates["filter_active"] = host.filter_active
    if loaded_image_paths is not UNSET:
        host.image_files = list(loaded_image_paths or [])
        updates["loaded_image_paths"] = list(host.image_files)
        updates["image_count"] = len(host.image_files)
    if all_image_paths is not UNSET:
        host.all_image_files = list(all_image_paths or [])
        updates["all_image_paths"] = list(host.all_image_files)
    if filtered_image_paths is not UNSET:
        host.filtered_image_files = list(filtered_image_paths or [])
        updates["filtered_image_paths"] = list(host.filtered_image_files)
    if filter_query is not UNSET:
        updates["filter_query"] = str(filter_query or "")

    if updates:
        host._update_runtime_state_section("selection", updates)


def sync_runtime_settings_state(host: Any) -> None:
    host._replace_runtime_state_section("settings", host._runtime_settings_dict())


def sync_runtime_selection_state(host: Any) -> None:
    current_image_path = runtime_current_image_path(host)
    root_dir_path = getattr(host, "root_dir_path", None)
    if root_dir_path is None:
        root_dir_path = runtime_selection_value(host, "root_dir_path", "")
    current_index = getattr(host, "current_index", None)
    if current_index is None:
        current_index = runtime_selection_value(host, "current_index", -1)
    image_paths = runtime_loaded_image_paths(host)
    filter_active = bool(getattr(host, "filter_active", runtime_selection_value(host, "filter_active", False)))
    if hasattr(host, "filter_input") and host.filter_input is not None:
        filter_query = host.filter_input.text()
    else:
        filter_query = str(runtime_selection_value(host, "filter_query", "") or "")
    if hasattr(host, "chk_filter_tags") and host.chk_filter_tags is not None:
        filter_tags = bool(host.chk_filter_tags.isChecked())
    else:
        filter_tags = bool(runtime_selection_value(host, "filter_tags", True))
    if hasattr(host, "chk_filter_text") and host.chk_filter_text is not None:
        filter_text = bool(host.chk_filter_text.isChecked())
    else:
        filter_text = bool(runtime_selection_value(host, "filter_text", False))
    host._replace_runtime_state_section(
        "selection",
        state_projection.build_selection_state(
            root_dir_path=root_dir_path,
            current_image_path=current_image_path,
            current_index=current_index,
            loaded_image_paths=image_paths,
            all_image_paths=list(getattr(host, "all_image_files", []) or []),
            filtered_image_paths=list(getattr(host, "filtered_image_files", []) or []),
            has_raw_backup=bool(current_image_path and has_raw_backup(current_image_path)),
            filter_active=filter_active,
            filter_query=filter_query,
            filter_tags=filter_tags,
            filter_text=filter_text,
        ),
    )


def sync_runtime_task_state(host: Any) -> None:
    host._replace_runtime_state_section(
        "task",
        state_projection.build_task_state(**host.get_task_status()),
    )


def sync_runtime_ui_state(host: Any) -> None:
    current_tab_index = (
        host.tabs.currentIndex()
        if hasattr(host, "tabs") and host.tabs is not None
        else runtime_state_section(host, "ui").get("current_tab_index", 0)
    )
    host._replace_runtime_state_section(
        "ui",
        state_projection.build_ui_state(
            current_view_mode=getattr(host, "current_view_mode", 0),
            temp_view_mode=getattr(host, "temp_view_mode", None),
            current_prompt_mode=getattr(host, "current_prompt_mode", "default"),
            nl_page_index=getattr(host, "nl_page_index", 0),
            nl_page_count=len(getattr(host, "nl_pages", []) or []),
            current_tab_index=current_tab_index,
        ),
    )


def sync_runtime_content_state(host: Any) -> None:
    host._replace_runtime_state_section(
        "content",
        state_projection.build_content_state(
            prompt_text=runtime_prompt_text(host),
            image_process_prompt_text=runtime_image_process_prompt_text(host),
            txt_content=runtime_txt_content(host),
            nl_latest=getattr(host, "nl_latest", ""),
        ),
    )


def sync_runtime_controls_state(host: Any) -> None:
    current_index = getattr(host, "current_index", None)
    if current_index is None:
        current_index = max(int(runtime_controls_value(host, "current_index", 1)) - 1, -1)
    if hasattr(host, "filter_input") and host.filter_input is not None:
        filter_query = host.filter_input.text()
    else:
        filter_query = str(runtime_controls_value(host, "filter_query", "") or "")
    if hasattr(host, "chk_filter_tags") and host.chk_filter_tags is not None:
        filter_tags = bool(host.chk_filter_tags.isChecked())
    else:
        filter_tags = bool(runtime_controls_value(host, "filter_tags", True))
    if hasattr(host, "chk_filter_text") and host.chk_filter_text is not None:
        filter_text = bool(host.chk_filter_text.isChecked())
    else:
        filter_text = bool(runtime_controls_value(host, "filter_text", False))
    if hasattr(host, "cb_view_mode") and host.cb_view_mode is not None:
        view_mode = host.cb_view_mode.currentIndex()
    else:
        view_mode = int(runtime_controls_value(host, "view_mode", getattr(host, "current_view_mode", 0)) or 0)
    host._replace_runtime_state_section(
        "controls",
        state_projection.build_controls_state(
            current_index=int(current_index) + 1,
            filter_query=filter_query,
            filter_tags=filter_tags,
            filter_text=filter_text,
            view_mode=view_mode,
            tagger_save_to_txt=runtime_tagger_save_to_txt(host),
            llm_save_to_txt=runtime_llm_save_to_txt(host),
        ),
    )


def sync_runtime_tags_state(host: Any) -> None:
    folder_meta = list(getattr(host, "top_tags", []) or [])
    custom_tags = list(getattr(host, "custom_tags", []) or [])
    tagger_tags = list(getattr(host, "tagger_tags", []) or [])
    nl_tags = list(getattr(host, "nl_pages", []) or [])
    host._replace_runtime_state_section(
        "tags",
        state_projection.build_tags_state(
            folder_meta=folder_meta,
            custom=custom_tags,
            tagger=tagger_tags,
            nl=nl_tags,
            translations=build_tag_translations(
                host,
                folder_meta,
                custom_tags,
                tagger_tags,
                nl_tags,
            ),
        ),
    )
