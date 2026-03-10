# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import lib.runtime.control_plane as control_plane
import lib.runtime.task_service as task_service


def command_open_directory(host: Any) -> None:
    host.open_directory()


def command_set_root_dir(host: Any, dir_path: str) -> Dict[str, Any]:
    dir_path = str(dir_path or "").strip()
    if not dir_path:
        raise ValueError("dir_path is required")
    if not os.path.isdir(dir_path):
        raise ValueError(f"directory does not exist: {dir_path}")

    host.root_dir_path = dir_path
    new_cfg = host._runtime_settings_dict()
    new_cfg["last_open_dir"] = dir_path
    host._replace_runtime_settings_dict(new_cfg, persist=True)

    host.filter_active = False
    if hasattr(host, "filter_input") and host.filter_input is not None:
        host.filter_input.clear()
    host.all_image_files = []
    host.filtered_image_files = []
    host._update_runtime_state_section(
        "selection",
        {
            "root_dir_path": dir_path,
            "filter_active": False,
            "filter_query": "",
            "all_image_paths": [],
            "filtered_image_paths": [],
        },
    )
    host.refresh_file_list()
    host._sync_runtime_settings_state()
    host._sync_runtime_selection_state()
    host._sync_runtime_controls_state()
    return host._selection_command_result("set_root_dir", root_dir_path=dir_path)


def command_apply_filter(
    host: Any,
    query: Optional[str] = None,
    use_tags: Optional[bool] = None,
    use_text: Optional[bool] = None,
) -> Dict[str, Any]:
    selection_updates: Dict[str, Any] = {}
    control_updates: Dict[str, Any] = {}
    normalized_query = str(query) if query is not None else None
    normalized_use_tags = bool(use_tags) if use_tags is not None else None
    normalized_use_text = bool(use_text) if use_text is not None else None
    if query is not None and hasattr(host, "filter_input"):
        host.filter_input.setText(normalized_query)
    if query is not None:
        selection_updates["filter_query"] = normalized_query
        control_updates["filter_query"] = normalized_query
    if use_tags is not None and hasattr(host, "chk_filter_tags"):
        host.chk_filter_tags.setChecked(normalized_use_tags)
    if use_tags is not None:
        selection_updates["filter_tags"] = normalized_use_tags
        control_updates["filter_tags"] = normalized_use_tags
    if use_text is not None and hasattr(host, "chk_filter_text"):
        host.chk_filter_text.setChecked(normalized_use_text)
    if use_text is not None:
        selection_updates["filter_text"] = normalized_use_text
        control_updates["filter_text"] = normalized_use_text
    if selection_updates:
        host._update_runtime_state_section("selection", selection_updates)
    if control_updates:
        host._update_runtime_state_section("controls", control_updates)
    host.apply_filter()
    host._sync_runtime_controls_state()
    return host._selection_command_result(
        "apply_filter",
        query=host._runtime_selection_value("filter_query", ""),
        match_count=len(host._runtime_loaded_image_paths()),
    )


def command_clear_filter(host: Any) -> Dict[str, Any]:
    host.clear_filter()
    host._sync_runtime_controls_state()
    return host._selection_command_result("clear_filter", match_count=len(host._runtime_loaded_image_paths()))


def command_prev_image(host: Any) -> Dict[str, Any]:
    host.prev_image()
    return host._selection_command_result("prev_image")


def command_next_image(host: Any) -> Dict[str, Any]:
    host.next_image()
    return host._selection_command_result("next_image")


def command_first_image(host: Any) -> Dict[str, Any]:
    host.first_image()
    return host._selection_command_result("first_image")


def command_last_image(host: Any) -> Dict[str, Any]:
    host.last_image()
    return host._selection_command_result("last_image")


def command_jump_to_index(host: Any, index: int) -> Dict[str, Any]:
    if hasattr(host, "index_input") and host.index_input is not None:
        host.index_input.setText(str(index))
    host.jump_to_index(index=index)
    return host._selection_command_result("jump_to_index", requested_index=int(index))


def command_set_view_mode(host: Any, index: int) -> Dict[str, Any]:
    normalized_index = int(index)
    host.current_view_mode = normalized_index
    if hasattr(host, "cb_view_mode") and host.cb_view_mode is not None:
        host.cb_view_mode.setCurrentIndex(normalized_index)
        host._sync_runtime_controls_state()
        host._sync_runtime_ui_state()
        return host._ui_command_result("set_view_mode", view_mode=normalized_index)
    host._sync_runtime_ui_state()
    host._sync_runtime_controls_state()
    return host._ui_command_result("set_view_mode", view_mode=normalized_index)


def command_set_active_tab(
    host: Any,
    index: Optional[int] = None,
    tab_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not hasattr(host, "tabs") or host.tabs is None:
        return host._ui_command_result("set_active_tab", completed=False, reason="tabs_unavailable")
    if index is None:
        tab_map = {
            "tags_tab": 0,
            "nl_tab": 1,
            "image_process_tab": 2,
        }
        index = tab_map.get(str(tab_id or "").strip(), host.tabs.currentIndex())
    host.tabs.setCurrentIndex(int(index))
    host._sync_runtime_ui_state()
    return host._ui_command_result("set_active_tab", active_tab_index=int(index), tab_id=tab_id)


def command_set_control_value(host: Any, control_id: str, value: Any) -> Dict[str, Any]:
    control_id = str(control_id or "").strip()
    if not control_id:
        raise ValueError("control_id is required")

    control_updates: Dict[str, Any] = {}
    selection_updates: Dict[str, Any] = {}

    if control_id == "filter_tags" and hasattr(host, "chk_filter_tags") and host.chk_filter_tags is not None:
        host.chk_filter_tags.setChecked(bool(value))
        host._sync_runtime_selection_state()
        control_updates["filter_tags"] = bool(value)
        selection_updates["filter_tags"] = bool(value)
    elif control_id == "filter_text" and hasattr(host, "chk_filter_text") and host.chk_filter_text is not None:
        host.chk_filter_text.setChecked(bool(value))
        host._sync_runtime_selection_state()
        control_updates["filter_text"] = bool(value)
        selection_updates["filter_text"] = bool(value)
    elif control_id == "view_mode":
        host.command_set_view_mode(int(value))
        control_updates["view_mode"] = int(value)
    elif control_id == "tagger_save_to_txt" and hasattr(host, "chk_tags_save_txt") and host.chk_tags_save_txt is not None:
        host.chk_tags_save_txt.setChecked(bool(value))
        control_updates["tagger_save_to_txt"] = bool(value)
    elif control_id == "llm_save_to_txt" and hasattr(host, "chk_llm_save_txt") and host.chk_llm_save_txt is not None:
        host.chk_llm_save_txt.setChecked(bool(value))
        control_updates["llm_save_to_txt"] = bool(value)
    elif control_id == "filter_query" and hasattr(host, "filter_input") and host.filter_input is not None:
        host.filter_input.setText(str(value or ""))
        host._sync_runtime_selection_state()
        control_updates["filter_query"] = str(value or "")
        selection_updates["filter_query"] = str(value or "")
    elif control_id == "current_index":
        host.command_jump_to_index(int(value))
        control_updates["current_index"] = int(value)
    elif control_id == "tagger_save_to_txt":
        control_updates["tagger_save_to_txt"] = bool(value)
    elif control_id == "llm_save_to_txt":
        control_updates["llm_save_to_txt"] = bool(value)
    elif control_id == "filter_query":
        control_updates["filter_query"] = str(value or "")
        selection_updates["filter_query"] = str(value or "")
    elif control_id == "filter_tags":
        control_updates["filter_tags"] = bool(value)
        selection_updates["filter_tags"] = bool(value)
    elif control_id == "filter_text":
        control_updates["filter_text"] = bool(value)
        selection_updates["filter_text"] = bool(value)
    else:
        raise ValueError(f"unsupported control id: {control_id}")

    if control_updates:
        host._update_runtime_state_section("controls", control_updates)
    if selection_updates:
        host._update_runtime_state_section("selection", selection_updates)
    host._sync_runtime_controls_state()
    return host._ui_command_result("set_control_value", control_id=control_id, value=value)


def command_delete_current_image(host: Any, confirm: bool = False) -> Dict[str, Any]:
    result = host.delete_current_image(require_confirmation=not bool(confirm))
    if result.get("deleted"):
        host._get_runtime_event_bus().emit("image.deleted", result)
    return result


def command_add_custom_tag(host: Any, tag: Optional[str] = None) -> Dict[str, Any]:
    if tag is None:
        host.add_custom_tag_dialog()
        return {
            "added": False,
            "reason": "dialog",
        }
    normalized_tag = str(tag)
    added = bool(host.add_custom_tag(normalized_tag))
    result = {
        "added": added,
        "tag": normalized_tag,
        "current_folder_path": getattr(host, "current_folder_path", ""),
        "custom_tags": list(getattr(host, "custom_tags", []) or []),
    }
    if added:
        host._get_runtime_event_bus().emit("tags.custom.added", result)
    return result


def command_use_default_prompt(host: Any) -> Dict[str, Any]:
    host.use_default_prompt()
    host._sync_runtime_ui_state()
    host._sync_runtime_content_state()
    return host._content_command_result("use_default_prompt")


def command_use_custom_prompt(host: Any) -> Dict[str, Any]:
    host.use_custom_prompt()
    host._sync_runtime_ui_state()
    host._sync_runtime_content_state()
    return host._content_command_result("use_custom_prompt")


def command_use_default_image_prompt(host: Any) -> Dict[str, Any]:
    host.use_default_image_prompt()
    host._sync_runtime_content_state()
    return host._content_command_result("use_default_image_prompt")


def command_set_prompt_text(host: Any, text: str) -> Dict[str, Any]:
    normalized_text = str(text or "")
    if hasattr(host, "prompt_edit") and host.prompt_edit is not None:
        host.prompt_edit.setPlainText(normalized_text)
    else:
        host._update_runtime_state_section("content", {"prompt_text": normalized_text})
    host._sync_runtime_content_state()
    return host._content_command_result("set_prompt_text")


def command_set_image_process_prompt_text(host: Any, text: str) -> Dict[str, Any]:
    normalized_text = str(text or "")
    if hasattr(host, "img_prompt_edit") and host.img_prompt_edit is not None:
        host.img_prompt_edit.setPlainText(normalized_text)
    else:
        host._update_runtime_state_section("content", {"image_process_prompt_text": normalized_text})
    host._sync_runtime_content_state()
    return host._content_command_result("set_image_process_prompt_text")


def command_set_txt_content(host: Any, text: str) -> Dict[str, Any]:
    normalized_text = str(text or "")
    if hasattr(host, "txt_edit") and host.txt_edit is not None:
        host.txt_edit.setPlainText(normalized_text)
    else:
        host._update_runtime_state_section("content", {"txt_content": normalized_text})
    host._sync_runtime_content_state()
    return host._content_command_result("set_txt_content")


def command_open_find_replace(
    host: Any,
    find_text: Optional[str] = None,
    replace_text: str = "",
    scope_all: bool = False,
    case_sensitive: bool = False,
    regex: bool = False,
) -> Dict[str, Any]:
    if find_text is None:
        host.open_find_replace()
        return {
            "executed": False,
            "reason": "dialog",
        }
    result = host.run_find_replace(
        find_text=find_text,
        replace_text=replace_text,
        scope_all=bool(scope_all),
        case_sensitive=bool(case_sensitive),
        regex=bool(regex),
    )
    host._get_runtime_event_bus().emit("editor.find_replace.completed", result)
    return result


def command_editor_undo(host: Any) -> None:
    if hasattr(host, "txt_edit") and host.txt_edit is not None:
        host.txt_edit.undo()
        host._sync_runtime_content_state()


def command_editor_redo(host: Any) -> None:
    if hasattr(host, "txt_edit") and host.txt_edit is not None:
        host.txt_edit.redo()
        host._sync_runtime_content_state()


def command_prev_nl_page(host: Any) -> None:
    host.prev_nl_page()


def command_next_nl_page(host: Any) -> None:
    host.next_nl_page()


def command_start_runtime_bridge(host: Any, host_name: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
    return control_plane.command_start_runtime_bridge(host, bind_host=host_name, port=port)


def command_stop_runtime_bridge(host: Any) -> Dict[str, Any]:
    return control_plane.command_stop_runtime_bridge(host)


def command_scan_workers(host: Any) -> Dict[str, Any]:
    return control_plane.command_scan_workers(host)


def command_run_tagger(host: Any, image_paths: List[str]) -> None:
    host._start_named_runtime_task("tagger", image_paths)


def command_run_tagger_loaded(host: Any) -> None:
    host.command_run_tagger(host._get_loaded_image_paths())


def command_run_llm(
    host: Any,
    image_paths: List[str],
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    host._start_named_runtime_task(
        "llm",
        image_paths,
        extra=task_service.build_llm_task_extra(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        ),
    )


def command_run_llm_loaded(
    host: Any,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    host.command_run_llm(
        host._get_loaded_image_paths(),
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )


def command_run_image_process(host: Any, image_paths: List[str], edit_prompt: Optional[str] = None) -> None:
    host._start_named_runtime_task(
        "image_process",
        image_paths,
        extra=task_service.build_image_process_task_extra(edit_prompt=edit_prompt),
    )


def command_run_image_process_loaded(host: Any, edit_prompt: Optional[str] = None) -> None:
    host.command_run_image_process(host._get_loaded_image_paths(), edit_prompt=edit_prompt)


def command_run_unmask(host: Any, image_paths: List[str]) -> None:
    host._start_named_runtime_task("unmask", image_paths)


def command_run_unmask_loaded(host: Any) -> None:
    host.command_run_unmask(host._get_loaded_image_paths())


def command_run_mask_text(host: Any, image_paths: List[str]) -> None:
    host._start_named_runtime_task("mask_text", image_paths)


def command_run_mask_text_loaded(host: Any) -> None:
    host.command_run_mask_text(host._get_loaded_image_paths())


def command_run_restore(host: Any, image_paths: List[str]) -> None:
    host._start_named_runtime_task("restore", image_paths)


def command_run_restore_loaded(host: Any) -> None:
    host.command_run_restore(host._get_loaded_image_paths())


def command_smoke_command(host: Any, mode: Optional[str] = None) -> Dict[str, Any]:
    registry = host._get_command_registry()
    return {
        "commands": registry.list_commands(access_mode=mode),
        "command_access_modes": registry.get_command_access_modes(),
        "command_metadata": registry.get_command_metadata(access_mode=mode),
        "task_status": host.get_task_status(),
        "event_count": len(host._get_runtime_event_bus().history()),
        "state": host.get_runtime_state(),
        "bridge": host.get_runtime_bridge_status(),
    }
