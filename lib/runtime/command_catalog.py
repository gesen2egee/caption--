# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Iterable, Tuple


RUNTIME_COMMAND_BINDINGS: Tuple[Tuple[str, str], ...] = (
    ("app.get_runtime_events", "get_runtime_events"),
    ("app.get_event_profiles", "get_runtime_event_profiles"),
    ("app.get_runtime_state", "get_runtime_state"),
    ("app.shutdown", "command_shutdown_app"),
    ("app.get_agent_manifest", "get_agent_manifest"),
    ("app.get_ui_spec", "get_ui_spec"),
    ("app.get_ui_spec_override", "get_ui_spec_override"),
    ("app.update_ui_spec", "update_ui_spec"),
    ("app.replace_ui_spec_override", "replace_ui_spec_override"),
    ("app.patch_ui_node", "patch_ui_node"),
    ("app.reset_ui_spec", "reset_ui_spec"),
    ("app.get_task_status", "get_task_status"),
    ("app.get_capabilities", "get_runtime_capabilities"),
    ("backend.list_reloadables", "command_list_reloadable_runtime_services"),
    ("backend.get_reload_policy", "command_get_reload_policy"),
    ("backend.reload_services", "command_reload_runtime_services"),
    ("backend.watch_reload_start", "command_start_runtime_service_watch"),
    ("backend.watch_reload_status", "get_runtime_service_watch_status"),
    ("backend.watch_reload_stop", "command_stop_runtime_service_watch"),
    ("bridge.get_status", "get_runtime_bridge_status"),
    ("bridge.start", "command_start_runtime_bridge"),
    ("bridge.stop", "command_stop_runtime_bridge"),
    ("settings.get", "get_runtime_settings"),
    ("settings.schema", "get_runtime_settings_schema"),
    ("settings.update", "update_runtime_settings"),
    ("workers.list", "get_runtime_workers"),
    ("workers.services_status", "get_worker_services_status"),
    ("workers.scan", "command_scan_workers"),
    ("workers.services_reload", "command_reload_worker_services"),
    ("workers.services_stop", "command_stop_worker_services"),
    ("selection.open_directory", "command_open_directory"),
    ("selection.set_root_dir", "command_set_root_dir"),
    ("selection.apply_filter", "command_apply_filter"),
    ("selection.clear_filter", "command_clear_filter"),
    ("selection.prev", "command_prev_image"),
    ("selection.next", "command_next_image"),
    ("selection.first", "command_first_image"),
    ("selection.last", "command_last_image"),
    ("selection.jump_to_index", "command_jump_to_index"),
    ("ui.set_view_mode", "command_set_view_mode"),
    ("ui.set_active_tab", "command_set_active_tab"),
    ("ui.set_control_value", "command_set_control_value"),
    ("image.delete_current", "command_delete_current_image"),
    ("action.run_tagger_current", "command_run_tagger_current"),
    ("action.run_llm_current", "command_run_llm_current"),
    ("action.run_image_process_current", "command_run_image_process_current"),
    ("action.run_unmask_current", "command_run_unmask_current"),
    ("action.run_mask_text_current", "command_run_mask_text_current"),
    ("action.run_restore_current", "command_run_restore_current"),
    ("action.run_stroke_eraser_current", "command_run_stroke_eraser_current"),
    ("tags.add_custom", "command_add_custom_tag"),
    ("prompt.use_default", "command_use_default_prompt"),
    ("prompt.use_custom", "command_use_custom_prompt"),
    ("prompt.use_default_image_process", "command_use_default_image_prompt"),
    ("content.set_prompt_text", "command_set_prompt_text"),
    ("content.set_image_process_prompt_text", "command_set_image_process_prompt_text"),
    ("content.set_txt_content", "command_set_txt_content"),
    ("editor.find_replace", "command_open_find_replace"),
    ("editor.undo", "command_editor_undo"),
    ("editor.redo", "command_editor_redo"),
    ("nl.prev_page", "command_prev_nl_page"),
    ("nl.next_page", "command_next_nl_page"),
    ("task.run_tagger", "command_run_tagger"),
    ("task.run_tagger_loaded", "command_run_tagger_loaded"),
    ("task.run_llm", "command_run_llm"),
    ("task.run_llm_loaded", "command_run_llm_loaded"),
    ("task.run_image_process", "command_run_image_process"),
    ("task.run_image_process_loaded", "command_run_image_process_loaded"),
    ("task.run_unmask", "command_run_unmask"),
    ("task.run_unmask_loaded", "command_run_unmask_loaded"),
    ("task.run_mask_text", "command_run_mask_text"),
    ("task.run_mask_text_loaded", "command_run_mask_text_loaded"),
    ("task.run_restore", "command_run_restore"),
    ("task.run_restore_loaded", "command_run_restore_loaded"),
    ("batch.run_tagger", "command_run_batch_tagger"),
    ("batch.run_llm", "command_run_batch_llm"),
    ("batch.run_image_process", "command_run_batch_image_process"),
    ("batch.run_unmask", "command_run_batch_unmask"),
    ("batch.run_mask_text", "command_run_batch_mask_text"),
    ("batch.run_restore", "command_run_batch_restore"),
    ("task.cancel", "stop_current_task"),
    ("test.smoke_command", "command_smoke_command"),
    ("test.run_runtime_regression", "command_run_runtime_regression"),
)


def register_runtime_commands(host: Any) -> int:
    count = 0
    for command_name, attribute_name in RUNTIME_COMMAND_BINDINGS:
        host._register_runtime_command(command_name, getattr(host, attribute_name))
        count += 1
    return count


def list_runtime_command_bindings() -> Iterable[Tuple[str, str]]:
    return tuple(RUNTIME_COMMAND_BINDINGS)
