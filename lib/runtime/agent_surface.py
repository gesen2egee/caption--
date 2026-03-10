# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List


SAFE_AGENT_COMMANDS = [
    "app.get_agent_manifest",
    "app.get_capabilities",
    "app.get_event_profiles",
    "app.get_runtime_events",
    "app.get_runtime_state",
    "app.shutdown",
    "app.get_task_status",
    "app.get_ui_spec",
    "app.get_ui_spec_override",
    "backend.get_reload_policy",
    "batch.run_image_process",
    "batch.run_llm",
    "batch.run_mask_text",
    "batch.run_restore",
    "batch.run_tagger",
    "batch.run_unmask",
    "bridge.get_status",
    "content.set_image_process_prompt_text",
    "content.set_prompt_text",
    "content.set_txt_content",
    "editor.find_replace",
    "editor.redo",
    "editor.undo",
    "nl.next_page",
    "nl.prev_page",
    "prompt.use_custom",
    "prompt.use_default",
    "prompt.use_default_image_process",
    "selection.apply_filter",
    "selection.clear_filter",
    "selection.first",
    "selection.jump_to_index",
    "selection.last",
    "selection.next",
    "selection.prev",
    "selection.set_root_dir",
    "settings.get",
    "settings.schema",
    "tags.add_custom",
    "test.smoke_command",
    "task.cancel",
    "ui.set_active_tab",
    "ui.set_control_value",
    "ui.set_view_mode",
    "workers.list",
    "workers.services_status",
    "workers.scan",
    "action.run_image_process_current",
    "action.run_llm_current",
    "action.run_mask_text_current",
    "action.run_restore_current",
    "action.run_stroke_eraser_current",
    "action.run_tagger_current",
    "action.run_unmask_current",
]

DEVELOPMENT_ONLY_AGENT_COMMANDS = [
    "app.patch_ui_node",
    "app.replace_ui_spec_override",
    "app.reset_ui_spec",
    "app.update_ui_spec",
    "backend.list_reloadables",
    "backend.reload_services",
    "backend.watch_reload_start",
    "backend.watch_reload_status",
    "backend.watch_reload_stop",
    "bridge.start",
    "bridge.stop",
    "image.delete_current",
    "selection.open_directory",
    "settings.update",
    "workers.services_reload",
    "workers.services_stop",
    "task.run_image_process",
    "task.run_image_process_loaded",
    "task.run_llm",
    "task.run_llm_loaded",
    "task.run_mask_text",
    "task.run_mask_text_loaded",
    "task.run_restore",
    "task.run_restore_loaded",
    "task.run_tagger",
    "task.run_tagger_loaded",
    "task.run_unmask",
    "task.run_unmask_loaded",
    "test.run_runtime_regression",
]

RECOMMENDED_SAFE_AGENT_COMMANDS = [
    "selection.set_root_dir",
    "selection.apply_filter",
    "selection.next",
    "selection.prev",
    "action.run_tagger_current",
    "action.run_llm_current",
    "action.run_image_process_current",
    "action.run_stroke_eraser_current",
    "batch.run_tagger",
    "batch.run_llm",
    "editor.find_replace",
    "app.get_ui_spec",
    "app.get_runtime_state",
    "app.get_runtime_events",
]

RECOMMENDED_DEVELOPMENT_AGENT_COMMANDS = RECOMMENDED_SAFE_AGENT_COMMANDS + [
    "backend.reload_services",
    "image.delete_current",
    "settings.update",
    "app.patch_ui_node",
]

COMMAND_CATEGORY_LABELS = {
    "app": "runtime",
    "backend": "backend-runtime",
    "bridge": "bridge",
    "settings": "settings",
    "workers": "workers",
    "selection": "selection",
    "ui": "ui-control",
    "image": "image",
    "action": "single-image",
    "batch": "batch",
    "task": "low-level-task",
    "editor": "editor",
    "tags": "tags",
    "prompt": "prompt",
    "content": "content",
    "nl": "nl",
    "test": "diagnostic",
}

DESTRUCTIVE_COMMANDS = {
    "image.delete_current",
}

REQUIRES_CONFIRMATION_COMMANDS = {
    "app.shutdown",
    "app.reset_ui_spec",
    "image.delete_current",
    "action.run_restore_current",
    "batch.run_restore",
}

MUTATES_UI_COMMANDS = {
    "app.patch_ui_node",
    "app.replace_ui_spec_override",
    "app.reset_ui_spec",
    "app.update_ui_spec",
}

MUTATES_SETTINGS_COMMANDS = {
    "settings.update",
}

MUTATES_RUNTIME_COMMANDS = {
    "app.shutdown",
    "backend.reload_services",
    "backend.watch_reload_start",
    "backend.watch_reload_stop",
    "bridge.start",
    "bridge.stop",
    "task.cancel",
    "workers.scan",
    "workers.services_reload",
    "workers.services_stop",
}

FILE_WRITING_COMMAND_PREFIXES = (
    "action.run_",
    "batch.run_",
    "task.run_",
)

FILE_WRITING_COMMANDS = {
    "editor.find_replace",
    "image.delete_current",
    "settings.update",
    "tags.add_custom",
}

LONG_RUNNING_COMMAND_PREFIXES = (
    "action.run_",
    "batch.run_",
    "task.run_",
)

LONG_RUNNING_COMMANDS = {
    "backend.watch_reload_start",
    "test.run_runtime_regression",
    "workers.scan",
    "workers.services_reload",
    "workers.services_stop",
}

READ_ONLY_COMMAND_PREFIXES = (
    "app.get_",
    "bridge.get_",
)

READ_ONLY_COMMANDS = {
    "backend.list_reloadables",
    "backend.watch_reload_status",
    "backend.get_reload_policy",
    "settings.get",
    "settings.schema",
    "test.smoke_command",
    "workers.list",
    "workers.services_status",
}


def get_command_access_modes(name: str) -> List[str]:
    if name in SAFE_AGENT_COMMANDS:
        return ["safe", "development", "internal"]
    if name in DEVELOPMENT_ONLY_AGENT_COMMANDS:
        return ["development", "internal"]
    return ["internal"]


def _command_category(name: str) -> str:
    return str(name or "").split(".", 1)[0].strip().lower() or "runtime"


def _command_category_label(name: str) -> str:
    category = _command_category(name)
    return COMMAND_CATEGORY_LABELS.get(category, category)


def build_command_metadata(name: str) -> Dict[str, Any]:
    normalized_name = str(name or "").strip()
    writes_files = (
        normalized_name in FILE_WRITING_COMMANDS
        or normalized_name.startswith(FILE_WRITING_COMMAND_PREFIXES)
    )
    destructive = normalized_name in DESTRUCTIVE_COMMANDS
    requires_confirmation = normalized_name in REQUIRES_CONFIRMATION_COMMANDS
    mutates_ui = normalized_name in MUTATES_UI_COMMANDS
    mutates_settings = normalized_name in MUTATES_SETTINGS_COMMANDS
    mutates_runtime = normalized_name in MUTATES_RUNTIME_COMMANDS
    long_running = (
        normalized_name in LONG_RUNNING_COMMANDS
        or normalized_name.startswith(LONG_RUNNING_COMMAND_PREFIXES)
    )
    read_only = (
        normalized_name in READ_ONLY_COMMANDS
        or normalized_name.startswith(READ_ONLY_COMMAND_PREFIXES)
    )

    notes: List[str] = []
    if read_only:
        notes.append("Read-only inspection command.")
    if long_running:
        notes.append("Starts or manages a longer-running task and emits task events.")
    if writes_files:
        notes.append("May create, overwrite, or move files on disk.")
    if destructive:
        notes.append("Destructive file operation. Keep explicit confirmation enabled.")
    if mutates_settings:
        notes.append("Persists app settings and changes future runs.")
    if mutates_ui:
        notes.append("Mutates the runtime UI spec or override state.")
    if mutates_runtime:
        notes.append("Changes runtime infrastructure or execution state.")
    if requires_confirmation and not destructive:
        notes.append("Usually expects an explicit confirmation flag before high-impact changes.")

    risk_level = "low"
    if destructive:
        risk_level = "high"
    elif writes_files or mutates_ui or mutates_settings or mutates_runtime or requires_confirmation or long_running:
        risk_level = "medium"

    return {
        "category": _command_category(normalized_name),
        "category_label": _command_category_label(normalized_name),
        "access_modes": get_command_access_modes(normalized_name),
        "read_only": read_only,
        "writes_files": writes_files,
        "destructive": destructive,
        "requires_confirmation": requires_confirmation,
        "mutates_ui": mutates_ui,
        "mutates_settings": mutates_settings,
        "mutates_runtime": mutates_runtime,
        "long_running": long_running,
        "risk_level": risk_level,
        "notes": notes,
    }


def build_agent_surface(mode: str) -> Dict[str, Any]:
    mode_name = str(mode or "").strip().lower() or "development"
    if mode_name not in {"safe", "development"}:
        raise ValueError(f"unknown agent mode: {mode_name}")

    if mode_name == "safe":
        allowed_commands = list(SAFE_AGENT_COMMANDS)
        recommended_commands = list(RECOMMENDED_SAFE_AGENT_COMMANDS)
        restricted_commands = list(DEVELOPMENT_ONLY_AGENT_COMMANDS)
        ui_mutation_allowed = False
        settings_mutation_allowed = False
        destructive_file_ops_allowed = False
        skills_note = "Prefer the control profile. This surface excludes destructive file ops, settings mutation, and UI-spec mutation."
    else:
        allowed_commands = list(dict.fromkeys(SAFE_AGENT_COMMANDS + DEVELOPMENT_ONLY_AGENT_COMMANDS))
        recommended_commands = list(RECOMMENDED_DEVELOPMENT_AGENT_COMMANDS)
        restricted_commands = []
        ui_mutation_allowed = True
        settings_mutation_allowed = True
        destructive_file_ops_allowed = True
        skills_note = "Development surface. Prefer the control profile by default and escalate to diagnostic only for debugging."

    return {
        "mode_name": mode_name,
        "allowed_commands": allowed_commands,
        "recommended_commands": recommended_commands,
        "restricted_commands": restricted_commands,
        "ui_mutation_allowed": ui_mutation_allowed,
        "settings_mutation_allowed": settings_mutation_allowed,
        "destructive_file_ops_allowed": destructive_file_ops_allowed,
        "skills_note": skills_note,
    }


def get_command_examples() -> Dict[str, Dict[str, Any]]:
    return {
        "app.shutdown": {
            "kwargs": {"confirm": True},
        },
        "backend.list_reloadables": {
            "kwargs": {},
        },
        "backend.get_reload_policy": {
            "kwargs": {"changed_path": "lib/workers/image_restore_raw.py"},
        },
        "backend.reload_services": {
            "kwargs": {"names": ["selection", "processing"]},
        },
        "backend.watch_reload_start": {
            "kwargs": {"interval_seconds": 1.0},
        },
        "backend.watch_reload_status": {
            "kwargs": {},
        },
        "backend.watch_reload_stop": {
            "kwargs": {},
        },
        "selection.set_root_dir": {
            "kwargs": {"dir_path": "E:\\images\\set"},
        },
        "workers.services_reload": {
            "kwargs": {"category": "LLM"},
        },
        "workers.services_stop": {
            "kwargs": {"worker_name": "llm_llama_cpp_local"},
        },
        "editor.find_replace": {
            "kwargs": {
                "find_text": "old tag",
                "replace_text": "new tag",
                "scope_all": True,
                "case_sensitive": False,
                "regex": False,
            },
        },
        "tags.add_custom": {
            "kwargs": {"tag": "dramatic lighting"},
        },
        "image.delete_current": {
            "kwargs": {"confirm": True},
        },
        "action.run_llm_current": {
            "kwargs": {
                "user_prompt": "describe image using {tags}",
                "confirm_missing_tags": True,
            },
        },
        "action.run_image_process_current": {
            "kwargs": {"edit_prompt": "remove all text"},
        },
        "action.run_stroke_eraser_current": {
            "kwargs": {"mask_path": "E:\\masks\\stroke-mask.png"},
        },
        "batch.run_tagger": {
            "kwargs": {"save_to_txt": True, "delete_chars": False},
        },
        "batch.run_llm": {
            "kwargs": {
                "user_prompt": "describe image",
                "save_to_txt": True,
                "delete_chars": False,
                "confirm_character_prompt": True,
            },
        },
        "batch.run_restore": {
            "kwargs": {"confirm": True},
        },
        "test.run_runtime_regression": {
            "kwargs": {},
        },
    }
