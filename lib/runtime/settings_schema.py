# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from lib.core.settings import DEFAULT_APP_SETTINGS


SETTINGS_GROUPS: Dict[str, List[str]] = {
    "llm": [
        "llm_provider",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "llm_system_prompt",
        "llm_user_prompt_template",
        "llm_custom_prompt_template",
        "llm_skip_nsfw_on_batch",
        "llm_use_gray_mask",
        "llm_input_repeat_count",
        "llm_temperature",
        "llm_top_p",
        "llm_thinking_mode",
        "llm_max_image_dimension",
    ],
    "llama_cpp": [
        "llama_cpp_base_url",
        "llama_cpp_api_key",
        "llama_cpp_model_alias",
        "llama_cpp_model_path",
        "llama_cpp_n_ctx",
        "llama_cpp_n_threads",
        "llama_cpp_n_gpu_layers",
        "llama_cpp_max_tokens",
        "llama_cpp_temperature",
        "llama_cpp_top_p",
        "llama_cpp_top_k",
        "llama_cpp_min_p",
        "llama_cpp_presence_penalty",
        "llama_cpp_repeat_penalty",
        "llama_cpp_chat_format",
        "llama_cpp_local_files_only",
        "llama_cpp_mmproj_path",
        "llama_cpp_enable_vision",
        "llama_cpp_server_autostart",
        "llama_cpp_server_exe",
        "llama_cpp_server_workers",
        "llama_cpp_server_start_timeout",
    ],
    "image_process": [
        "image_process_worker",
        "image_process_model",
        "image_process_prompt_template",
        "image_process_base_url",
        "image_process_steps",
        "image_process_guidance_scale",
        "image_process_max_dimension",
        "image_process_seed",
        "image_process_local_files_only",
        "image_process_allow_full_model_fallback",
        "image_process_gguf_filename",
        "image_process_server_autostart",
        "image_process_server_exe",
        "image_process_server_start_timeout",
        "image_process_server_args_extra",
        "image_process_diffusion_model_path",
        "image_process_vae_path",
        "image_process_llm_path",
    ],
    "workers": [
        "tagger_worker",
        "unmask_worker",
        "mask_text_worker",
        "detect_text_worker",
        "tagger_model",
        "mask_remover_mode",
    ],
    "text": [
        "general_threshold",
        "general_mcut_enabled",
        "character_threshold",
        "character_mcut_enabled",
        "drop_overlap",
        "english_force_lowercase",
        "text_auto_remove_empty_lines",
        "text_auto_format",
        "text_auto_save",
        "batch_to_txt_mode",
        "batch_to_txt_folder_trigger",
        "default_custom_tags",
        "char_tag_blacklist_words",
        "char_tag_whitelist_words",
    ],
    "mask": [
        "mask_default_alpha",
        "mask_default_format",
        "mask_reverse",
        "mask_save_map_file",
        "mask_only_output_map",
        "mask_batch_only_if_has_background_tag",
        "mask_batch_detect_text_enabled",
        "mask_delete_npz_on_move",
        "mask_padding",
        "mask_blur_radius",
        "mask_bg_shrink_size",
        "mask_bg_blur_radius",
        "mask_bg_min_alpha",
        "mask_text_shrink_size",
        "mask_text_blur_radius",
        "mask_text_min_alpha",
        "mask_batch_skip_once_processed",
        "mask_batch_min_foreground_ratio",
        "mask_batch_max_foreground_ratio",
        "mask_batch_skip_if_scenery_tag",
        "mask_ocr_max_candidates",
        "mask_ocr_heat_threshold",
        "mask_ocr_box_threshold",
        "mask_ocr_unclip_ratio",
        "mask_text_alpha",
    ],
    "ui": [
        "last_open_dir",
        "startup_defer_worker_scan",
        "tokenizer_local_only",
        "tokenizer_retry_on_failure",
        "ui_language",
        "ui_theme",
    ],
}

SETTINGS_ENUMS: Dict[str, List[Any]] = {
    "ui_language": ["zh_tw", "en"],
    "ui_theme": ["light"],
    "batch_to_txt_mode": ["append", "replace"],
    "mask_default_format": ["webp", "png"],
}


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _group_for_key(key: str) -> str:
    for group_name, keys in SETTINGS_GROUPS.items():
        if key in keys:
            return group_name
    return "other"


def build_settings_schema(current_settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    current_settings = current_settings or {}
    fields: List[Dict[str, Any]] = []

    for key, default_value in DEFAULT_APP_SETTINGS.items():
        current_value = current_settings.get(key, default_value)
        field: Dict[str, Any] = {
            "key": key,
            "group": _group_for_key(key),
            "type": _infer_type(default_value),
            "default": default_value,
            "value": current_value,
        }
        options = SETTINGS_ENUMS.get(key)
        if options:
            field["options"] = list(options)
        fields.append(field)

    return {
        "version": "0.1",
        "field_count": len(fields),
        "groups": sorted({field["group"] for field in fields}),
        "fields": fields,
    }
