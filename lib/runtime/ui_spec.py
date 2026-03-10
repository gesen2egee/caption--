# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


SPEC_TEXT = {
    "en": {
        "selection": "Selection",
        "navigation": "Navigation",
        "filter": "Filter",
        "filter_tags": "Tags",
        "filter_text": "Text",
        "view": "View",
        "view_original": "Original",
        "view_rgb": "RGB",
        "view_alpha": "Alpha",
        "preview": "Preview",
        "image_actions": "Image Actions",
        "prev": "Prev",
        "next": "Next",
        "delete": "Delete",
        "utility_actions": "Utility Actions",
        "unmask": "Unmask",
        "mask_text": "Mask Text",
        "restore": "Restore",
        "cancel": "Cancel",
        "workspace": "Workspace",
        "feature_tabs": "Feature Tabs",
        "tags": "Tags",
        "tag_actions": "Tag Actions",
        "tag_current": "Tag Current",
        "tag_loaded": "Tag Loaded Set",
        "save_to_txt": "Save To Txt",
        "add_custom_tag": "Add Custom Tag",
        "folder_meta": "Folder Meta",
        "custom_tags": "Custom Tags",
        "tagger_tags": "Tagger Tags",
        "prompt_nl": "Prompt / NL",
        "llm_actions": "LLM Actions",
        "run_current": "Run Current",
        "run_loaded": "Run Loaded Set",
        "default": "Default",
        "custom": "Custom",
        "pages": "Pages",
        "latest_nl": "Latest NL",
        "prompt_editor": "Prompt Editor",
        "image_process": "Image Process",
        "image_process_actions": "Image Process Actions",
        "reset_prompt": "Reset Prompt",
        "image_process_prompt": "Image Process Prompt",
        "text_editor": "Text Editor",
        "text_toolbar": "Text Toolbar",
        "find_replace": "Find / Replace",
        "undo": "Undo",
        "redo": "Redo",
        "caption_text": "Caption Text",
    },
    "zh_tw": {
        "selection": "選擇",
        "navigation": "導覽",
        "filter": "篩選",
        "filter_tags": "標籤",
        "filter_text": "文字",
        "view": "顯示",
        "view_original": "原圖",
        "view_rgb": "RGB",
        "view_alpha": "Alpha",
        "preview": "預覽",
        "image_actions": "圖片操作",
        "prev": "上一張",
        "next": "下一張",
        "delete": "刪除",
        "utility_actions": "工具操作",
        "unmask": "去背",
        "mask_text": "去文字",
        "restore": "還原",
        "cancel": "取消",
        "workspace": "工作區",
        "feature_tabs": "功能分頁",
        "tags": "標籤",
        "tag_actions": "標籤操作",
        "tag_current": "標註目前圖片",
        "tag_loaded": "標註已載入圖片",
        "save_to_txt": "寫入 txt",
        "add_custom_tag": "新增自訂標籤",
        "folder_meta": "資料夾標籤",
        "custom_tags": "自訂標籤",
        "tagger_tags": "Tagger 標籤",
        "prompt_nl": "提示 / NL",
        "llm_actions": "LLM 操作",
        "run_current": "執行目前圖片",
        "run_loaded": "執行已載入圖片",
        "default": "預設",
        "custom": "自訂",
        "pages": "頁面",
        "latest_nl": "最新 NL",
        "prompt_editor": "提示編輯器",
        "image_process": "修圖",
        "image_process_actions": "修圖操作",
        "reset_prompt": "重設提示",
        "image_process_prompt": "修圖提示",
        "text_editor": "文字編輯器",
        "text_toolbar": "文字工具列",
        "find_replace": "尋找 / 取代",
        "undo": "復原",
        "redo": "重做",
        "caption_text": "Caption 文字",
    },
}


def _spec_language(state: Dict[str, Any] | None) -> str:
    language = str((((state or {}).get("settings") or {}).get("ui_language") or "en")).strip().lower()
    if language.startswith("zh"):
        return "zh_tw"
    return "en"


def _spec_text(language: str, key: str) -> str:
    return str(SPEC_TEXT.get(language, SPEC_TEXT["en"]).get(key, SPEC_TEXT["en"].get(key, key)))


def strip_runtime_state(spec: Dict[str, Any] | None) -> Dict[str, Any]:
    spec = deepcopy(spec or {})
    spec.pop("state", None)
    return spec


def merge_ui_spec(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    result = deepcopy(base)
    override = strip_runtime_state(override)
    return _merge_dicts(result, override)


def patch_ui_spec_node(spec: Dict[str, Any], node_id: str, patch: Dict[str, Any]) -> bool:
    if not node_id:
        return False
    return _patch_node_in_place(spec, str(node_id), deepcopy(patch or {}))


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dicts(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _patch_node_in_place(node: Any, node_id: str, patch: Dict[str, Any]) -> bool:
    if isinstance(node, dict):
        if str(node.get("id", "")) == node_id:
            _merge_dicts(node, patch)
            return True
        for value in node.values():
            if isinstance(value, (dict, list)) and _patch_node_in_place(value, node_id, patch):
                return True
    elif isinstance(node, list):
        for item in node:
            if _patch_node_in_place(item, node_id, patch):
                return True
    return False


def build_legacy_ui_spec(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Return a JSON-serializable description of the current legacy layout.

    This is intentionally a migration artifact, not the final DSL. The goal is
    to preserve the current structure in a form that a future renderer can
    consume.
    """

    state = state or {}
    language = _spec_language(state)

    return {
        "version": "0.1",
        "layout": {
            "type": "split",
            "direction": "horizontal",
            "children": [
                {
                    "id": "left_panel",
                    "type": "panel",
                    "title": _spec_text(language, "selection"),
                    "sections": [
                        {
                            "id": "navigation_bar",
                            "type": "toolbar",
                            "title": _spec_text(language, "navigation"),
                            "controls": [
                                {"id": "current_index", "type": "index_input", "bind": "selection.current_index"},
                                {"type": "label", "bind": "selection.image_count"},
                                {"type": "label", "bind": "selection.current_image_path"},
                                {"id": "filter_query", "type": "filter_input", "label": _spec_text(language, "filter")},
                                {"type": "toggle", "id": "filter_tags", "label": _spec_text(language, "filter_tags")},
                                {"type": "toggle", "id": "filter_text", "label": _spec_text(language, "filter_text")},
                                {
                                    "type": "select",
                                    "id": "view_mode",
                                    "label": _spec_text(language, "view"),
                                    "options": [
                                        {"label": _spec_text(language, "view_original"), "value": 0},
                                        {"label": _spec_text(language, "view_rgb"), "value": 1},
                                        {"label": _spec_text(language, "view_alpha"), "value": 2},
                                    ],
                                },
                            ],
                        },
                        {
                            "id": "image_viewer",
                            "type": "image_preview",
                            "title": _spec_text(language, "preview"),
                            "bind": "selection.current_image_path",
                        },
                        {
                            "id": "image_actions",
                            "type": "toolbar",
                            "title": _spec_text(language, "image_actions"),
                            "controls": [
                                {"type": "button", "command": "selection.prev", "label": _spec_text(language, "prev")},
                                {"type": "button", "command": "selection.next", "label": _spec_text(language, "next")},
                                {"type": "button", "command": "image.delete_current", "label": _spec_text(language, "delete")},
                            ],
                        },
                        {
                            "id": "utility_actions",
                            "type": "toolbar",
                            "title": _spec_text(language, "utility_actions"),
                            "controls": [
                                {"type": "button", "command": "action.run_unmask_current", "label": _spec_text(language, "unmask")},
                                {"type": "button", "command": "action.run_mask_text_current", "label": _spec_text(language, "mask_text")},
                                {"type": "button", "command": "action.run_restore_current", "label": _spec_text(language, "restore")},
                                {"type": "button", "command": "task.cancel", "label": _spec_text(language, "cancel")},
                            ],
                        },
                    ],
                },
                {
                    "id": "right_panel",
                    "type": "split",
                    "title": _spec_text(language, "workspace"),
                    "direction": "vertical",
                    "children": [
                        {
                            "id": "feature_tabs",
                            "type": "tabs",
                            "title": _spec_text(language, "feature_tabs"),
                            "tabs": [
                                {
                                    "id": "tags_tab",
                                    "type": "tab",
                                    "title": _spec_text(language, "tags"),
                                    "sections": [
                                        {
                                            "id": "tag_actions",
                                            "type": "toolbar",
                                            "title": _spec_text(language, "tag_actions"),
                                            "controls": [
                                                {"type": "button", "command": "action.run_tagger_current", "label": _spec_text(language, "tag_current")},
                                                {"type": "button", "command": "batch.run_tagger", "mode": "batch", "label": _spec_text(language, "tag_loaded")},
                                                {"type": "toggle", "id": "tagger_save_to_txt", "label": _spec_text(language, "save_to_txt")},
                                                {"type": "button", "command": "tags.add_custom", "label": _spec_text(language, "add_custom_tag")},
                                            ],
                                        },
                                        {"id": "folder_meta_tags", "type": "tag_flow", "title": _spec_text(language, "folder_meta")},
                                        {"id": "custom_tags", "type": "tag_flow", "title": _spec_text(language, "custom_tags")},
                                        {"id": "tagger_tags", "type": "tag_flow", "title": _spec_text(language, "tagger_tags")},
                                    ],
                                },
                                {
                                    "id": "nl_tab",
                                    "type": "tab",
                                    "title": _spec_text(language, "prompt_nl"),
                                    "sections": [
                                        {
                                            "id": "llm_actions",
                                            "type": "toolbar",
                                            "title": _spec_text(language, "llm_actions"),
                                            "controls": [
                                                {"type": "button", "command": "action.run_llm_current", "label": _spec_text(language, "run_current")},
                                                {"type": "button", "command": "batch.run_llm", "mode": "batch", "label": _spec_text(language, "run_loaded")},
                                                {"type": "toggle", "id": "llm_save_to_txt", "label": _spec_text(language, "save_to_txt")},
                                                {"type": "button", "command": "prompt.use_default", "label": _spec_text(language, "default")},
                                                {"type": "button", "command": "prompt.use_custom", "label": _spec_text(language, "custom")},
                                            ],
                                        },
                                        {"id": "nl_pager", "type": "pager", "title": _spec_text(language, "pages")},
                                        {"id": "nl_result_tags", "type": "tag_flow", "title": _spec_text(language, "latest_nl")},
                                        {"id": "prompt_editor", "type": "text_area", "title": _spec_text(language, "prompt_editor")},
                                    ],
                                },
                                {
                                    "id": "image_process_tab",
                                    "type": "tab",
                                    "title": _spec_text(language, "image_process"),
                                    "sections": [
                                        {
                                            "id": "image_process_actions",
                                            "type": "toolbar",
                                            "title": _spec_text(language, "image_process_actions"),
                                            "controls": [
                                                {"type": "button", "command": "action.run_image_process_current", "label": _spec_text(language, "run_current")},
                                                {"type": "button", "command": "batch.run_image_process", "mode": "batch", "label": _spec_text(language, "run_loaded")},
                                                {"type": "button", "command": "prompt.use_default_image_process", "label": _spec_text(language, "reset_prompt")},
                                            ],
                                        },
                                        {"id": "image_process_prompt", "type": "text_area", "title": _spec_text(language, "image_process_prompt")},
                                    ],
                                },
                            ],
                        },
                        {
                            "id": "text_editor_panel",
                            "type": "panel",
                            "title": _spec_text(language, "text_editor"),
                            "sections": [
                                {
                                    "id": "text_editor_toolbar",
                                    "type": "toolbar",
                                    "title": _spec_text(language, "text_toolbar"),
                                    "controls": [
                                        {"type": "label", "bind": "task.running"},
                                        {"type": "button", "command": "editor.find_replace", "label": _spec_text(language, "find_replace")},
                                        {"type": "button", "command": "editor.undo", "label": _spec_text(language, "undo")},
                                        {"type": "button", "command": "editor.redo", "label": _spec_text(language, "redo")},
                                    ],
                                },
                                {"id": "text_editor", "type": "plain_text", "title": _spec_text(language, "caption_text")},
                                {"id": "progress_bar", "type": "progress", "bind": "task"},
                            ],
                        },
                    ],
                },
            ],
        },
        "state": state,
    }


class RuntimeUiSpecStore:
    """
    In-memory UI spec override store for runtime and agent-driven patching.

    Overrides are ephemeral by design and reset when the process restarts.
    """

    def __init__(self):
        self._override: Dict[str, Any] = {}

    def override_snapshot(self) -> Dict[str, Any]:
        return deepcopy(self._override)

    def merge_override(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(patch, dict):
            raise ValueError("ui spec patch must be a dict")
        self._override = _merge_dicts(self.override_snapshot(), strip_runtime_state(patch))
        return self.override_snapshot()

    def replace_override(self, override: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(override, dict):
            raise ValueError("ui spec override must be a dict")
        self._override = strip_runtime_state(override)
        return self.override_snapshot()

    def reset_override(self) -> Dict[str, Any]:
        self._override = {}
        return self.override_snapshot()

    def apply(self, base_spec: Dict[str, Any]) -> Dict[str, Any]:
        return merge_ui_spec(base_spec, self._override)
