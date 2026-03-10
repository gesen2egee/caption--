# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from lib.core.settings import (
    DEFAULT_APP_SETTINGS,
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_CUSTOM_TAGS,
    DEFAULT_USER_PROMPT_TEMPLATE,
    load_app_settings,
    save_app_settings,
)
from lib.locales import load_locale, tr as _tr
from lib.ui.mixins.batch_mixin import BatchMixin
from lib.ui.mixins.editor_mixin import EditorMixin
from lib.ui.mixins.navigation_mixin import NavigationMixin
from lib.ui.mixins.pipeline_handler_mixin import PipelineHandlerMixin
from lib.ui.mixins.processing_mixin import ProcessingMixin
from lib.utils.boorutag import load_translations
from lib.utils.memory_utils import unload_all_models
from lib.workers.registry import scan_workers


class _MemoryControl:
    def __init__(self, value: Any = None):
        self._value = value
        self._enabled = True
        self._visible = True
        self._tooltip = ""
        self._style = ""
        self._blocked = False

    def setEnabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def isEnabled(self) -> bool:
        return self._enabled

    def setVisible(self, value: bool) -> None:
        self._visible = bool(value)

    def isVisible(self) -> bool:
        return self._visible

    def setToolTip(self, value: str) -> None:
        self._tooltip = str(value or "")

    def setStatusTip(self, value: str) -> None:
        self._tooltip = str(value or "")

    def setStyleSheet(self, value: str) -> None:
        self._style = str(value or "")

    def blockSignals(self, blocked: bool) -> bool:
        previous = self._blocked
        self._blocked = bool(blocked)
        return previous

    def clear(self) -> None:
        self.setText("")

    def setText(self, value: str) -> None:
        self._value = str(value or "")

    def text(self) -> str:
        return str(self._value or "")

    def setPlainText(self, value: str) -> None:
        self.setText(value)

    def toPlainText(self) -> str:
        return self.text()

    def undo(self) -> None:
        return None

    def redo(self) -> None:
        return None

    def moveCursor(self, *args, **kwargs) -> None:
        return None

    def setFocus(self) -> None:
        return None

    def ensureCursorVisible(self) -> None:
        return None

    def hasFocus(self) -> bool:
        return False

    def setChecked(self, value: bool) -> None:
        self._value = bool(value)

    def isChecked(self) -> bool:
        return bool(self._value)

    def setCurrentIndex(self, value: int) -> None:
        self._value = int(value)

    def currentIndex(self) -> int:
        try:
            return int(self._value)
        except Exception:
            return 0

    def setMaximum(self, value: int) -> None:
        self._maximum = int(value)

    def setValue(self, value: int) -> None:
        self._current_value = int(value)

    def setFormat(self, value: str) -> None:
        self._format = str(value or "")

    def clearSelection(self) -> None:
        return None


class _MemoryTextEdit(_MemoryControl):
    def __init__(self, text: str = ""):
        super().__init__(text)
        self._history: list[str] = []
        self._redo: list[str] = []

    def setPlainText(self, value: str) -> None:
        new_value = str(value or "")
        current = self.text()
        if new_value != current and not self._blocked:
            self._history.append(current)
            self._redo.clear()
        self._value = new_value

    def undo(self) -> None:
        if not self._history:
            return
        current = self.text()
        previous = self._history.pop()
        self._redo.append(current)
        self._value = previous

    def redo(self) -> None:
        if not self._redo:
            return
        current = self.text()
        next_value = self._redo.pop()
        self._history.append(current)
        self._value = next_value


class _MemoryStatusBar:
    def __init__(self):
        self.last_message = ""
        self.last_timeout_ms = 0

    def showMessage(self, message: str, timeout_ms: int = 0) -> None:
        self.last_message = str(message or "")
        self.last_timeout_ms = int(timeout_ms or 0)

    def addPermanentWidget(self, widget: Any) -> None:
        return None


class RuntimeBackendHost(
    BatchMixin,
    NavigationMixin,
    EditorMixin,
    ProcessingMixin,
    PipelineHandlerMixin,
):
    """
    Pure-Python runtime host for web/service modes.

    It keeps the command/event/state surface alive without creating a
    QApplication or MainWindow.
    """

    def __init__(self):
        self._runtime_host_backend = "headless-python"
        self._runtime_task_backend_name = "python-thread"
        self._runtime_task_dispatcher_name = "direct-thread-callback"
        self._qt_residual_components: list[str] = []
        self._shutdown_event = threading.Event()
        self._status_bar = _MemoryStatusBar()
        self.settings = load_app_settings()

        self._init_runtime_attributes()
        self._init_headless_controls()

        self._current_task = None
        self._get_command_registry()
        self._get_runtime_event_bus()
        self._get_task_runner()
        self._maybe_start_runtime_http_bridge()
        self._maybe_start_runtime_service_watch()

        self._sync_runtime_settings_state()
        self._sync_runtime_selection_state()
        self._sync_runtime_ui_state()
        self._sync_runtime_controls_state()
        self._sync_runtime_content_state()
        self._sync_runtime_tags_state()

        if bool(self.settings.get("startup_defer_worker_scan", True)):
            self.start_worker_scan_background()
        else:
            self.scan_workers_blocking()

        last_dir = str(self.settings.get("last_open_dir", "") or "").strip()
        if last_dir and os.path.isdir(last_dir):
            self.root_dir_path = last_dir
            self.refresh_file_list()

    def _init_runtime_attributes(self) -> None:
        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", DEFAULT_APP_SETTINGS["llm_api_key"]))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        self.default_user_prompt_template = str(
            self.settings.get("llm_user_prompt_template", DEFAULT_APP_SETTINGS["llm_user_prompt_template"])
        )
        self.custom_prompt_template = str(
            self.settings.get(
                "llm_custom_prompt_template",
                DEFAULT_APP_SETTINGS.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE),
            )
        )
        self.image_process_prompt_template = str(
            self.settings.get(
                "image_process_prompt_template",
                DEFAULT_APP_SETTINGS.get("image_process_prompt_template", "幫我移除圖中所有的文字、文字氣泡、文字框"),
            )
        )
        self.current_prompt_mode = "default"
        self.default_custom_tags_global = list(self.settings.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
        self.english_force_lowercase = bool(self.settings.get("english_force_lowercase", True))
        self.translations_csv = load_translations()

        self.current_view_mode = 0
        self.temp_view_mode = None
        self.image_files: list[str] = []
        self.current_index = -1
        self.current_image_path = ""
        self.current_folder_path = ""
        self.root_dir_path = ""
        self.filter_active = False
        self.filtered_image_files: list[str] = []
        self.all_image_files: list[str] = []

        self.top_tags: list[str] = []
        self.custom_tags: list[str] = []
        self.tagger_tags: list[str] = []
        self.nl_pages: list[str] = []
        self.nl_page_index = 0
        self.nl_latest = ""

        self._hf_tokenizer = None
        self._clip_tokenizer = None
        self._tokenizer_failed = False
        self._tokenizer_warned = False
        self._app_startup_complete = True
        self._workers_scan_completed = False
        self._workers_scan_thread: Optional[threading.Thread] = None
        self._is_batch_to_txt = False
        self._batch_delete_chars = False

        load_locale(self.settings.get("ui_language", "zh_tw"))

    def _init_headless_controls(self) -> None:
        self.tabs = _MemoryControl(0)
        self.filter_input = _MemoryControl("")
        self.chk_filter_tags = _MemoryControl(True)
        self.chk_filter_text = _MemoryControl(False)
        self.cb_view_mode = _MemoryControl(0)
        self.index_input = _MemoryControl("0")
        self.total_info_label = _MemoryControl(" / 0")
        self.img_file_label = _MemoryControl(self.tr("label_no_image"))
        self.image_label = _MemoryControl("")
        self.progress_bar = _MemoryControl(0)
        self.btn_cancel_batch = _MemoryControl("")

        self.btn_auto_tag = _MemoryControl(self.tr("btn_auto_tag"))
        self.btn_batch_tagger = _MemoryControl(self.tr("btn_batch_tagger"))
        self.chk_tags_save_txt = _MemoryControl(True)
        self.btn_add_custom_tag = _MemoryControl(self.tr("btn_add_tag"))
        self.btn_run_llm = _MemoryControl(self.tr("btn_run_llm"))
        self.btn_batch_llm = _MemoryControl(self.tr("btn_batch_llm"))
        self.chk_llm_save_txt = _MemoryControl(True)
        self.btn_prev_nl = _MemoryControl(self.tr("btn_prev"))
        self.btn_next_nl = _MemoryControl(self.tr("btn_next"))
        self.btn_default_prompt = _MemoryControl(self.tr("btn_default_prompt"))
        self.btn_custom_prompt = _MemoryControl(self.tr("btn_custom_prompt"))
        self.btn_run_imgproc = _MemoryControl(self.tr("btn_run_imgproc"))
        self.btn_batch_imgproc = _MemoryControl(self.tr("btn_batch_imgproc"))
        self.btn_default_img_prompt = _MemoryControl(self.tr("btn_default_img_prompt"))
        self.btn_prev_img = _MemoryControl(self.tr("btn_prev_img"))
        self.btn_next_img = _MemoryControl(self.tr("btn_next_img"))
        self.btn_del_img = _MemoryControl(self.tr("btn_delete_img"))
        self.btn_find_replace = _MemoryControl(self.tr("btn_find_replace"))
        self.btn_txt_undo = _MemoryControl(self.tr("btn_undo"))
        self.btn_txt_redo = _MemoryControl(self.tr("btn_redo"))

        self.action_unmask = _MemoryControl(self.tr("btn_unmask"))
        self.action_mask_text = _MemoryControl(self.tr("btn_mask_text"))
        self.action_batch_unmask = _MemoryControl(self.tr("btn_batch_unmask"))
        self.action_batch_mask_text = _MemoryControl(self.tr("btn_batch_mask_text"))
        self.action_batch_restore = _MemoryControl(self.tr("btn_batch_restore"))

        self.prompt_edit = _MemoryTextEdit(self.default_user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE)
        self.img_prompt_edit = _MemoryTextEdit(self.image_process_prompt_template)
        self.txt_edit = _MemoryTextEdit("")

        self.txt_token_label = _MemoryControl(f"{self.tr('label_tokens')}0")
        self.nl_page_label = _MemoryControl(f"{self.tr('label_page')} 0/0")
        self.nl_label = _MemoryControl("")
        self.bot_label = _MemoryControl("")
        self.nl_result_title = _MemoryControl("")
        self.img_proc_label = _MemoryControl("")
        self.img_prompt_label = _MemoryControl("")
        self.sec1_title = _MemoryControl("")
        self.sec2_title = _MemoryControl("")
        self.sec3_title = _MemoryControl("")

        self.flow_top = None
        self.flow_custom = None
        self.flow_tagger = None
        self.flow_nl = None

    def tr(self, key: str) -> str:
        load_locale(self.settings.get("ui_language", "zh_tw"))
        return _tr(key)

    def statusBar(self) -> _MemoryStatusBar:
        return self._status_bar

    def open_directory(self) -> None:
        raise RuntimeError("selection.open_directory is not supported in headless mode; use selection.set_root_dir")

    def command_open_directory(self) -> dict[str, Any]:
        return {
            "opened": False,
            "reason": "headless_dialog_unsupported",
            "recommended_command": "selection.set_root_dir",
        }

    def add_custom_tag_dialog(self) -> None:
        return None

    def open_find_replace(self) -> None:
        return None

    def open_stroke_eraser(self) -> None:
        return None

    def check_worker_availability(self) -> None:
        return None

    def apply_runtime_settings(self, new_cfg: dict) -> None:
        self._replace_runtime_settings_dict(dict(new_cfg or {}), persist=True)
        load_locale(self.settings.get("ui_language", "zh_tw"))
        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", ""))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        self.default_user_prompt_template = str(
            self.settings.get("llm_user_prompt_template", DEFAULT_APP_SETTINGS["llm_user_prompt_template"])
        )
        self.custom_prompt_template = str(
            self.settings.get(
                "llm_custom_prompt_template",
                DEFAULT_APP_SETTINGS.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE),
            )
        )
        self.default_custom_tags_global = list(self.settings.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
        self.english_force_lowercase = bool(self.settings.get("english_force_lowercase", True))
        self.image_process_prompt_template = str(
            self.settings.get(
                "image_process_prompt_template",
                DEFAULT_APP_SETTINGS.get("image_process_prompt_template", "幫我移除圖中所有的文字、文字氣泡、文字框"),
            )
        )
        if self.current_prompt_mode == "default":
            self.prompt_edit.setPlainText(self.default_user_prompt_template)
        self.img_prompt_edit.setPlainText(self.image_process_prompt_template)
        self._sync_runtime_settings_state()
        self._sync_runtime_content_state()

    def start_worker_scan_background(self) -> None:
        if self._workers_scan_thread is not None and self._workers_scan_thread.is_alive():
            return
        self._get_runtime_event_bus().emit("worker.scan.started", {"background": True})

        def _run() -> None:
            ok = True
            err = ""
            try:
                scan_workers()
            except Exception as exc:
                ok = False
                err = str(exc)
            self.on_worker_scan_done(ok, err)
            self.on_worker_scan_finished()

        self._workers_scan_thread = threading.Thread(target=_run, daemon=True, name="RuntimeWorkerScan")
        self._workers_scan_thread.start()

    def scan_workers_blocking(self) -> None:
        ok = True
        err = ""
        self._get_runtime_event_bus().emit("worker.scan.started", {"background": False})
        try:
            scan_workers()
        except Exception as exc:
            ok = False
            err = str(exc)
        self.on_worker_scan_done(ok, err)
        self.on_worker_scan_finished()

    def on_worker_scan_done(self, ok: bool, err: str) -> None:
        self._workers_scan_completed = True
        payload = {
            "ok": bool(ok),
            "error": str(err or ""),
            "workers": self.get_runtime_workers(),
        }
        event_name = "worker.scan.finished" if ok else "worker.scan.failed"
        self._get_runtime_event_bus().emit(event_name, payload)
        self.statusBar().showMessage(self.tr("status_ready") if ok else f"worker scan failed: {err}", 3000 if ok else 8000)

    def on_worker_scan_finished(self) -> None:
        self._workers_scan_thread = None

    def refresh_tags_tab(self) -> None:
        if hasattr(self, "_sync_runtime_tags_state"):
            self._sync_runtime_tags_state()

    def refresh_nl_tab(self) -> None:
        if hasattr(self, "_sync_runtime_tags_state"):
            self._sync_runtime_tags_state()

    def update_nl_result_height(self) -> None:
        return None

    def update_nl_page_controls(self) -> None:
        total = len(self.nl_pages)
        if total <= 0:
            self.nl_page_label.setText(f"{self.tr('label_page')} 0/0")
            self.btn_prev_nl.setEnabled(False)
            self.btn_next_nl.setEnabled(False)
        else:
            self.nl_page_index = max(0, min(self.nl_page_index, total - 1))
            self.nl_page_label.setText(
                self.tr("label_page_fmt")
                .replace("{current}", str(self.nl_page_index + 1))
                .replace("{total}", str(total))
            )
            self.btn_prev_nl.setEnabled(self.nl_page_index > 0)
            self.btn_next_nl.setEnabled(self.nl_page_index < total - 1)
        self._sync_runtime_ui_state()

    def update_image_display(self) -> None:
        return None

    def load_image(self) -> None:
        if not (0 <= self.current_index < len(self.image_files)):
            self._set_runtime_selection_values(
                current_image_path="",
                current_folder_path="",
                current_index=-1 if not self.image_files else self.current_index,
            )
            self.txt_edit.setPlainText("")
            self.top_tags = []
            self.custom_tags = []
            self.tagger_tags = []
            self.nl_pages = []
            self.nl_page_index = 0
            self.nl_latest = ""
            self._sync_runtime_selection_state()
            self._sync_runtime_content_state()
            self._sync_runtime_tags_state()
            self._sync_runtime_ui_state()
            return

        next_image_path = self.image_files[self.current_index]
        next_folder_path = str(Path(next_image_path).parent)
        self._set_runtime_selection_values(
            current_image_path=next_image_path,
            current_folder_path=next_folder_path,
            current_index=self.current_index,
        )
        total_count = len(self.filtered_image_files) if self.filter_active else len(self.image_files)
        current_num = (
            self.filtered_image_files.index(self.current_image_path) + 1
            if self.filter_active and self.current_image_path in self.filtered_image_files
            else self.current_index + 1
        )
        self.index_input.setText(str(current_num))
        self.total_info_label.setText(f" / {total_count}")
        self.img_file_label.setText(f" : {os.path.basename(self.current_image_path)}")

        txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
        content = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        self.txt_edit.blockSignals(True)
        self.txt_edit.setPlainText(content)
        self.txt_edit.blockSignals(False)

        self.top_tags = self.build_top_tags_for_current_image()
        self.custom_tags = self.load_folder_custom_tags(self.current_folder_path)
        self.tagger_tags = self.load_tagger_tags_for_current_image()
        self.nl_pages = self.load_nl_pages_for_image(self.current_image_path)
        if self.nl_pages:
            self.nl_page_index = len(self.nl_pages) - 1
            self.nl_latest = self.nl_pages[self.nl_page_index]
        else:
            self.nl_page_index = 0
            self.nl_latest = ""

        self.refresh_tags_tab()
        self.refresh_nl_tab()
        self.update_nl_page_controls()
        self.on_text_changed()
        self._sync_runtime_selection_state()
        self._sync_runtime_content_state()
        self._sync_runtime_tags_state()
        self._sync_runtime_controls_state()
        self._sync_runtime_ui_state()

    def command_shutdown_app(self, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            return {"scheduled": False, "reason": "confirmation_required"}
        self._get_runtime_event_bus().emit(
            "app.shutdown_requested",
            {
                "bridge": self.get_runtime_bridge_status(),
                "runtime_host_backend": self._runtime_host_backend,
            },
        )
        threading.Thread(target=self.close, daemon=True, name="RuntimeBackendShutdown").start()
        return {"scheduled": True}

    def close(self) -> None:
        try:
            if self.is_task_running():
                self.stop_current_task()
        except Exception:
            pass
        try:
            self._stop_runtime_http_bridge()
        except Exception:
            pass
        try:
            self._stop_runtime_service_watch()
        except Exception:
            pass
        try:
            self.command_stop_worker_services()
        except Exception:
            pass
        try:
            unload_all_models()
        except Exception:
            pass
        self._shutdown_event.set()

    def wait_forever(self, poll_interval: float = 0.5) -> None:
        try:
            while not self._shutdown_event.wait(timeout=poll_interval):
                time.sleep(0.0)
        except KeyboardInterrupt:
            self.close()
