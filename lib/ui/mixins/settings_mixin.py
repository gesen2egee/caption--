from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QDialog, QMessageBox, QApplication
from PyQt6.QtGui import QKeySequence, QAction

from lib.core.settings import (
    save_app_settings, DEFAULT_APP_SETTINGS, DEFAULT_CUSTOM_TAGS, 
    DEFAULT_CUSTOM_PROMPT_TEMPLATE, DEFAULT_USER_PROMPT_TEMPLATE
)
from lib.core.dataclasses import Settings
from lib.ui.dialogs.settings_dialog import SettingsDialog
from lib.ui.themes import THEME_STYLES
from lib.locales import load_locale, tr as _tr
from lib.ui.components.tag_flow import TagButton

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class SettingsMixin:
    """
    Mixin handling Application Settings, Theme, Localization, and Menus.
    """

    def tr(self, key: str) -> str:
        lang = self.settings.get("ui_language", "zh_tw")
        load_locale(lang)
        return _tr(key)

    def apply_theme(self):
        theme = self.settings.get("ui_theme", "light")
        self.setStyleSheet(THEME_STYLES.get(theme, ""))
        # 強制刷新所有 TagButton 的樣式
        for btn in self.findChildren(TagButton):
            btn.update_style()

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_cfg = dlg.get_cfg()
            self.settings = new_cfg
            save_app_settings(new_cfg)

            # 現在設定會在 Task 啟動時動態抓取，不需手動同步到已移除的 pipeline_manager
            
            # Check worker availability again (in case worker settings changed effectively enabling/disabling features)
            self.check_worker_availability()

            # apply immediately
            self.apply_theme()
            self.retranslate_ui()

            # update LLM props
            self.llm_base_url = str(new_cfg.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
            self.api_key = str(new_cfg.get("llm_api_key", ""))
            self.model_name = str(new_cfg.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
            self.llm_system_prompt = str(new_cfg.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
            self.default_user_prompt_template = str(new_cfg.get("llm_user_prompt_template", DEFAULT_APP_SETTINGS["llm_user_prompt_template"]))
            self.custom_prompt_template = str(new_cfg.get("llm_custom_prompt_template", DEFAULT_APP_SETTINGS.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
            self.default_custom_tags_global = list(new_cfg.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
            self.english_force_lowercase = bool(new_cfg.get("english_force_lowercase", True))

            if hasattr(self, "prompt_edit") and self.prompt_edit:
                try:
                    self.prompt_edit.setPlainText(self.default_user_prompt_template)
                except Exception:
                    pass

            self.statusBar().showMessage(self.tr("status_ready"), 4000)

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("app_title"))
        # Update main controls
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_batch_tagger.setText(self.tr("btn_batch_tagger"))
        self.btn_batch_tagger_to_txt.setText(self.tr("btn_batch_tagger_to_txt"))
        self.btn_add_custom_tag.setText(self.tr("btn_add_tag"))
        self.btn_run_llm.setText(self.tr("btn_run_llm"))
        self.btn_batch_llm.setText(self.tr("btn_batch_llm"))
        self.btn_batch_llm_to_txt.setText(self.tr("btn_batch_llm_to_txt"))
        self.btn_prev_nl.setText(self.tr("btn_prev"))
        self.btn_next_nl.setText(self.tr("btn_next"))
        self.btn_find_replace.setText(self.tr("btn_find_replace"))
        self.btn_default_prompt.setText(self.tr("btn_default_prompt"))
        self.btn_custom_prompt.setText(self.tr("btn_custom_prompt"))
        self.btn_txt_undo.setText(self.tr("btn_undo"))
        self.btn_txt_redo.setText(self.tr("btn_redo"))
        
        if hasattr(self, 'btn_prev_img'):
            self.btn_prev_img.setText(self.tr("btn_prev_img"))
            self.btn_prev_img.setToolTip(self.tr("tip_btn_prev_img"))
        if hasattr(self, 'btn_next_img'):
            self.btn_next_img.setText(self.tr("btn_next_img"))
            self.btn_next_img.setToolTip(self.tr("tip_btn_next_img"))
        if hasattr(self, 'btn_del_img'):
            self.btn_del_img.setText(self.tr("btn_delete_img"))
            self.btn_del_img.setToolTip(self.tr("tip_btn_delete_img"))
        
        self.nl_label.setText(f"<b>{self.tr('sec_nl')}</b>")
        self.bot_label.setText(f"<b>{self.tr('label_txt_content')}</b>")
        self.nl_result_title.setText(f"<b>{self.tr('label_nl_result')}</b>")
        self.update_txt_token_count()
        self.update_nl_page_controls()

        # Update tabs
        self.tabs.setTabText(0, self.tr("sec_tags"))
        self.tabs.setTabText(1, self.tr("sec_nl"))
        
        # Labels
        self.sec1_title.setText(f"<b>{self.tr('sec_folder_meta')}</b>")
        if hasattr(self, 'btn_cancel_batch') and self.btn_cancel_batch:
            self.btn_cancel_batch.setText(self.tr("btn_cancel_batch"))
        self.sec2_title.setText(f"<b>{self.tr('sec_custom')}</b>")
        self.sec3_title.setText(f"<b>{self.tr('sec_tagger')}</b>")
        
        # Menus
        self.menuBar().clear()
        self._setup_menus()

    def _setup_menus(self):
        menubar = self.menuBar()
        menubar.clear()
        file_menu = menubar.addMenu(self.tr("menu_file"))
        open_action = QAction(self.tr("menu_open_dir"), self)
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)

        refresh_action = QAction(self.tr("menu_refresh"), self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.triggered.connect(self.refresh_file_list)
        file_menu.addAction(refresh_action)
        settings_action = QAction(self.tr("btn_settings"), self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        tools_menu = menubar.addMenu(self.tr("menu_tools"))
        
        self.action_unmask = QAction(self.tr("btn_unmask"), self)
        self.action_unmask.setStatusTip(self.tr("tip_unmask"))
        self.action_unmask.triggered.connect(self.unmask_current_image)
        tools_menu.addAction(self.action_unmask)

        self.action_mask_text = QAction(self.tr("btn_mask_text"), self)
        self.action_mask_text.setStatusTip(self.tr("tip_mask_text"))
        self.action_mask_text.triggered.connect(self.mask_text_current_image)
        tools_menu.addAction(self.action_mask_text)

        restore_action = QAction(self.tr("btn_restore_original"), self)
        restore_action.setStatusTip(self.tr("tip_restore"))
        restore_action.triggered.connect(self.restore_current_image)
        tools_menu.addAction(restore_action)

        tools_menu.addSeparator()
        
        self.action_batch_unmask = QAction(self.tr("btn_batch_unmask"), self)
        self.action_batch_unmask.setStatusTip(self.tr("tip_batch_unmask"))
        self.action_batch_unmask.triggered.connect(self.run_batch_unmask_background)
        tools_menu.addAction(self.action_batch_unmask)

        self.action_batch_mask_text = QAction(self.tr("btn_batch_mask_text"), self)
        self.action_batch_mask_text.setStatusTip(self.tr("tip_batch_mask_text"))
        self.action_batch_mask_text.triggered.connect(self.run_batch_mask_text)
        tools_menu.addAction(self.action_batch_mask_text)

        tools_menu.addSeparator()

        self.action_batch_restore = QAction(self.tr("btn_batch_restore"), self)
        self.action_batch_restore.setStatusTip(self.tr("tip_batch_restore"))
        self.action_batch_restore.triggered.connect(self.run_batch_restore)
        tools_menu.addAction(self.action_batch_restore)

        tools_menu.addSeparator()

        stroke_action = QAction(self.tr("btn_stroke_eraser"), self)
        stroke_action.setStatusTip(self.tr("tip_stroke_eraser"))
        stroke_action.triggered.connect(self.open_stroke_eraser)
        tools_menu.addAction(stroke_action)
