"""
主題和本地化 Mixin

負責處理：
- 主題應用和切換
- 多語言翻譯
- UI 元素重新翻譯
- 選單設定

依賴的屬性：
- self.settings: dict - 應用程式設定
- self.menuBar() - 選單欄
- self.findChildren() - 查找子元件
- self.btn_* - 各種按鈕元件
- self.tabs - 標籤頁元件
- self.*_label - 各種標籤元件
- self.update_txt_token_count() - 更新 Token 計數方法
- self.update_nl_page_controls() - 更新 NL 頁面控制方法
- self.open_directory(), self.refresh_file_list(), self.open_settings() - 檔案操作方法
- self.unmask_current_image(), self.mask_text_current_image(), self.restore_current_image() - 圖片處理方法
- self.run_batch_* - 批量處理方法
- self.open_stroke_eraser() - 手繪橡皮擦方法
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QAction, QKeySequence
from lib.const import THEME_STYLES
from lib.ui.widgets import TagButton
from lib import localization


class ThemeMixin:
    """主題和本地化 Mixin"""
    
    def tr(self, key: str, **kwargs) -> str:
        """
        翻譯函數
        
        Args:
            key: 翻譯鍵值 (支援點分隔路徑如 "ui.app_title")
            **kwargs: 格式化參數
            
        Returns:
            翻譯後的字串，如果找不到則返回原鍵值
        """
        # 確保本地化模組使用與 settings 相同的語言
        lang = self.settings.get("ui_language", "zh_tw")
        if localization.get_current_language() != lang:
            localization.set_language(lang)
        
        # 支援舊式不帶前綴的 key (向後兼容)
        if "." not in key:
            # 嘗試在各區域查找
            for prefix in ["ui", "dialog", "settings", "status", "error", "context_menu", "tooltip"]:
                result = localization.get_text(f"{prefix}.{key}", **kwargs)
                if result != f"{prefix}.{key}":
                    return result
            return localization.get_text(f"ui.{key}", **kwargs)
        
        return localization.get_text(key, **kwargs)

    def apply_theme(self):
        """
        應用主題樣式
        
        根據設定中的 ui_theme 應用對應的樣式表，
        並強制刷新所有 TagButton 的樣式
        """
        theme = self.settings.get("ui_theme", "light")
        self.setStyleSheet(THEME_STYLES.get(theme, ""))
        # 強制刷新所有 TagButton 的樣式
        for btn in self.findChildren(TagButton):
            btn.update_style()

    def retranslate_ui(self):
        """
        重新翻譯 UI 元素
        
        當語言設定變更時，更新所有 UI 元素的文字
        """
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
        self.btn_reset_prompt.setText(self.tr("btn_reset_prompt"))
        self.btn_txt_undo.setText(self.tr("btn_undo"))
        self.btn_txt_redo.setText(self.tr("btn_redo"))
        
        if hasattr(self, 'nl_label'):
            self.nl_label.setText(f"<b>{self.tr('sec_nl')}</b>")
        if hasattr(self, 'bot_label'):
            self.bot_label.setText(f"<b>{self.tr('label_txt_content')}</b>")
        if hasattr(self, 'nl_result_title'):
             self.nl_result_title.setText(f"<b>{self.tr('label_nl_result')}</b>")
        
        if hasattr(self, 'update_txt_token_count'):
            self.update_txt_token_count()
        if hasattr(self, 'update_nl_page_controls'):
            self.update_nl_page_controls()

        # Update tabs
        if hasattr(self, 'tabs'):
            self.tabs.setTabText(0, self.tr("sec_tags"))
            self.tabs.setTabText(1, self.tr("sec_nl"))
        
        # Labels
        if hasattr(self, 'sec1_title'):
            self.sec1_title.setText(f"<b>{self.tr('sec_folder_meta')}</b>")
        if hasattr(self, 'sec2_title'):
            self.sec2_title.setText(f"<b>{self.tr('sec_custom')}</b>")
        if hasattr(self, 'sec3_title'):
            self.sec3_title.setText(f"<b>{self.tr('sec_tagger')}</b>")
        
        if hasattr(self, 'btn_cancel_batch') and self.btn_cancel_batch:
            self.btn_cancel_batch.setText(self.tr("btn_cancel_batch"))
        
        # Menus
        self.menuBar().clear()
        self._setup_menus()

    def _setup_menus(self):
        """
        設定選單欄
        
        創建檔案選單和工具選單，包含所有主要功能的入口
        """
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
        
        unmask_action = QAction(self.tr("btn_unmask"), self)
        unmask_action.setStatusTip("用 AI 自動去除當前圖片的背景，原圖會備份到 unmask 資料夾")
        unmask_action.triggered.connect(self.unmask_current_image)
        tools_menu.addAction(unmask_action)

        mask_text_action = QAction(self.tr("btn_mask_text"), self)
        mask_text_action.setStatusTip("用 OCR 自動偵測並遮蔽當前圖片中的文字區域")
        mask_text_action.triggered.connect(self.mask_text_current_image)
        tools_menu.addAction(mask_text_action)

        restore_action = QAction(self.tr("btn_restore_original"), self)
        restore_action.setStatusTip("從 unmask 資料夾還原原圖，覆蓋目前的去背版本")
        restore_action.triggered.connect(self.restore_current_image)
        tools_menu.addAction(restore_action)

        tools_menu.addSeparator()
        
        self.action_batch_unmask = QAction(self.tr("btn_batch_unmask"), self)
        self.action_batch_unmask.setStatusTip("對所有圖片執行批量去背，可在設定中調整過濾條件")
        self.action_batch_unmask.triggered.connect(self.run_batch_unmask_background)
        tools_menu.addAction(self.action_batch_unmask)

        self.action_batch_mask_text = QAction(self.tr("btn_batch_mask_text"), self)
        self.action_batch_mask_text.setStatusTip("對所有圖片執行批量 OCR 去文字")
        self.action_batch_mask_text.triggered.connect(self.run_batch_mask_text)
        tools_menu.addAction(self.action_batch_mask_text)

        tools_menu.addSeparator()

        self.action_batch_restore = QAction(self.tr("btn_batch_restore"), self)
        self.action_batch_restore.setStatusTip("批量還原所有圖片的原圖 (從 unmask 資料夾)")
        self.action_batch_restore.triggered.connect(self.run_batch_restore)
        tools_menu.addAction(self.action_batch_restore)

        tools_menu.addSeparator()

        stroke_action = QAction(self.tr("btn_stroke_eraser"), self)
        stroke_action.setStatusTip("手動用滑鼠繪製要擦除的區域，適合精細去除")
        stroke_action.triggered.connect(self.open_stroke_eraser)
        tools_menu.addAction(stroke_action)
