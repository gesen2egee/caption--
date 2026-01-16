"""
NL (Natural Language) 頁面管理 Mixin

負責處理：
- NL 頁面刷新和渲染
- NL 頁面導航（上一頁/下一頁）
- NL 頁面控制更新
- NL 結果區域高度動態調整

依賴的屬性：
- self.nl_pages: list - NL 頁面列表
- self.nl_page_index: int - 當前頁面索引
- self.nl_latest: str - 最新的 NL 內容
- self.txt_edit - 文本編輯器
- self.flow_nl - NL 標籤流式佈局
- self.nl_page_label - 頁面標籤
- self.btn_prev_nl - 上一頁按鈕
- self.btn_next_nl - 下一頁按鈕
- self.prompt_edit - 提示詞編輯器
- self.settings: dict - 應用程式設定
- self.tr() - 翻譯函數
- self.on_text_changed() - 文本變更處理方法
"""

from lib.utils import smart_parse_tags


class NLMixin:
    """NL 頁面管理 Mixin"""
    
    def refresh_nl_tab(self):
        """
        刷新 NL 標籤頁
        
        根據當前的 nl_latest 內容重新渲染 NL 標籤流式佈局
        """
        active_text = self.txt_edit.toPlainText()
        
        # 清理 LLM 輸出中的提示標記（使用統一的清理函數）
        from lib.utils import clean_llm_output
        cleaned_content = clean_llm_output(self.nl_latest)
        
        self.flow_nl.render_tags_flow(
            smart_parse_tags(cleaned_content),
            active_text,
            self.settings
        )

    def set_current_nl_page(self, idx: int):
        """
        設定當前 NL 頁面
        
        Args:
            idx: 頁面索引（0-based）
        """
        if not self.nl_pages:
            self.nl_page_index = 0
            self.nl_latest = ""
            self.refresh_nl_tab()
            self.update_nl_page_controls()
            return

        idx = max(0, min(int(idx), len(self.nl_pages) - 1))
        self.nl_page_index = idx
        self.nl_latest = self.nl_pages[self.nl_page_index]

        self.refresh_nl_tab()
        self.update_nl_page_controls()
        self.on_text_changed()

    def update_nl_page_controls(self):
        """
        更新 NL 頁面控制元件
        
        更新頁面標籤顯示和上一頁/下一頁按鈕的啟用狀態
        """
        total = len(self.nl_pages)
        if total <= 0:
            if hasattr(self, "nl_page_label"):
                self.nl_page_label.setText(f"{self.tr('label_page')} 0/0")
            if hasattr(self, "btn_prev_nl"):
                self.btn_prev_nl.setEnabled(False)
            if hasattr(self, "btn_next_nl"):
                self.btn_next_nl.setEnabled(False)
        else:
            self.nl_page_index = max(0, min(self.nl_page_index, total - 1))
            if hasattr(self, "nl_page_label"):
                self.nl_page_label.setText(f"Page {self.nl_page_index + 1}/{total}")
            if hasattr(self, "btn_prev_nl"):
                self.btn_prev_nl.setEnabled(self.nl_page_index > 0)
            if hasattr(self, "btn_next_nl"):
                self.btn_next_nl.setEnabled(self.nl_page_index < total - 1)

        self.update_nl_result_height()

    def prev_nl_page(self):
        """切換到上一頁 NL"""
        if self.nl_pages and self.nl_page_index > 0:
            self.set_current_nl_page(self.nl_page_index - 1)

    def next_nl_page(self):
        """切換到下一頁 NL"""
        if self.nl_pages and self.nl_page_index < len(self.nl_pages) - 1:
            self.set_current_nl_page(self.nl_page_index + 1)

    def update_nl_result_height(self):
        """
        動態調整 NL 結果區域高度
        
        根據內容行數自動調整 flow_nl 的最小高度和 prompt_edit 的最大高度：
        - 16+ 行：flow_nl 760px, prompt_edit 220px
        - 10-15 行：flow_nl 660px, prompt_edit 280px
        - <10 行：flow_nl 520px, prompt_edit 無限制
        """
        # make RESULT taller when content is long, and shrink prompt area accordingly
        try:
            lines = [l for l in (self.nl_latest or "").splitlines() if l.strip()]
            n = len(lines)

            if n >= 16:
                self.flow_nl.setMinimumHeight(760)
                self.prompt_edit.setMaximumHeight(220)
            elif n >= 10:
                self.flow_nl.setMinimumHeight(660)
                self.prompt_edit.setMaximumHeight(280)
            else:
                self.flow_nl.setMinimumHeight(520)
                self.prompt_edit.setMaximumHeight(9999)
        except Exception:
            pass
