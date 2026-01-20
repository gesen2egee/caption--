# -*- coding: utf-8 -*-
"""
標籤流式顯示組件 (TagFlowWidget) 和標籤按鈕 (TagButton)
"""
import re
from PyQt6.QtWidgets import (
    QPushButton, QSizePolicy, QVBoxLayout, QLabel, 
    QWidget, QScrollArea, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal, Qt

from lib.core.settings import DEFAULT_APP_SETTINGS
from lib.utils.parsing import (
    split_csv_like_text, normalize_for_match, 
    is_basic_character_tag, remove_underline
)


class TagButton(QPushButton):
    toggled_tag = pyqtSignal(str, bool)

    def __init__(self, text, translation=None, parent=None):
        super().__init__(parent)
        self.raw_text = text
        self.translation = translation
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(2)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.label.setStyleSheet("background: none; border: none;")

        self.layout.addWidget(self.label)

        self.update_label()

        self.is_character = False
        self.update_style()
        self.clicked.connect(self.on_click)

    def set_is_character(self, val: bool):
        self.is_character = val
        self.update_style()

    def update_style(self):
        # 嘗試從 Parent 鏈取得主題
        theme = "light"
        p = self.parent()
        while p:
            if hasattr(p, "settings"):
                theme = p.settings.get("ui_theme", "light")
                break
            p = p.parent()
        
        is_dark = (theme == "dark")
        # 紅框僅在未點選時顯示，點選後統一為藍色
        border_color = "red" if self.is_character else ("#555" if is_dark else "#ccc")
        border_width = "2px" if self.is_character else "1px"
        # 點選後一律變藍色
        checked_border = "#007acc" if is_dark else "#0055aa"
        bg_color = "#333" if is_dark else "#f0f0f0"
        checked_bg = "#444" if is_dark else "#d0e8ff"
        text_color = "#d4d4d4" if is_dark else "#333"

        self.setStyleSheet(f"""
            QPushButton {{
                border: {border_width} solid {border_color};
                border-radius: 4px;
                background-color: {bg_color};
                color: {text_color};
            }}
            QPushButton:checked {{
                background-color: {checked_bg};
                border: 2px solid {checked_border};
            }}
            QPushButton:hover {{
                border: {border_width} solid {"#777" if is_dark else "#999"};
            }}
        """)

    def update_label(self):
        # 取得主題以決定顏色
        theme = "light"
        p = self.parent()
        while p:
            if hasattr(p, "settings"):
                theme = p.settings.get("ui_theme", "light")
                break
            p = p.parent()
        text_color = "#d4d4d4" if theme == "dark" else "#000"
        trans_color = "#999" if theme == "dark" else "#666"

        safe_text = str(self.raw_text).replace("<", "&lt;").replace(">", "&gt;")
        content = f"<span style='font-size:13px; font-weight:bold; color:{text_color};'>{safe_text}</span>"

        if self.translation:
            safe_trans = str(self.translation).replace("<", "&lt;").replace(">", "&gt;")
            content += f"<br><span style='color:{trans_color}; font-size:11px;'>{safe_trans}</span>"

        self.label.setText(content)

    def on_click(self):
        self.toggled_tag.emit(self.raw_text, self.isChecked())


class TagFlowWidget(QWidget):
    tag_clicked = pyqtSignal(str, bool)

    def __init__(self, parent=None, use_scroll=True):
        super().__init__(parent)
        self.use_scroll = use_scroll
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.translations_csv = {}
        self.buttons = {}

        if self.use_scroll:
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

            self.container = QWidget()
            self.container_layout = QVBoxLayout(self.container)
            self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.container_layout.setSpacing(0)

            self.scroll.setWidget(self.container)
            self.layout.addWidget(self.scroll)
        else:
            self.scroll = None
            self.container = QWidget()
            self.container_layout = QVBoxLayout(self.container)
            self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.container_layout.setSpacing(0)
            self.layout.addWidget(self.container)

    def set_translations_csv(self, trans):
        self.translations_csv = trans

    def render_tags_flow(self, parsed_items, active_text_content, cfg=None):
        while self.container_layout.count():
            child = self.container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.buttons = {}

        is_long_text_mode = False
        for item in parsed_items:
            if len(item['text']) > 40 or (" " in item['text'] and "." in item['text']):
                is_long_text_mode = True
                break

        MAX_ITEMS_PER_ROW = 1 if is_long_text_mode else 4

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(2)

        current_row_widget = QWidget()
        current_row_layout = QHBoxLayout(current_row_widget)
        current_row_layout.setContentsMargins(0, 0, 0, 0)
        current_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        wrapper_layout.addWidget(current_row_widget)

        item_count_in_row = 0

        for item in parsed_items:
            text = item['text']
            trans = item['trans']
            if not trans:
                # 使用 remove_underline 進行統一匹配
                lookup_key = remove_underline(text)
                trans = self.translations_csv.get(lookup_key)

            btn = TagButton(text, trans)

            if is_long_text_mode:
                btn.setMinimumHeight(65)
            else:
                btn.setFixedHeight(55 if trans else 35)

            btn.toggled_tag.connect(self.handle_tag_toggle)
            self.buttons[text] = btn
            # 特徵標籤標記
            if cfg:
                if is_basic_character_tag(text, cfg):
                    btn.set_is_character(True)
            elif is_basic_character_tag(text, DEFAULT_APP_SETTINGS):
                btn.set_is_character(True)
            
            current_row_layout.addWidget(btn)
            item_count_in_row += 1

            if item_count_in_row >= MAX_ITEMS_PER_ROW:
                current_row_widget = QWidget()
                current_row_layout = QHBoxLayout(current_row_widget)
                current_row_layout.setContentsMargins(0, 0, 0, 0)
                current_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                wrapper_layout.addWidget(current_row_widget)
                item_count_in_row = 0

        self.container_layout.addWidget(wrapper)

        self.sync_state(active_text_content)

    def handle_tag_toggle(self, tag, checked):
        self.tag_clicked.emit(tag, checked)

    def sync_state(self, active_text_content: str):
        # 1. CSV split matching (exact full match of segments)
        current_tokens = split_csv_like_text(active_text_content)
        current_norm = set(normalize_for_match(t) for t in current_tokens)
        
        # 2. Text search (Word boundary regex match)
        # 用於處理 LLM 產生的自然語言句子
        search_text = active_text_content.lower()

        for tag, btn in self.buttons.items():
            btn.blockSignals(True)
            
            # Check 1: CSV match
            is_active = normalize_for_match(tag) in current_norm
            
            # Check 2: Regex match if not found yet
            if not is_active:
                try:
                    # Escape tag for regex, add word boundaries
                    esc = re.escape(tag.lower())
                    # allow matching "tag" inside "tag," or "tag." but not "tagging"
                    if re.search(rf"\b{esc}\b", search_text):
                        is_active = True
                except Exception:
                    pass
            
            btn.setChecked(is_active)
            btn.blockSignals(False)
