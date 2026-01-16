"""
App Core Mixin

負責處理：
- UI 初始化 (init_ui)
- 資源管理
- 關閉事件
- 系統設置初始化

依賴的屬性：
- self.settings: dict
- self.app_settings: AppSettings
- self.shortcuts: dict
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                             QScrollArea, QTextEdit, QLabel, QProgressBar, 
                             QTabWidget, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QAction

from lib.ui.widgets import TagFlowWidget, TagButton, StrokeCanvas
from lib.const import DEFAULT_APP_SETTINGS, DEFAULT_CUSTOM_TAGS, THEME_STYLES
from lib.data import save_app_settings
import os


class AppCoreMixin:
    """應用程式核心 Mixin (UI 初始化)"""
    
    def setup_ui_components(self):
        """初始化 UI 組件"""
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        if hasattr(self, 'setup_shortcuts'):
            self.setup_shortcuts()

        # Splitter (Image | Controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # === LEFT: Image Area ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_label = QLabel(self.tr("label_no_image"))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        scroll_area.setWidget(self.image_label)
        left_layout.addWidget(scroll_area)
        
        splitter.addWidget(left_widget)

        # === RIGHT: Controls Area ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        
        # 1. Tags Tab
        self.tab_tags = QWidget()
        self.setup_tags_tab(self.tab_tags)
        self.tabs.addTab(self.tab_tags, self.tr("sec_tags"))
        
        # 2. NL Tab
        self.tab_nl = QWidget()
        self.setup_nl_tab(self.tab_nl)
        self.tabs.addTab(self.tab_nl, self.tr("sec_nl"))

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Status Bar
        self.statusBar().showMessage(self.tr("status_ready"))

        # Add to splitter
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # Initialize Menus
        if hasattr(self, '_setup_menus'):
            self._setup_menus()
            
        # Apply Theme
        if hasattr(self, 'apply_theme'):
             self.apply_theme()
             
    def closeEvent(self, event):
        """應用程式關閉事件"""
        # Save window state if needed
        event.accept()

    # 以下是 setup_tags_tab 和 setup_nl_tab 的具體實現
    # 由於這些代碼之前混雜在 init_ui 中，這裡需要將它們組織起來
    # 為了保持 Mixin 簡潔，我們假設這兩個方法會在 MainWindow 中實現或在此處實現
    # 這裡我們將其實現，因為它們屬於 UI 構建

    def setup_tags_tab(self, parent):
        layout = QVBoxLayout(parent)
        
        # Folder & Meta
        self.sec1_title = QLabel(f"<b>{self.tr('sec_folder_meta')}</b>")
        layout.addWidget(self.sec1_title)
        
        self.flow_top = TagFlowWidget(self.settings, 
                                    on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        layout.addWidget(self.flow_top)
        
        # Custom Tags
        self.sec2_title = QLabel(f"<b>{self.tr('sec_custom')}</b>")
        layout.addWidget(self.sec2_title)
        
        # Add Tag Button
        self.btn_add_custom_tag = TagButton("Add Tag", is_active=False)
        self.btn_add_custom_tag.clicked.connect(self.add_custom_tag_dialog)
        layout.addWidget(self.btn_add_custom_tag)
        
        self.flow_custom = TagFlowWidget(self.settings,
                                       on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        layout.addWidget(self.flow_custom)
        
        # Tagger Tags
        self.sec3_title = QLabel(f"<b>{self.tr('sec_tagger')}</b>")
        layout.addWidget(self.sec3_title)
        
        # Tagger Controls
        from PyQt6.QtWidgets import QPushButton
        h_tagger = QHBoxLayout()
        self.btn_auto_tag = QPushButton(self.tr("btn_auto_tag"))
        self.btn_auto_tag.clicked.connect(self.auto_tag_current_image)
        h_tagger.addWidget(self.btn_auto_tag)
        
        self.btn_batch_tagger = QPushButton(self.tr("btn_batch_tagger"))
        self.btn_batch_tagger.clicked.connect(lambda: self.run_batch_tagger(to_txt=False))
        h_tagger.addWidget(self.btn_batch_tagger)
        
        self.btn_batch_tagger_to_txt = QPushButton(self.tr("btn_batch_tagger_to_txt"))
        self.btn_batch_tagger_to_txt.clicked.connect(lambda: self.run_batch_tagger(to_txt=True))
        h_tagger.addWidget(self.btn_batch_tagger_to_txt)
        
        layout.addLayout(h_tagger)
        
        self.flow_tagger = TagFlowWidget(self.settings,
                                       on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        layout.addWidget(self.flow_tagger)
        
        layout.addStretch()

    def setup_nl_tab(self, parent):
        layout = QVBoxLayout(parent)
        
        # Prompt Editor
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("User Prompt for LLM...")
        self.prompt_edit.setMaximumHeight(100)
        self.prompt_edit.setPlainText(self.default_user_prompt_template)
        layout.addWidget(self.prompt_edit)
        
        # Reset Prompt Button
        from PyQt6.QtWidgets import QPushButton
        self.btn_reset_prompt = QPushButton(self.tr("btn_reset_prompt"))
        self.btn_reset_prompt.clicked.connect(lambda: self.prompt_edit.setPlainText(self.default_user_prompt_template))
        layout.addWidget(self.btn_reset_prompt)

        # LLM Controls
        h_llm = QHBoxLayout()
        self.btn_run_llm = QPushButton(self.tr("btn_run_llm"))
        self.btn_run_llm.clicked.connect(self.run_llm_single)
        h_llm.addWidget(self.btn_run_llm)
        
        self.btn_batch_llm = QPushButton(self.tr("btn_batch_llm"))
        self.btn_batch_llm.clicked.connect(lambda: self.run_batch_llm(to_txt=False))
        h_llm.addWidget(self.btn_batch_llm)
        
        self.btn_batch_llm_to_txt = QPushButton(self.tr("btn_batch_llm_to_txt"))
        self.btn_batch_llm_to_txt.clicked.connect(lambda: self.run_batch_llm(to_txt=True))
        h_llm.addWidget(self.btn_batch_llm_to_txt)
        
        self.btn_cancel_batch = QPushButton(self.tr("btn_cancel_batch"))
        self.btn_cancel_batch.clicked.connect(self.cancel_batch)
        self.btn_cancel_batch.setVisible(False)
        self.btn_cancel_batch.setStyleSheet("background-color: #ffcccc;")
        h_llm.addWidget(self.btn_cancel_batch)
        
        layout.addLayout(h_llm)
        
        # Result Navigation
        h_nav = QHBoxLayout()
        self.nl_result_title = QLabel(f"<b>{self.tr('label_nl_result')}</b>")
        h_nav.addWidget(self.nl_result_title)
        
        self.btn_prev_nl = QPushButton(self.tr("btn_prev"))
        self.btn_prev_nl.clicked.connect(self.prev_nl_page)
        self.btn_prev_nl.setEnabled(False)
        h_nav.addWidget(self.btn_prev_nl)
        
        self.nl_page_label = QLabel("Page 0/0")
        h_nav.addWidget(self.nl_page_label)
        
        self.btn_next_nl = QPushButton(self.tr("btn_next"))
        self.btn_next_nl.clicked.connect(self.next_nl_page)
        self.btn_next_nl.setEnabled(False)
        h_nav.addWidget(self.btn_next_nl)
        
        layout.addLayout(h_nav)
        
        # Result Flow
        self.flow_nl = TagFlowWidget(self.settings,
                                   on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        # Use ScrollArea for result
        sa_nl = QScrollArea()
        sa_nl.setWidgetResizable(True)
        sa_nl.setWidget(self.flow_nl)
        layout.addWidget(sa_nl)
        
        # Text Editor (Bottom)
        self.bot_label = QLabel(f"<b>{self.tr('label_txt_content')}</b>")
        layout.addWidget(self.bot_label)
        
        self.txt_edit = QTextEdit()
        self.txt_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.txt_edit)
        
        # Text Controls
        h_txt_ctrl = QHBoxLayout()
        self.txt_token_label = QLabel(f"{self.tr('label_tokens')}0")
        h_txt_ctrl.addWidget(self.txt_token_label)
        
        self.btn_txt_undo = QPushButton(self.tr("btn_undo"))
        self.btn_txt_undo.clicked.connect(self.txt_edit.undo)
        h_txt_ctrl.addWidget(self.btn_txt_undo)
        
        self.btn_txt_redo = QPushButton(self.tr("btn_redo"))
        self.btn_txt_redo.clicked.connect(self.txt_edit.redo)
        h_txt_ctrl.addWidget(self.btn_txt_redo)
        
        self.btn_find_replace = QPushButton(self.tr("btn_find_replace"))
        self.btn_find_replace.clicked.connect(self.open_find_replace)
        h_txt_ctrl.addWidget(self.btn_find_replace)
        
        layout.addLayout(h_txt_ctrl)
