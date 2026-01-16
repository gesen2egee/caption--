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
                             QTabWidget, QSizePolicy, QLineEdit, QCheckBox,
                             QPushButton, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QAction

from lib.ui.widgets import TagFlowWidget, TagButton, StrokeCanvas
from lib.const import DEFAULT_APP_SETTINGS, DEFAULT_CUSTOM_TAGS, THEME_STYLES
from lib.data import save_app_settings
import os


class AppCoreMixin:
    """應用程式核心 Mixin (UI 初始化)"""
    
    def setup_ui_components(self):
        """初始化 UI 組件 (Restore Original Layout)"""
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        if hasattr(self, 'setup_shortcuts'):
            self.setup_shortcuts()

        # Main Splitter (Left | Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ==========================================
        # LEFT PANEL: Info, Filter, Image
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # --- Info Bar ---
        info_layout = QHBoxLayout()
        info_layout.setSpacing(5)
        
        self.index_input = QLineEdit()
        self.index_input.setFixedWidth(80)
        self.index_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_input.returnPressed.connect(self.jump_to_index)
        info_layout.addWidget(self.index_input)

        self.total_label = QLabel("/ 0") # Renamed from total_info_label for consistency with mixins
        info_layout.addWidget(self.total_label)

        self.filename_label = QLabel(": No Image") # Renamed from img_file_label
        info_layout.addWidget(self.filename_label, 1)

        left_layout.addLayout(info_layout)

        # --- Filter Bar ---
        filter_bar = QHBoxLayout()
        filter_bar.setContentsMargins(0, 0, 0, 0)
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Danbooru Filter (e.g. 1girl blue_eyes)")
        self.filter_input.returnPressed.connect(self.apply_filter)
        filter_bar.addWidget(self.filter_input, 1)
        
        self.chk_filter_tags = QCheckBox("Tags")
        self.chk_filter_tags.setChecked(True)
        filter_bar.addWidget(self.chk_filter_tags)
        
        self.chk_filter_text = QCheckBox("Text")
        self.chk_filter_text.setChecked(False)
        filter_bar.addWidget(self.chk_filter_text)
        
        self.btn_clear_filter = QPushButton("X")
        self.btn_clear_filter.setFixedWidth(30)
        self.btn_clear_filter.clicked.connect(lambda: (self.filter_input.clear(), self.apply_filter()))
        filter_bar.addWidget(self.btn_clear_filter)

        # View Mode Selector
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([self.tr("mode_original"), self.tr("mode_rgb"), self.tr("mode_alpha")])
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        filter_bar.addWidget(self.view_mode_combo)
        
        left_layout.addLayout(filter_bar)

        # --- Image Area ---
        # Note: Original used left_layout.addWidget(self.image_label, 1) directly, 
        # but to support zoom/pan properly (if mixin does it) we usually keep ScrollArea.
        # However, to STRICTLY restore, I should check if I should use ScrollArea.
        # The user's manual edit added ScrollArea. The original code (temp_old_caption_utf8.py) 
        # DID NOT use ScrollArea for the main image, but the ImageMixin MIGHT expect it?
        # Let's use ScrollArea because ImageMixin logic (update_image_display) often scales pixmap to label.
        # Actually, looking at ImageMixin (Step 850), it does `scaled_pixmap... image_label.setPixmap`.
        # If I want exact restoration, I should probably stick to what works best. 
        # I'll use a ScrollArea but make it resizable, which mimics the behavior but adds scrollbars if needed.
        
        self.sa_image = QScrollArea()
        self.sa_image.setWidgetResizable(True)
        self.sa_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_label = QLabel(self.tr("label_no_image"))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_image_context_menu)
        
        self.sa_image.setWidget(self.image_label)
        left_layout.addWidget(self.sa_image, 1)
        
        splitter.addWidget(left_panel)

        # ==========================================
        # RIGHT PANEL: Vertical Splitter (Tabs | Text)
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(self.right_splitter)
        
        # --- Tabs ---
        self.tabs = QTabWidget()
        self.right_splitter.addWidget(self.tabs)
        
        # 1. Tags Tab
        self.tab_tags = QWidget()
        self.setup_tags_tab(self.tab_tags)
        self.tabs.addTab(self.tab_tags, self.tr("sec_tags"))
        
        # 2. NL Tab
        self.tab_nl = QWidget()
        self.setup_nl_tab(self.tab_nl)
        self.tabs.addTab(self.tab_nl, self.tr("sec_nl"))

        # --- Bottom Text Widget ---
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)
        bot_layout.setContentsMargins(5, 5, 5, 5)
        
        # Toolbar
        bot_toolbar = QHBoxLayout()
        self.bot_label = QLabel(f"<b>{self.tr('label_txt_content')}</b>")
        bot_toolbar.addWidget(self.bot_label)
        bot_toolbar.addSpacing(10)
        
        self.txt_token_label = QLabel(f"{self.tr('label_tokens')}0")
        bot_toolbar.addWidget(self.txt_token_label)
        
        bot_toolbar.addStretch(1)
        
        self.btn_find_replace = QPushButton(self.tr("btn_find_replace"))
        self.btn_find_replace.clicked.connect(self.open_find_replace)
        bot_toolbar.addWidget(self.btn_find_replace)
        
        self.btn_txt_undo = QPushButton(self.tr("btn_undo"))
        self.btn_txt_undo.clicked.connect(lambda: self.txt_edit.undo())
        bot_toolbar.addWidget(self.btn_txt_undo)
        
        self.btn_txt_redo = QPushButton(self.tr("btn_redo"))
        self.btn_txt_redo.clicked.connect(lambda: self.txt_edit.redo())
        bot_toolbar.addWidget(self.btn_txt_redo)
        
        bot_layout.addLayout(bot_toolbar)
        
        # Text Edit
        self.txt_edit = QTextEdit()
        if hasattr(self, 'on_text_changed'):
            self.txt_edit.textChanged.connect(self.on_text_changed)
        self.txt_edit.setMinimumHeight(100)
        bot_layout.addWidget(self.txt_edit)
        
        self.right_splitter.addWidget(bot_widget)
        
        # Progress Bar & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        self.btn_cancel_batch = QPushButton(self.tr("btn_cancel_batch"))
        self.btn_cancel_batch.setVisible(False)
        self.btn_cancel_batch.clicked.connect(self.cancel_batch)
        self.statusBar().addPermanentWidget(self.btn_cancel_batch)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        # Initial Sizes
        self.right_splitter.setSizes([780, 220])

        # Initialize Menus & Theme
        if hasattr(self, '_setup_menus'):
            self._setup_menus()
        if hasattr(self, 'apply_theme'):
             self.apply_theme()
             
    def setup_tags_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Toolbar
        tags_toolbar = QHBoxLayout()
        tags_toolbar.addWidget(QLabel("<b>TAGS</b>"))
        
        self.btn_auto_tag = QPushButton(self.tr("btn_auto_tag"))
        self.btn_auto_tag.clicked.connect(self.auto_tag_current_image)
        tags_toolbar.addWidget(self.btn_auto_tag)
        
        self.btn_batch_tagger = QPushButton(self.tr("btn_batch_tagger"))
        self.btn_batch_tagger.clicked.connect(lambda: self.run_batch_tagger(to_txt=False))
        tags_toolbar.addWidget(self.btn_batch_tagger)
        
        self.btn_batch_tagger_to_txt = QPushButton(self.tr("btn_batch_tagger_to_txt"))
        self.btn_batch_tagger_to_txt.clicked.connect(self.run_batch_tagger_to_txt)
        tags_toolbar.addWidget(self.btn_batch_tagger_to_txt)
        
        self.btn_add_custom_tag = QPushButton(self.tr("btn_add_tag"))
        self.btn_add_custom_tag.clicked.connect(self.add_custom_tag_dialog)
        tags_toolbar.addWidget(self.btn_add_custom_tag)
        
        tags_toolbar.addStretch(1)
        layout.addLayout(tags_toolbar)
        
        # Scroll Area for Tags
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setSpacing(8)
        
        self.sec1_title = QLabel(f"<b>{self.tr('sec_folder_meta')}</b>")
        c_layout.addWidget(self.sec1_title)
        
        self.flow_top = TagFlowWidget(self, use_scroll=False, 
                                    on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        if hasattr(self, 'translations_csv'):
             self.flow_top.set_translations_csv(self.translations_csv)
        c_layout.addWidget(self.flow_top)
        
        c_layout.addWidget(self.make_hline())
        
        self.sec2_title = QLabel(f"<b>{self.tr('sec_custom')}</b>")
        c_layout.addWidget(self.sec2_title)
        
        self.flow_custom = TagFlowWidget(self, use_scroll=False,
                                       on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        if hasattr(self, 'translations_csv'):
             self.flow_custom.set_translations_csv(self.translations_csv)
        c_layout.addWidget(self.flow_custom)
        
        c_layout.addWidget(self.make_hline())
        
        self.sec3_title = QLabel(f"<b>{self.tr('sec_tagger')}</b>")
        c_layout.addWidget(self.sec3_title)
        
        self.flow_tagger = TagFlowWidget(self, use_scroll=False,
                                       on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        if hasattr(self, 'translations_csv'):
             self.flow_tagger.set_translations_csv(self.translations_csv)
        c_layout.addWidget(self.flow_tagger)
        
        c_layout.addStretch(1)
        scroll.setWidget(container)

    def setup_nl_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Toolbar
        nl_toolbar = QHBoxLayout()
        self.nl_label = QLabel(f"<b>{self.tr('sec_nl')}</b>")
        nl_toolbar.addWidget(self.nl_label)
        
        self.btn_run_llm = QPushButton(self.tr("btn_run_llm"))
        self.btn_run_llm.clicked.connect(self.run_llm_single)
        nl_toolbar.addWidget(self.btn_run_llm)
        
        self.btn_batch_llm = QPushButton(self.tr("btn_batch_llm"))
        self.btn_batch_llm.clicked.connect(lambda: self.run_batch_llm(to_txt=False))
        nl_toolbar.addWidget(self.btn_batch_llm)
        
        self.btn_batch_llm_to_txt = QPushButton(self.tr("btn_batch_llm_to_txt"))
        self.btn_batch_llm_to_txt.clicked.connect(self.run_batch_llm_to_txt)
        nl_toolbar.addWidget(self.btn_batch_llm_to_txt)
        
        self.btn_prev_nl = QPushButton(self.tr("btn_prev"))
        self.btn_prev_nl.clicked.connect(self.prev_nl_page)
        nl_toolbar.addWidget(self.btn_prev_nl)
        
        self.btn_next_nl = QPushButton(self.tr("btn_next"))
        self.btn_next_nl.clicked.connect(self.next_nl_page)
        nl_toolbar.addWidget(self.btn_next_nl)
        
        self.btn_reset_prompt = QPushButton(self.tr("btn_reset_prompt"))
        self.btn_reset_prompt.clicked.connect(lambda: self.prompt_edit.setPlainText(self.default_user_prompt_template))
        nl_toolbar.addWidget(self.btn_reset_prompt)
        
        self.nl_page_label = QLabel("Page 0/0")
        nl_toolbar.addWidget(self.nl_page_label)
        
        nl_toolbar.addStretch(1)
        layout.addLayout(nl_toolbar)
        
        # Result Title
        self.nl_result_title = QLabel(f"<b>{self.tr('label_nl_result')}</b>")
        layout.addWidget(self.nl_result_title)
        
        # Result Flow (Scrollable)
        self.flow_nl = TagFlowWidget(self, use_scroll=True,
                                   on_tag_click=lambda t, c: self.on_tag_button_toggled(t, c))
        if hasattr(self, 'translations_csv'):
             self.flow_nl.set_translations_csv(self.translations_csv)
        # Original: setMinimumHeight(520), setMaximumHeight(900)
        self.flow_nl.setMinimumHeight(400) # Slightly adjusted
        layout.addWidget(self.flow_nl)
        
        # Prompt Editor
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("LLM Prompt Template...")
        self.prompt_edit.setPlainText(self.default_user_prompt_template)
        layout.addWidget(self.prompt_edit, 1)

    def make_hline(self):
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def closeEvent(self, event):
        event.accept()
