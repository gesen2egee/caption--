# -*- coding: utf-8 -*-
import os
import sys
import shutil
import json
import re
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QLineEdit, QLabel, QPushButton, QCheckBox, QComboBox, 
    QScrollArea, QTabWidget, QTextEdit, QPlainTextEdit, 
    QProgressBar, QMessageBox, QFrame, QMenu, QFileDialog, 
    QInputDialog, QApplication, QStyle, QLayout
)
from lib.ui.mixins.batch_mixin import BatchMixin
from lib.ui.mixins.navigation_mixin import NavigationMixin
from lib.ui.mixins.editor_mixin import EditorMixin
from lib.ui.mixins.processing_mixin import ProcessingMixin
from lib.ui.mixins.settings_mixin import SettingsMixin
from lib.ui.mixins.pipeline_handler_mixin import PipelineHandlerMixin
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize, QUrl, QBuffer, QIODevice, QByteArray, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QAction, QKeySequence, QShortcut, QDesktopServices, QPalette, QBrush, QIcon


from PIL import Image, ImageChops

from lib.core.settings import (
    load_app_settings, DEFAULT_APP_SETTINGS,
    DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_CUSTOM_TAGS
)
from lib.core.dataclasses import Settings, ImageData

from lib.utils.file_ops import (
    load_image_sidecar, save_image_sidecar
)

from lib.utils.boorutag import load_translations



from lib.ui.components.tag_flow import TagFlowWidget, TagButton
from lib.ui.components.stroke import create_checkerboard_png_bytes



from lib.pipeline.manager import PipelineManager

from lib.utils.batch_writer import write_batch_result


from lib.utils.memory_utils import unload_all_models
from lib.workers.registry import get_registry

try:
    from transparent_background import Remover
except ImportError:
    Remover = None

CLIPTokenizer = None
from lib.workers.registry import scan_workers, get_registry

    window = MainWindow()
class MainWindow(SettingsMixin, QMainWindow, BatchMixin, NavigationMixin, EditorMixin, ProcessingMixin, PipelineHandlerMixin):
    def __init__(self):
        self.settings = load_app_settings()
        super().__init__()
        self.setWindowTitle(self.tr("app_title"))

        # Init PipelineManager
        self.pipeline_manager = PipelineManager(self)
        # Convert dict settings to dataclass safely
        valid_keys = Settings.__annotations__.keys()
        clean_settings = {k: v for k, v in self.settings.items() if k in valid_keys}
        self.pipeline_manager.set_settings(Settings(**clean_settings))
        
        self.pipeline_manager.image_done.connect(self.on_pipeline_image_done)
        self.pipeline_manager.progress.connect(self.on_pipeline_progress)
        self.pipeline_manager.pipeline_done.connect(self.on_pipeline_done)
        self.pipeline_manager.error.connect(self.on_pipeline_error)

        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", DEFAULT_APP_SETTINGS["llm_api_key"]))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        self.default_user_prompt_template = str(self.settings.get("llm_user_prompt_template", DEFAULT_APP_SETTINGS["llm_user_prompt_template"]))
        self.custom_prompt_template = str(self.settings.get("llm_custom_prompt_template", DEFAULT_APP_SETTINGS.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
        self.current_prompt_mode = "default"
        self.default_custom_tags_global = list(self.settings.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
        self.english_force_lowercase = bool(self.settings.get("english_force_lowercase", True))

        self.current_view_mode = 0  # 0=Original, 1=RGB, 2=Alpha
        self.temp_view_mode = None  # For N/M keys override
        
        self.translations_csv = load_translations()

        self.image_files = []
        self.current_index = -1
        self.current_image_path = ""
        self.current_folder_path = ""

        self.top_tags = []
        self.custom_tags = []
        self.tagger_tags = []

        self.root_dir_path = ""
        
        # Filter state
        self.filter_active = False
        self.filtered_image_files = []
        self.all_image_files = []  # Original unfiltered list

        self.nl_pages = []
        self.nl_page_index = 0
        self.nl_latest = ""



        self.init_ui()
        self.apply_theme()
        self.setup_shortcuts()
        self._hf_tokenizer = None

        # Auto-load last directory
        last_dir = self.settings.get("last_open_dir", "")
        if last_dir and os.path.exists(last_dir):
            self.root_dir_path = last_dir
            self.refresh_file_list()

        # Check CUDA availability
        try:
            import torch
            if not torch.cuda.is_available():
                # Use singleShot to show message after UI is fully loaded
                QTimer.singleShot(1000, lambda: QMessageBox.warning(
                    self, 
                    self.tr("msg_cuda_warning_title"), 
                    self.tr("msg_cuda_warning_content")
                ))
        except ImportError:
            pass



    def check_worker_availability(self):
        """檢查 Worker 可用性並更新 UI 狀態"""
        reg = get_registry()
        
        # TAGGER
        has_tagger = reg.has_available_workers("TAGGER")
        self.btn_auto_tag.setEnabled(has_tagger)
        self.btn_batch_tagger.setEnabled(has_tagger)
        self.btn_batch_tagger_to_txt.setEnabled(has_tagger)
        if not has_tagger:
            self.btn_auto_tag.setToolTip(self.tr("tip_no_worker"))
            self.btn_batch_tagger.setToolTip(self.tr("tip_no_worker"))

        # LLM
        has_llm = reg.has_available_workers("LLM")
        self.btn_run_llm.setEnabled(has_llm)
        self.btn_batch_llm.setEnabled(has_llm)
        self.btn_batch_llm_to_txt.setEnabled(has_llm)
        if not has_llm:
            self.btn_run_llm.setToolTip(self.tr("tip_no_worker"))
        
        # UNMASK (Background Removal)
        has_unmask = reg.has_available_workers("UNMASK")
        if hasattr(self, "action_unmask"):
            self.action_unmask.setEnabled(has_unmask)
        if hasattr(self, "action_batch_unmask"):
            self.action_batch_unmask.setEnabled(has_unmask)
            
        # MASK TEXT
        has_mask_text = reg.has_available_workers("MASK_TEXT")
        if hasattr(self, "action_mask_text"):
            self.action_mask_text.setEnabled(has_mask_text)
        if hasattr(self, "action_batch_mask_text"):
            self.action_batch_mask_text.setEnabled(has_mask_text)

    def init_ui(self):
        font = QFont()
        font.setPointSize(12)
        QApplication.instance().setFont(font)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # === Left Side ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # === Info Bar (Interactive Index & Filename) ===
        info_layout = QHBoxLayout()
        info_layout.setSpacing(5)
        
        self.index_input = QLineEdit()
        self.index_input.setFixedWidth(80)
        self.index_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_input.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.index_input.returnPressed.connect(self.jump_to_index)
        info_layout.addWidget(self.index_input)

        self.total_info_label = QLabel("/ 0")
        self.total_info_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        info_layout.addWidget(self.total_info_label)

        self.img_file_label = QLabel(self.tr("label_no_image"))
        self.img_file_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.img_file_label.setWordWrap(False)
        info_layout.addWidget(self.img_file_label, 1)

        left_layout.addLayout(info_layout)

        # === Filter Bar ===
        filter_bar = QHBoxLayout()
        filter_bar.setContentsMargins(0, 0, 0, 0)
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(self.tr("filter_placeholder"))
        self.filter_input.setToolTip(self.tr("tip_filter_input"))
        self.filter_input.returnPressed.connect(self.apply_filter)
        filter_bar.addWidget(self.filter_input, 1)
        
        self.chk_filter_tags = QCheckBox(self.tr("filter_by_tags"))
        self.chk_filter_tags.setChecked(True)
        self.chk_filter_tags.setToolTip(self.tr("tip_filter_tags"))
        filter_bar.addWidget(self.chk_filter_tags)
        
        self.chk_filter_text = QCheckBox(self.tr("filter_by_text"))
        self.chk_filter_text.setChecked(False)
        self.chk_filter_text.setToolTip(self.tr("tip_filter_text"))
        filter_bar.addWidget(self.chk_filter_text)
        
        self.btn_clear_filter = QPushButton("✕")
        self.btn_clear_filter.setFixedWidth(30)
        self.btn_clear_filter.setToolTip(self.tr("tip_clear_filter"))
        self.btn_clear_filter.clicked.connect(self.clear_filter)
        filter_bar.addWidget(self.btn_clear_filter)

        # === View Mode Selector (RGB/Alpha) ===
        self.cb_view_mode = QComboBox()
        self.cb_view_mode.addItems([self.tr("view_mode_original"), self.tr("view_mode_rgb"), self.tr("view_mode_alpha")])
        self.cb_view_mode.setToolTip(self.tr("tip_view_mode"))
        self.cb_view_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus) # 避免搶走焦點影響快速鍵
        self.cb_view_mode.currentIndexChanged.connect(self.on_view_mode_changed)
        filter_bar.addWidget(self.cb_view_mode)
        
        left_layout.addLayout(filter_bar)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        
        # Context Menu
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_image_context_menu)

        # --- 棋盤格背景：用 Palette Brush（避免 data URI pixmap 警告） ---
        png_bytes = create_checkerboard_png_bytes()
        bg = QPixmap()
        bg.loadFromData(png_bytes, "PNG")

        self.image_label.setAutoFillBackground(True)
        pal = self.image_label.palette()
        pal.setBrush(QPalette.ColorRole.Window, QBrush(bg))  # 會自動平鋪
        self.image_label.setPalette(pal)
        self.image_label.setStyleSheet("background-color:#888;")

        left_layout.addWidget(self.image_label, 1)
        splitter.addWidget(left_panel)

        # === Right Side ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(self.right_splitter)

        # Tabs (TAGS / NL)
        self.tabs = QTabWidget()
        self.right_splitter.addWidget(self.tabs)

        # ---- TAGS Tab ----
        tags_tab = QWidget()
        tags_tab_layout = QVBoxLayout(tags_tab)
        tags_tab_layout.setContentsMargins(5, 5, 5, 5)

        tags_toolbar = QHBoxLayout()
        tags_label = QLabel(f"<b>{self.tr('sec_tags_header')}</b>")
        tags_toolbar.addWidget(tags_label)

        self.btn_auto_tag = QPushButton(self.tr("btn_auto_tag"))
        self.btn_auto_tag.setToolTip(self.tr("tip_auto_tag"))
        self.btn_auto_tag.clicked.connect(self.run_tagger)
        tags_toolbar.addWidget(self.btn_auto_tag)

        self.btn_batch_tagger = QPushButton(self.tr("btn_batch_tagger"))
        self.btn_batch_tagger.setToolTip(self.tr("tip_batch_tagger"))
        self.btn_batch_tagger.clicked.connect(self.run_batch_tagger)
        tags_toolbar.addWidget(self.btn_batch_tagger)

        self.btn_batch_tagger_to_txt = QPushButton(self.tr("btn_batch_tagger_to_txt"))
        self.btn_batch_tagger_to_txt.setToolTip(self.tr("tip_batch_tagger_txt"))
        self.btn_batch_tagger_to_txt.clicked.connect(self.run_batch_tagger_to_txt)
        tags_toolbar.addWidget(self.btn_batch_tagger_to_txt)

        self.btn_add_custom_tag = QPushButton(self.tr("btn_add_tag"))
        self.btn_add_custom_tag.setToolTip(self.tr("tip_add_custom_tag"))
        self.btn_add_custom_tag.clicked.connect(self.add_custom_tag_dialog)
        tags_toolbar.addWidget(self.btn_add_custom_tag)

        tags_toolbar.addStretch(1)
        tags_tab_layout.addLayout(tags_toolbar)

        self.tags_scroll = QScrollArea()
        self.tags_scroll.setWidgetResizable(True)
        self.tags_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tags_tab_layout.addWidget(self.tags_scroll)

        tags_scroll_container = QWidget()
        self.tags_scroll_layout = QVBoxLayout(tags_scroll_container)
        self.tags_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.tags_scroll_layout.setSpacing(8)

        self.sec1_title = QLabel(f"<b>{self.tr('sec_folder_meta')}</b>")
        self.tags_scroll_layout.addWidget(self.sec1_title)
        self.flow_top = TagFlowWidget(use_scroll=False)
        self.flow_top.set_translations_csv(self.translations_csv)
        self.flow_top.tag_clicked.connect(self.on_tag_button_toggled)
        self.tags_scroll_layout.addWidget(self.flow_top)

        self.tags_scroll_layout.addWidget(self.make_hline())

        self.sec2_title = QLabel(f"<b>{self.tr('sec_custom')}</b>")
        self.tags_scroll_layout.addWidget(self.sec2_title)
        self.flow_custom = TagFlowWidget(use_scroll=False)
        self.flow_custom.set_translations_csv(self.translations_csv)
        self.flow_custom.tag_clicked.connect(self.on_tag_button_toggled)
        self.tags_scroll_layout.addWidget(self.flow_custom)

        self.tags_scroll_layout.addWidget(self.make_hline())

        self.sec3_title = QLabel(f"<b>{self.tr('sec_tagger')}</b>")
        self.tags_scroll_layout.addWidget(self.sec3_title)
        self.flow_tagger = TagFlowWidget(use_scroll=False)
        self.flow_tagger.set_translations_csv(self.translations_csv)
        self.flow_tagger.tag_clicked.connect(self.on_tag_button_toggled)
        self.tags_scroll_layout.addWidget(self.flow_tagger)

        self.tags_scroll_layout.addStretch(1)
        self.tags_scroll.setWidget(tags_scroll_container)
        
        self.tabs.addTab(tags_tab, self.tr("sec_tags"))

        # ---- NL Tab ----
        nl_tab = QWidget()
        nl_layout = QVBoxLayout(nl_tab)
        nl_layout.setContentsMargins(5, 5, 5, 5)

        nl_toolbar = QHBoxLayout()
        self.nl_label = QLabel(f"<b>{self.tr('sec_nl')}</b>")
        nl_toolbar.addWidget(self.nl_label)

        self.btn_run_llm = QPushButton(self.tr("btn_run_llm"))
        self.btn_run_llm.setToolTip(self.tr("tip_run_llm"))
        self.btn_run_llm.clicked.connect(self.run_llm_generation)
        nl_toolbar.addWidget(self.btn_run_llm)

        # ✅ Batch 按鍵保留在上方
        self.btn_batch_llm = QPushButton(self.tr("btn_batch_llm"))
        self.btn_batch_llm.setToolTip(self.tr("tip_batch_llm"))
        self.btn_batch_llm.clicked.connect(self.run_batch_llm)
        nl_toolbar.addWidget(self.btn_batch_llm)

        self.btn_batch_llm_to_txt = QPushButton(self.tr("btn_batch_llm_to_txt"))
        self.btn_batch_llm_to_txt.setToolTip(self.tr("tip_batch_llm_txt"))
        self.btn_batch_llm_to_txt.clicked.connect(self.run_batch_llm_to_txt)
        nl_toolbar.addWidget(self.btn_batch_llm_to_txt)

        self.btn_prev_nl = QPushButton(self.tr("btn_prev"))
        self.btn_prev_nl.setToolTip(self.tr("tip_prev_nl"))
        self.btn_prev_nl.clicked.connect(self.prev_nl_page)
        nl_toolbar.addWidget(self.btn_prev_nl)

        self.btn_next_nl = QPushButton(self.tr("btn_next"))
        self.btn_next_nl.setToolTip(self.tr("tip_next_nl"))
        self.btn_next_nl.clicked.connect(self.next_nl_page)
        nl_toolbar.addWidget(self.btn_next_nl)

        self.btn_default_prompt = QPushButton(self.tr("btn_default_prompt"))
        self.btn_default_prompt.setToolTip(self.tr("tip_default_prompt"))
        self.btn_default_prompt.clicked.connect(self.use_default_prompt)
        nl_toolbar.addWidget(self.btn_default_prompt)

        self.btn_custom_prompt = QPushButton(self.tr("btn_custom_prompt"))
        self.btn_custom_prompt.setToolTip(self.tr("tip_custom_prompt"))
        self.btn_custom_prompt.clicked.connect(self.use_custom_prompt)
        nl_toolbar.addWidget(self.btn_custom_prompt)
        self.nl_page_label = QLabel(f"{self.tr('label_page')} 0/0")
        nl_toolbar.addWidget(self.nl_page_label)

        nl_toolbar.addStretch(1)
        nl_layout.addLayout(nl_toolbar)

        # ✅ RESULT 最上面
        self.nl_result_title = QLabel(f"<b>{self.tr('label_nl_result')}</b>")
        nl_layout.addWidget(self.nl_result_title)

        self.flow_nl = TagFlowWidget(use_scroll=True)
        self.flow_nl.set_translations_csv(self.translations_csv)
        self.flow_nl.tag_clicked.connect(self.on_tag_button_toggled)
        nl_layout.addWidget(self.flow_nl)

        # ✅ RESULT 大小：大致顯示倒 9 行（可依你字體再微調）
        self.flow_nl.setMinimumHeight(520)
        self.flow_nl.setMaximumHeight(900)

        # ✅ Prompt 放中間（Result 下方）
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFont(QFont("Consolas", 11))
        self.prompt_edit.setPlainText(self.default_user_prompt_template)
        nl_layout.addWidget(self.prompt_edit, 1)

        self.tabs.addTab(nl_tab, self.tr("sec_nl"))

        # ---- Bottom: txt ----
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)
        bot_layout.setContentsMargins(5, 5, 5, 5)

        bot_toolbar = QHBoxLayout()
        self.bot_label = QLabel(f"<b>{self.tr('label_txt_content')}</b>")
        bot_toolbar.addWidget(self.bot_label)
        bot_toolbar.addSpacing(10)
        self.txt_token_label = QLabel(f"{self.tr('label_tokens')}0")
        self.txt_token_label.setToolTip(self.tr("tip_token_count"))
        bot_toolbar.addWidget(self.txt_token_label)
        bot_toolbar.addStretch(1)

        self.btn_find_replace = QPushButton(self.tr("btn_find_replace"))
        self.btn_find_replace.setToolTip(self.tr("tip_find_replace"))
        self.btn_find_replace.clicked.connect(self.open_find_replace)
        bot_toolbar.addWidget(self.btn_find_replace)

        self.btn_txt_undo = QPushButton(self.tr("btn_undo"))
        self.btn_txt_undo.setToolTip(self.tr("tip_undo"))
        
        self.btn_txt_redo = QPushButton(self.tr("btn_redo"))
        self.btn_txt_redo.setToolTip(self.tr("tip_redo"))
        
        bot_toolbar.addWidget(self.btn_txt_undo)
        bot_toolbar.addWidget(self.btn_txt_redo)

        bot_layout.addLayout(bot_toolbar)

        self.txt_edit = QPlainTextEdit()
        self.txt_edit.setFont(QFont("Consolas", 12))
        self.txt_edit.textChanged.connect(self.on_text_changed)

        # ✅ txt 小一點（大約 300 英文字）
        self.txt_edit.setMaximumHeight(260)

        bot_layout.addWidget(self.txt_edit)

        self.btn_txt_undo.clicked.connect(self.txt_edit.undo)
        self.btn_txt_redo.clicked.connect(self.txt_edit.redo)
        self.txt_edit.undoAvailable.connect(self.btn_txt_undo.setEnabled)
        self.txt_edit.redoAvailable.connect(self.btn_txt_redo.setEnabled)

        self.right_splitter.addWidget(bot_widget)

        splitter.addWidget(right_panel)
        splitter.setSizes([700, 900])

        # ✅ 調整三個欄的大小（上:tabs 下:txt）
        self.right_splitter.setSizes([780, 220])

        # status bar progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        self.btn_cancel_batch = QPushButton(self.tr("btn_cancel_batch"))
        self.btn_cancel_batch.setVisible(False)
        self.btn_cancel_batch.clicked.connect(self.cancel_batch)
        self.statusBar().addPermanentWidget(self.btn_cancel_batch)

        self._setup_menus()
        
        # Check workers availability
        self.check_worker_availability()

    def make_hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def setup_shortcuts(self):
        # ✅ 圖片左右/翻頁鍵：用 ApplicationShortcut，焦點在 txt 也能翻
        for key, fn in [
            (Qt.Key.Key_Left, self.prev_image),
            (Qt.Key.Key_Right, self.next_image),
            (Qt.Key.Key_PageUp, self.prev_image),
            (Qt.Key.Key_PageDown, self.next_image),
            (Qt.Key.Key_Home, self.first_image),
            (Qt.Key.Key_End, self.last_image),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)

        # Delete 保持原本（避免在 txt 刪字誤觸搬圖）
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.delete_current_image)

    def wheelEvent(self, event):
        pos = event.position().toPoint()
        widget = self.childAt(pos)
        if widget is self.image_label or (widget and self.image_label.isAncestorOf(widget)):
            dy = event.angleDelta().y()
            if dy > 0:
                self.prev_image()
            elif dy < 0:
                self.next_image()
            event.accept()
            return
        super().wheelEvent(event)

    # ==========================


    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key.Key_N:
            self.temp_view_mode = 1 # RGB
            self.update_image_display()
        elif key == Qt.Key.Key_M:
            self.temp_view_mode = 2 # Alpha
            self.update_image_display()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
            
        key = event.key()
        if key == Qt.Key.Key_N or key == Qt.Key.Key_M:
            # 放開時檢查是否還有其他鍵按著 (簡單起見，直接重置)
            # 如果使用者同時按住 N 和 M，放開一個時會回到 View Mode
            self.temp_view_mode = None
            self.update_image_display()
        else:
            super().keyReleaseEvent(event)



    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 使用 QTimer 避免縮放時過於頻繁的重繪造成卡頓
        QTimer.singleShot(10, self.update_image_display)





    # ==========================
    # Logic: Tagger & LLM
    # ==========================














    # ==========================
    # Tools: Unmask (Remove BG) / Stroke Eraser
    # ==========================




    # ==========================
    # Batch: Tagger / LLM
    # ==========================




    



    # ==========================
    # Tools: Batch Mask Text (OCR)
    # ==========================




    # ==========================
    # Settings
    # ==========================



