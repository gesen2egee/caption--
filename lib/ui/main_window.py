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
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize, QUrl, QBuffer, QIODevice, QByteArray, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QAction, QKeySequence, QShortcut, QDesktopServices, QPalette, QBrush, QIcon, QTextCursor

from natsort import natsorted
from PIL import Image, ImageChops

from lib.core.settings import (
    load_app_settings, save_app_settings, DEFAULT_APP_SETTINGS, 
    DEFAULT_CUSTOM_TAGS, DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CUSTOM_PROMPT_TEMPLATE
)
from lib.core.dataclasses import Settings, ImageData
from lib.locales import load_locale, locale_tr
from lib.utils.file_ops import (
    load_image_sidecar, save_image_sidecar, 
    backup_raw_image, restore_raw_image, has_raw_backup,
    delete_matching_npz, get_raw_image_dir
)
from lib.utils.parsing import (
    smart_parse_tags, extract_bracket_content, is_basic_character_tag, 
    try_tags_to_text_list, cleanup_csv_like_text, extract_llm_content_and_postprocess
)
from lib.utils.boorutag import parse_boorutag_meta, load_translations
from lib.utils.query_filter import DanbooruQueryFilter
from lib.workers.compat import BatchMaskTextWorker

from lib.ui.themes import THEME_STYLES
from lib.ui.components.tag_flow import TagFlowWidget, TagButton
from lib.ui.components.stroke import StrokeEraseDialog, create_checkerboard_png_bytes
from lib.ui.dialogs.find_replace import AdvancedFindReplaceDialog
from lib.ui.dialogs.settings_dialog import SettingsDialog

from lib.pipeline.manager import PipelineManager, create_image_data_from_path, create_image_data_list


def unload_all_models():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

try:
    from transparent_background import Remover
except ImportError:
    Remover = None

try:
    from transformers import AutoTokenizer, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPTokenizer = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Captioning Assistant")
        self._clip_tokenizer = None
        self.resize(1600, 1000)

        self.settings = load_app_settings()

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
                    "CUDA Warning", 
                    "偵測不到 NVIDIA GPU (CUDA)。\n\n這可能是因為 venv 中的 PyTorch 版本錯誤。\n請執行根目錄下的 'fix_torch_gpu.bat' 來修復。\n\n目前將使用 CPU 執行，速度會非常慢。"
                ))
        except ImportError:
            pass

    def tr(self, key: str) -> str:
        lang = self.settings.get("ui_language", "zh_tw")
        load_locale(lang)
        return locale_tr(key)

    def apply_theme(self):
        theme = self.settings.get("ui_theme", "light")
        self.setStyleSheet(THEME_STYLES.get(theme, ""))
        # 強制刷新所有 TagButton 的樣式
        for btn in self.findChildren(TagButton):
            btn.update_style()

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

        self.img_file_label = QLabel(": No Image")
        self.img_file_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.img_file_label.setWordWrap(False)
        info_layout.addWidget(self.img_file_label, 1)

        left_layout.addLayout(info_layout)

        # === Filter Bar ===
        filter_bar = QHBoxLayout()
        filter_bar.setContentsMargins(0, 0, 0, 0)
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(self.tr("filter_placeholder"))
        self.filter_input.setToolTip("用 Danbooru 語法篩選圖片，輸入後按 Enter\n例如：blonde_hair blue_eyes (同時含)\n-rating:explicit (排除 NSFW)\norder:landscape (橫圖優先)")
        self.filter_input.returnPressed.connect(self.apply_filter)
        filter_bar.addWidget(self.filter_input, 1)
        
        self.chk_filter_tags = QCheckBox(self.tr("filter_by_tags"))
        self.chk_filter_tags.setChecked(True)
        self.chk_filter_tags.setToolTip("勾選後，會搜尋圖片的標籤 (Sidecar JSON)")
        filter_bar.addWidget(self.chk_filter_tags)
        
        self.chk_filter_text = QCheckBox(self.tr("filter_by_text"))
        self.chk_filter_text.setChecked(False)
        self.chk_filter_text.setToolTip("勾選後，會搜尋圖片的 .txt 檔案內容")
        filter_bar.addWidget(self.chk_filter_text)
        
        self.btn_clear_filter = QPushButton("✕")
        self.btn_clear_filter.setFixedWidth(30)
        self.btn_clear_filter.setToolTip("清除篩選條件，顯示所有圖片")
        self.btn_clear_filter.clicked.connect(self.clear_filter)
        filter_bar.addWidget(self.btn_clear_filter)

        # === View Mode Selector (RGB/Alpha) ===
        self.cb_view_mode = QComboBox()
        self.cb_view_mode.addItems(["預覽: 原圖", "預覽: RGB 色版 (N)", "預覽: Alpha 色版 (M)"])
        self.cb_view_mode.setToolTip("切換圖片預覽模式\n- 原圖: 顯示原始圖片\n- RGB: 強制不透明顯示顏色\n- Alpha: 顯示透明度遮罩 (黑透白不透)\n\n快速鍵: 按住 N (RGB) / 按住 M (Alpha)")
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
        tags_label = QLabel("<b>TAGS</b>")
        tags_toolbar.addWidget(tags_label)

        self.btn_auto_tag = QPushButton(self.tr("btn_auto_tag"))
        self.btn_auto_tag.setToolTip("用 WD14 AI 模型自動識別當前圖片的標籤\n點選標籤可加入下方的文字框")
        self.btn_auto_tag.clicked.connect(self.run_tagger)
        tags_toolbar.addWidget(self.btn_auto_tag)

        self.btn_batch_tagger = QPushButton(self.tr("btn_batch_tagger"))
        self.btn_batch_tagger.setToolTip("對資料夾內所有圖片執行自動標籤\n結果儲存在 JSON 中，不會寫入 txt")
        self.btn_batch_tagger.clicked.connect(self.run_batch_tagger)
        tags_toolbar.addWidget(self.btn_batch_tagger)

        self.btn_batch_tagger_to_txt = QPushButton(self.tr("btn_batch_tagger_to_txt"))
        self.btn_batch_tagger_to_txt.setToolTip("對所有圖片執行標籤並寫入 .txt 檔案\n已有標籤記錄的圖片會直接使用快取")
        self.btn_batch_tagger_to_txt.clicked.connect(self.run_batch_tagger_to_txt)
        tags_toolbar.addWidget(self.btn_batch_tagger_to_txt)

        self.btn_add_custom_tag = QPushButton(self.tr("btn_add_tag"))
        self.btn_add_custom_tag.setToolTip("新增自定義標籤到當前資料夾\n這些標籤會儲存在 .custom_tags.json 中")
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
        self.btn_run_llm.setToolTip("用 AI 大型語言模型生成自然語言描述\n結果顯示在上方的 LLM 結果區")
        self.btn_run_llm.clicked.connect(self.run_llm_generation)
        nl_toolbar.addWidget(self.btn_run_llm)

        # ✅ Batch 按鍵保留在上方
        self.btn_batch_llm = QPushButton(self.tr("btn_batch_llm"))
        self.btn_batch_llm.setToolTip("對所有圖片執行 LLM 自然語言生成\n結果儲存在 JSON 中，不會寫入 txt")
        self.btn_batch_llm.clicked.connect(self.run_batch_llm)
        nl_toolbar.addWidget(self.btn_batch_llm)

        self.btn_batch_llm_to_txt = QPushButton(self.tr("btn_batch_llm_to_txt"))
        self.btn_batch_llm_to_txt.setToolTip("對所有圖片執行 LLM 並寫入 .txt 檔案\n已有 LLM 結果的圖片會直接使用快取")
        self.btn_batch_llm_to_txt.clicked.connect(self.run_batch_llm_to_txt)
        nl_toolbar.addWidget(self.btn_batch_llm_to_txt)

        self.btn_prev_nl = QPushButton(self.tr("btn_prev"))
        self.btn_prev_nl.setToolTip("查看上一次的 LLM 生成結果\n每張圖片的所有 LLM 歷史都會保留")
        self.btn_prev_nl.clicked.connect(self.prev_nl_page)
        nl_toolbar.addWidget(self.btn_prev_nl)

        self.btn_next_nl = QPushButton(self.tr("btn_next"))
        self.btn_next_nl.setToolTip("查看下一次的 LLM 生成結果")
        self.btn_next_nl.clicked.connect(self.next_nl_page)
        nl_toolbar.addWidget(self.btn_next_nl)

        self.btn_default_prompt = QPushButton(self.tr("btn_default_prompt"))
        self.btn_default_prompt.setToolTip("切換到預設的 Prompt 模板\n適合生成完整的多句式描述")
        self.btn_default_prompt.clicked.connect(self.use_default_prompt)
        nl_toolbar.addWidget(self.btn_default_prompt)

        self.btn_custom_prompt = QPushButton(self.tr("btn_custom_prompt"))
        self.btn_custom_prompt.setToolTip("切換到自訂的 Prompt 模板\n可在設定中修改自訂模板的內容")
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
        self.txt_token_label.setToolTip("CLIP Token 計數，SD 建議不超過 225\n超過後文字會變紅色警告")
        bot_toolbar.addWidget(self.txt_token_label)
        bot_toolbar.addStretch(1)

        self.btn_find_replace = QPushButton(self.tr("btn_find_replace"))
        self.btn_find_replace.setToolTip("在當前圖片或所有圖片的 txt 中\n尋找並取代文字 (支援正則表達式)")
        self.btn_find_replace.clicked.connect(self.open_find_replace)
        bot_toolbar.addWidget(self.btn_find_replace)

        self.btn_txt_undo = QPushButton(self.tr("btn_undo"))
        self.btn_txt_undo.setToolTip("復原上一步的文字編輯 (Ctrl+Z)")
        self.btn_txt_redo = QPushButton(self.tr("btn_redo"))
        self.btn_txt_redo.setToolTip("重做下一步的文字編輯 (Ctrl+Y)")
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
    # Logic: Storage & Init
    # ==========================
    def refresh_file_list(self, current_path=None): # 改回參數名以支援舊有呼叫
        if not self.root_dir_path or not os.path.exists(self.root_dir_path):
            return
        
        dir_path = self.root_dir_path
        # 若無外部傳入路徑，則嘗試保留目前選取的路徑
        if not current_path:
            current_path = self.current_image_path
        
        self.image_files = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        ignore_dirs = {"no_used", "unmask"}

        # Copied logic from open_directory scan
        try:
            for entry in os.scandir(dir_path):
                if entry.is_file() and entry.name.lower().endswith(valid_exts):
                    if any(part.lower() in ignore_dirs for part in Path(entry.path).parts):
                        continue
                    self.image_files.append(entry.path)
        except Exception:
            pass

        try:
            for entry in os.scandir(dir_path):
                if entry.is_dir():
                    if entry.name.lower() in ignore_dirs:
                        continue
                    try:
                        for sub in os.scandir(entry.path):
                            if sub.is_file() and sub.name.lower().endswith(valid_exts):
                                if any(part.lower() in ignore_dirs for part in Path(sub.path).parts):
                                    continue
                                self.image_files.append(sub.path)
                    except Exception:
                        pass
        except Exception:
            pass

        self.image_files = natsorted(self.image_files)

        if not self.image_files:
            self.image_label.clear()
            self.txt_edit.clear()
            self.img_info_label.setText("No Images Found")
            self.current_index = -1
            self.current_image_path = None
            return

        # Restore index
        if current_path and current_path in self.image_files:
            self.current_index = self.image_files.index(current_path)
        else:
            # If current file gone, try to stay at same index or 0
            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1
            if self.current_index < 0:
                self.current_index = 0
        
        self.load_image()
        self.statusBar().showMessage(f"已重新整理列表: 共 {len(self.image_files)} 張圖片", 3000)

    def open_directory(self):
        default_dir = self.settings.get("last_open_dir", "")
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("msg_select_dir"), default_dir)
        if dir_path:
            self.root_dir_path = dir_path
            self.settings["last_open_dir"] = dir_path
            save_app_settings(self.settings)

            # Reset filter state
            self.filter_active = False
            self.filter_input.clear()
            self.all_image_files = []
            self.filtered_image_files = []

            self.refresh_file_list()

    def load_image(self):
        if 0 <= self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
            self.current_folder_path = str(Path(self.current_image_path).parent)

            # Update info bar
            total_count = len(self.filtered_image_files) if self.filter_active else len(self.image_files)
            current_num = self.filtered_image_files.index(self.current_image_path) + 1 if self.filter_active and self.current_image_path in self.filtered_image_files else self.current_index + 1
            
            self.index_input.blockSignals(True)
            self.index_input.setText(str(current_num))
            self.index_input.blockSignals(False)
            
            if self.filter_active:
                self.total_info_label.setText(f"<span style='color:red;'> / {total_count}</span>")
            else:
                self.total_info_label.setText(f" / {total_count}")
            
            self.img_file_label.setText(f" : {os.path.basename(self.current_image_path)}")

            self.current_pixmap = QPixmap(self.current_image_path)
            if not self.current_pixmap.isNull():
                self.update_image_display()
            else:
                self.image_label.clear()

            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            content = ""
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            self.txt_edit.blockSignals(True)
            self.txt_edit.setPlainText(content)
            self.txt_edit.blockSignals(False)

            self.update_txt_token_count()

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

    # ==========================
    # Filter Logic
    # ==========================
    def _get_image_content_for_filter(self, image_path: str) -> str:
        """Get combined content (tags + text) for filtering."""
        content_parts = []
        
        if self.chk_filter_tags.isChecked():
            # Get tags from sidecar
            sidecar = load_image_sidecar(image_path)
            tags = sidecar.get("tagger_tags", "")
            content_parts.append(tags)
        
        if self.chk_filter_text.isChecked():
            # Get text from .txt file
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content_parts.append(f.read())
                except Exception:
                    pass
        
        return " ".join(content_parts)

    def apply_filter(self):
        """Apply Danbooru-style filter to image list."""
        query = self.filter_input.text().strip()
        
        if not query:
            self.clear_filter()
            return
        
        if not self.image_files and not self.all_image_files:
            return
        
        # Store original list if not already stored
        if not self.all_image_files:
            self.all_image_files = list(self.image_files)
        
        # Create filter
        qf = DanbooruQueryFilter(query)
        
        # Filter images
        matched = []
        for img_path in self.all_image_files:
            content = self._get_image_content_for_filter(img_path)
            if qf.matches(content):
                matched.append(img_path)
        
        # Apply ordering
        matched = qf.sort_images(matched)
        
        if not matched:
            self.statusBar().showMessage("篩選結果為空", 3000)
            return
        
        self.filtered_image_files = matched
        self.image_files = matched
        self.filter_active = True
        self.current_index = 0
        self.load_image()
        self.statusBar().showMessage(f"篩選結果: {len(matched)} 張圖片", 3000)

    def clear_filter(self):
        """Clear filter and restore original image list."""
        self.filter_input.clear()
        
        if self.all_image_files:
            current_path = self.current_image_path
            self.image_files = list(self.all_image_files)
            self.all_image_files = []
            self.filtered_image_files = []
            self.filter_active = False
            
            # Try to keep current image selected
            if current_path and current_path in self.image_files:
                self.current_index = self.image_files.index(current_path)
            else:
                self.current_index = 0
            
            if self.image_files:
                self.load_image()
            self.statusBar().showMessage("已清除篩選", 2000)

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def first_image(self):
        if self.image_files:
            self.current_index = 0
            self.load_image()

    def last_image(self):
        if self.image_files:
            self.current_index = len(self.image_files) - 1
            self.load_image()

    def jump_to_index(self):
        try:
            val = int(self.index_input.text())
            target_idx = val - 1
            
            if self.filter_active:
                if 0 <= target_idx < len(self.filtered_image_files):
                    target_path = self.filtered_image_files[target_idx]
                    self.current_index = self.image_files.index(target_path)
                    self.load_image()
                else:
                    self.load_image() # Reset to current
            else:
                if 0 <= target_idx < len(self.image_files):
                    self.current_index = target_idx
                    self.load_image()
                else:
                    self.load_image() # Reset to current
        except Exception:
            self.load_image()

    def update_image_display(self):
        """用於 Resize 或初次加載時更新圖片顯示"""
        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return
        scaled = self._get_processed_pixmap().scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def _get_processed_pixmap(self) -> QPixmap:
        """
        根據目前的檢視模式 (Original/RGB/Alpha) 回傳對應的 QPixmap。
        優先順序: 
        1. 暫時按鍵 (temp_view_mode: N/M)
        2. 下拉選單 (current_view_mode)
        """
        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return QPixmap()

        # 決定要用的模式 (0=Orig, 1=RGB, 2=Alpha)
        mode = self.current_view_mode
        if self.temp_view_mode is not None:
            mode = self.temp_view_mode

        # 如果是「原圖模式」，檢查是否有外部遮罩 (mask/*.png)
        if mode == 0:
            sidecar = load_image_sidecar(self.current_image_path)
            rel_mask = sidecar.get("mask_map_rel_path", "")
            if rel_mask:
                mask_abs = os.path.normpath(os.path.join(os.path.dirname(self.current_image_path), rel_mask))
                if os.path.exists(mask_abs):
                    # 動態合成
                    img_q = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                    mask_q = QImage(mask_abs).convertToFormat(QImage.Format.Format_Alpha8)
                    if img_q.size() == mask_q.size():
                        img_q.setAlphaChannel(mask_q)
                        return QPixmap.fromImage(img_q)

            return self.current_pixmap
        
        # 轉換處理
        img = self.current_pixmap.toImage()
        
        if mode == 1: # RGB Only (Force Opaque)
            # 轉換為 RGB888 (丟棄 Alpha)
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            return QPixmap.fromImage(img)
            
        elif mode == 2: # Alpha Only
            if img.hasAlphaChannel():
                # Convert to Alpha8 (data is 8-bit alpha)
                alpha_img = img.convertToFormat(QImage.Format.Format_Alpha8)
                # Interpret the data as Grayscale8
                # Note: We must ensure the data persists or is copied.
                # Constructing QImage from inputs shares memory? 
                # Safe way: Convert Alpha8 to RGBA, then fill RGB with Alpha? No.
                
                # Using PIL is robust and easy if we can convert.
                ptr = alpha_img.constBits()
                ptr.setsize(alpha_img.sizeInBytes())
                # Create Grayscale QImage from the bytes
                gray_img = QImage(ptr, alpha_img.width(), alpha_img.height(), alpha_img.bytesPerLine(), QImage.Format.Format_Grayscale8)
                return QPixmap.fromImage(gray_img.copy()) # copy to detach
            else:
                # No Alpha -> White
                white = QPixmap(img.size())
                white.fill(Qt.GlobalColor.white)
                return white
        
        return self.current_pixmap

    def on_view_mode_changed(self, index):
        self.current_view_mode = index
        self.update_image_display()

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

    def show_image_context_menu(self, pos: QPoint):
        if not self.current_image_path:
            return

        menu = QMenu(self)
        
        # 1. 複製圖片
        action_copy_img = QAction("複製圖片 (Copy Image)", self)
        action_copy_img.triggered.connect(self._ctx_copy_image)
        menu.addAction(action_copy_img)
        
        # 2. 複製路徑
        action_copy_path = QAction("複製路徑 (Copy Path)", self)
        action_copy_path.triggered.connect(self._ctx_copy_path)
        menu.addAction(action_copy_path)
        
        menu.addSeparator()
        
        # 3. 開啟檔案位置
        action_open_dir = QAction("打開檔案所在目錄 (Open Folder)", self)
        action_open_dir.triggered.connect(self._ctx_open_folder)
        menu.addAction(action_open_dir)
        
        menu.exec(self.image_label.mapToGlobal(pos))

    def _ctx_copy_image(self):
        if hasattr(self, 'current_pixmap') and not self.current_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.current_pixmap)
            self.statusBar().showMessage("圖片已複製到剪貼簿", 2000)

    def _ctx_copy_path(self):
        if self.current_image_path:
            QApplication.clipboard().setText(os.path.abspath(self.current_image_path))
            self.statusBar().showMessage("路徑已複製到剪貼簿", 2000)
    
    def _ctx_open_folder(self):
        if self.current_image_path:
            folder = os.path.dirname(self.current_image_path)
            # 使用 QDesktopServices 開啟目錄
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 使用 QTimer 避免縮放時過於頻繁的重繪造成卡頓
        QTimer.singleShot(10, self.update_image_display)

    def delete_current_image(self):
        if not self.current_image_path:
            return
        reply = QMessageBox.question(
            self, "Confirm", self.tr("msg_delete_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            src_dir = os.path.dirname(self.current_image_path)
            no_used_dir = os.path.join(src_dir, "no_used")
            if not os.path.exists(no_used_dir):
                os.makedirs(no_used_dir)

            files_to_move = [self.current_image_path]
            for ext in [".txt", ".npz", ".boorutag", ".pool.json", ".json"]:
                p = os.path.splitext(self.current_image_path)[0] + ext
                if os.path.exists(p):
                    files_to_move.append(p)

            for f_path in files_to_move:
                try:
                    shutil.move(f_path, os.path.join(no_used_dir, os.path.basename(f_path)))
                except Exception:
                    pass

            self.image_files.pop(self.current_index)
            if self.current_index >= len(self.image_files):
                self.current_index -= 1
            if self.image_files:
                self.load_image()
            else:
                self.image_label.clear()
                self.txt_edit.clear()

    def on_text_changed(self):
        if not self.current_image_path:
            return
        
        content = self.txt_edit.toPlainText()
        original_content = content
        
        # 自動移除空行
        if self.settings.get("text_auto_remove_empty_lines", True):
            lines = content.split("\n")
            lines = [line for line in lines if line.strip()]
            content = "\n".join(lines)
        
        # 自動格式化 (用 , 分割，去除空白，用 ', ' 重組)
        if self.settings.get("text_auto_format", True):
            # 如果內容看起來是 CSV 格式
            if "," in content and "\n" not in content.strip():
                parts = [p.strip() for p in content.split(",") if p.strip()]
                content = ", ".join(parts)
        
        # 如果內容有變動，更新編輯框
        if content != original_content:
            cursor_pos = self.txt_edit.textCursor().position()
            self.txt_edit.blockSignals(True)
            self.txt_edit.setPlainText(content)
            self.txt_edit.blockSignals(False)
            # 嘗試恢復游標位置
            cursor = self.txt_edit.textCursor()
            cursor.setPosition(min(cursor_pos, len(content)))
            self.txt_edit.setTextCursor(cursor)
        
        # 自動儲存 txt
        if self.settings.get("text_auto_save", True):
            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception:
                pass

        self.flow_top.sync_state(content)
        self.flow_custom.sync_state(content)
        self.flow_tagger.sync_state(content)
        self.flow_nl.sync_state(content)

        self.update_txt_token_count()

    
    def _get_clip_tokenizer(self):
        if CLIPTokenizer is None:
            return None
        if self._clip_tokenizer is None:
            try:
                self._clip_tokenizer = CLIPTokenizer("openai/clip-vit-large-patch14")
            except Exception:
                self._clip_tokenizer = None
        return self._clip_tokenizer

    def _get_tokenizer(self):
        """
        Lazy load tokenizer to avoid startup lag.
        Uses the standard SD 1.5 CLIP model (openai/clip-vit-large-patch14).
        """
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        if self._hf_tokenizer is None:
            try:
                # 這裡會下載約 1MB 的 tokenizer 設定檔 (只會下載一次)
                # 這是 Stable Diffusion 1.x / 2.x 最常用的 Text Encoder
                self._hf_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            except Exception as e:
                print(f"Failed to load CLIP tokenizer: {e}")
                self._hf_tokenizer = None
                
        return self._hf_tokenizer

    def update_txt_token_count(self):
            content = self.txt_edit.toPlainText()
            tokenizer = self._get_tokenizer()

            count = 0

            try:
                if tokenizer:
                    # 使用 CLIP Tokenizer 精確計算
                    tokens = tokenizer.encode(content, add_special_tokens=False)
                    count = len(tokens)
                else:
                    # 降級使用 Regex 估算
                    if content.strip():
                        tokens = re.findall(r'\w+|[^\w\s]', content)
                        count = len(tokens)
                
                # 設定顏色：超過 225 才變紅，否則全黑
                text_color = "red" if count > 225 else "black"
                self.txt_token_label.setStyleSheet(f"color: {text_color}")
                
                # 設定文字：只顯示 "Tokens: 數字"
                self.txt_token_label.setText(f"{self.tr('label_tokens')}{count}")
                
            except Exception as e:
                print(f"Token count error: {e}")
                self.txt_token_label.setText(self.tr("label_tokens_err"))

    def retranslate_ui(self):
        # ... other retranslate calls ...
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_reset_prompt.setText(self.tr("btn_reset_prompt"))
        # Update token label text if it's currently showing "Tokens: Err"
        if self.txt_token_label.text() == "Tokens: Err":
            self.txt_token_label.setText(self.tr("label_tokens_err"))
        # Update NL page label
        self.update_nl_page_controls()


# ==========================
    # TAGS sources
    # ==========================
    def build_top_tags_for_current_image(self):
        hints = []
        tags_from_meta = []

        meta_path = str(self.current_image_path) + ".boorutag"
        if os.path.isfile(meta_path):
            tags_meta, hint_info = parse_boorutag_meta(meta_path)
            tags_from_meta.extend(tags_meta)
            hints.extend(hint_info)

        parent = Path(self.current_image_path).parent.name
        if "_" in parent:
            folder_hint = parent.split("_", 1)[1]
            if "{" not in folder_hint:
                folder_hint = f"{{{folder_hint}}}"
            hints.append(folder_hint)

        initial_keywords = []
        for h in hints:
            initial_keywords.extend(extract_bracket_content(h))

        combined = initial_keywords + tags_from_meta
        seen = set()
        final_list = [x for x in combined if not (x in seen or seen.add(x))]
        final_list = [str(t).replace("_", " ").strip() for t in final_list if str(t).strip()]
        if self.english_force_lowercase:
            final_list = [t.lower() for t in final_list]
        return final_list

    def folder_custom_tags_path(self, folder_path):
        return os.path.join(folder_path, ".custom_tags.json")

    def load_folder_custom_tags(self, folder_path):
        p = self.folder_custom_tags_path(folder_path)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tags = data.get("custom_tags", [])
                tags = [str(t).strip() for t in tags if str(t).strip()]
                if not tags:
                    tags = list(self.default_custom_tags_global)
                return tags
            except Exception:
                return list(self.default_custom_tags_global)
        else:
            tags = list(self.default_custom_tags_global)
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"custom_tags": tags}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return tags

    def save_folder_custom_tags(self, folder_path, tags):
        p = self.folder_custom_tags_path(folder_path)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"custom_tags": tags}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_custom_tag_dialog(self):
        if not self.current_folder_path:
            return
        tag, ok = QInputDialog.getText(self, self.tr("dialog_add_tag_title"), self.tr("dialog_add_tag_label"))
        if not ok:
            return
        tag = str(tag).strip()
        if not tag:
            return
        tag = tag.replace("_", " ").strip()
        if self.english_force_lowercase:
            tag = tag.lower()

        tags = list(self.custom_tags)
        if tag not in tags:
            tags.append(tag)
            self.custom_tags = tags
            self.save_folder_custom_tags(self.current_folder_path, tags)
            self.refresh_tags_tab()
            self.on_text_changed()

    def load_tagger_tags_for_current_image(self):
        """從 JSON sidecar 載入 tagger_tags"""
        sidecar = load_image_sidecar(self.current_image_path)
        raw = sidecar.get("tagger_tags", "")
        if not raw:
            return []
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        parts = [t.replace("_", " ").strip() for t in parts]
        if self.english_force_lowercase:
            parts = [t.lower() for t in parts]
        return parts

    def save_tagger_tags_for_image(self, image_path, raw_tags_str):
        """儲存 tagger_tags 到 JSON sidecar"""
        sidecar = load_image_sidecar(image_path)
        sidecar["tagger_tags"] = raw_tags_str
        save_image_sidecar(image_path, sidecar)

    def load_nl_pages_for_image(self, image_path):
        """從 JSON sidecar 載入 nl_pages"""
        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if isinstance(pages, list):
            return [p for p in pages if p and str(p).strip()]
        return []

    def load_nl_for_current_image(self):
        pages = self.load_nl_pages_for_image(self.current_image_path)
        return pages[-1] if pages else ""

    def save_nl_for_image(self, image_path, content):
        """Append nl content 到 JSON sidecar"""
        if not content:
            return
        content = str(content).strip()
        if not content:
            return

        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if not isinstance(pages, list):
            pages = []
        pages.append(content)
        sidecar["nl_pages"] = pages
        save_image_sidecar(image_path, sidecar)

    def refresh_tags_tab(self):
        active_text = self.txt_edit.toPlainText()

        self.flow_top.render_tags_flow(
            smart_parse_tags(", ".join(self.top_tags)),
            active_text,
            self.settings
        )
        self.flow_custom.render_tags_flow(
            smart_parse_tags(", ".join(self.custom_tags)),
            active_text,
            self.settings
        )
        self.flow_tagger.render_tags_flow(
            smart_parse_tags(", ".join(self.tagger_tags)),
            active_text,
            self.settings
        )

    def refresh_nl_tab(self):
        active_text = self.txt_edit.toPlainText()
        self.flow_nl.render_tags_flow(
            smart_parse_tags(self.nl_latest),
            active_text,
            self.settings
        )


    # ==========================
    # NL paging / dynamic sizing
    # ==========================
    def set_current_nl_page(self, idx: int):
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
        if self.nl_pages and self.nl_page_index > 0:
            self.set_current_nl_page(self.nl_page_index - 1)

    def next_nl_page(self):
        if self.nl_pages and self.nl_page_index < len(self.nl_pages) - 1:
            self.set_current_nl_page(self.nl_page_index + 1)

    def update_nl_result_height(self):
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

    # ==========================
    # Logic: Insert / Remove to txt at cursor
    # ==========================
    def on_tag_button_toggled(self, tag, checked):
        if not self.current_image_path:
            return

        tag = str(tag).strip()
        if not tag:
            return

        if checked:
            self.insert_token_at_cursor(tag)
        else:
            self.remove_token_everywhere(tag)

        self.on_text_changed()

    def insert_token_at_cursor(self, token: str):
        token = token.strip()
        if not token:
            return

        edit = self.txt_edit
        text = edit.toPlainText()
        cursor = edit.textCursor()
        
        # (2) 如果沒有游標 (游標在開頭且沒焦點) 則附加在 text 尾
        # 在 PyQt 中，hasFocus() 可以在點擊按鈕前判斷是否有交互
        if cursor.position() == 0 and len(text) > 0 and not edit.hasFocus():
            cursor.movePosition(QTextCursor.MoveOperation.End)
            edit.setTextCursor(cursor)

        # (1) 優先插入在游標位置
        pos = cursor.position()
        before = text[:pos]
        after = text[pos:]

        # 前後加 ", " 然後格式化
        new_text = before + ", " + token + ", " + after
        final = cleanup_csv_like_text(new_text, self.english_force_lowercase)

        edit.blockSignals(True)
        edit.setPlainText(final)
        edit.blockSignals(False)
        
        # 格式化後嘗試把游標移到插入的 token 之後
        new_cursor = edit.textCursor()
        # 簡單搜尋 token 出現的位置 (從之前位置附近開始找)
        search_start = max(0, pos - 5)
        new_pos = final.find(token, search_start)
        if new_pos != -1:
            new_cursor.setPosition(new_pos + len(token))
        else:
            new_cursor.movePosition(QTextCursor.MoveOperation.End)
        
        edit.setTextCursor(new_cursor)
        edit.ensureCursorVisible()
        # 不需要強行 setFocus，保留按鈕焦點可能更方便連續按

    def remove_token_everywhere(self, token: str):
        token = token.strip()
        if not token:
            return
        text = self.txt_edit.toPlainText()

        new_text = text.replace(token, "")
        new_text = cleanup_csv_like_text(new_text)

        self.txt_edit.blockSignals(True)
        self.txt_edit.setPlainText(new_text)
        self.txt_edit.blockSignals(False)

        self.update_txt_token_count()

    # ==========================
    # Logic: Tagger & LLM
    # ==========================
    def run_tagger(self):
        if not self.current_image_path:
            return
        
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
            
        self.btn_auto_tag.setEnabled(False)
        self.btn_auto_tag.setText("Tagging...")
        self.statusBar().showMessage(f"正在分析標籤: {os.path.basename(self.current_image_path)}...")
        
        try:
            image_data = create_image_data_from_path(self.current_image_path)
            self.pipeline_manager.run_tagger(image_data)
        except Exception as e:
            self.on_pipeline_error(str(e))


    def reset_prompt(self):
        self.prompt_edit.setPlainText(DEFAULT_USER_PROMPT_TEMPLATE)

    def build_llm_tags_context_for_image(self, image_path: str) -> str:
        top_tags = []
        try:
            hints = []
            tags_from_meta = []

            meta_path = str(image_path) + ".boorutag"
            if os.path.isfile(meta_path):
                tags_meta, hint_info = parse_boorutag_meta(meta_path)
                tags_from_meta.extend(tags_meta)
                hints.extend(hint_info)

            parent = Path(image_path).parent.name
            if "_" in parent:
                folder_hint = parent.split("_", 1)[1]
                if "{" not in folder_hint:
                    folder_hint = f"{{{folder_hint}}}"
                hints.append(folder_hint)

            initial_keywords = []
            for h in hints:
                initial_keywords.extend(extract_bracket_content(h))

            combined = initial_keywords + tags_from_meta
            seen = set()
            top_tags = [x for x in combined if not (x in seen or seen.add(x))]
            top_tags = [str(t).replace("_", " ").strip() for t in top_tags if str(t).strip()]
        except Exception:
            top_tags = []

        tagger_parts = []
        sidecar = load_image_sidecar(image_path)
        raw = sidecar.get("tagger_tags", "")
        if raw:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
            parts = try_tags_to_text_list(parts)
            tagger_parts = [t.replace("_", " ").strip() for t in parts if t.strip()]

        all_tags = []
        seen2 = set()
        for t in (top_tags + tagger_parts):
            if t and t not in seen2:
                seen2.add(t)
                all_tags.append(t)

        return "\n".join(all_tags)

    def run_llm_generation(self):
        if not self.current_image_path:
            return
            
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        tags_text = self.build_llm_tags_context_for_image(self.current_image_path)
        user_prompt = self.prompt_edit.toPlainText()

        if "{tags}" in user_prompt and not tags_text.strip():
            reply = QMessageBox.question(
                self, "Warning", 
                "Prompt 包含 {tags} 但目前沒有標籤資料。\n確定要繼續嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.btn_run_llm.setEnabled(False)
        self.btn_run_llm.setText("Running LLM...")
        
        try:
            image_data = create_image_data_from_path(self.current_image_path)
            self.pipeline_manager.run_llm(
                image_data, 
                prompt_mode=self.current_prompt_mode, 
                user_prompt=user_prompt
            )
        except Exception as e:
            self.on_pipeline_error(str(e))


    # ==========================
    # Pipeline Handlers
    # ==========================
    def on_pipeline_progress(self, current, total, filename):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Processing... {current}/{total} : {filename}")
        self.btn_cancel_batch.setVisible(True)

    def on_pipeline_error(self, err_msg):
        self.statusBar().showMessage(f"Error: {err_msg}", 8000)
        self.progress_bar.setVisible(False)
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText(self.tr("btn_run_llm"))
        self.set_batch_ui_enabled(True) 

    def on_pipeline_done(self, name, results):
        self.statusBar().showMessage(f"Task '{name}' completed.", 5000)
        self.progress_bar.setVisible(False)
        self.btn_cancel_batch.setVisible(False)
        self.set_batch_ui_enabled(True)
        
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText(self.tr("btn_run_llm"))

    def cancel_batch(self):
        self.pipeline_manager.stop()

    def set_batch_ui_enabled(self, enabled):
        self.btn_batch_tagger.setEnabled(enabled)
        self.btn_batch_tagger_to_txt.setEnabled(enabled)
        self.btn_batch_llm.setEnabled(enabled)
        self.btn_batch_llm_to_txt.setEnabled(enabled)

    def on_pipeline_image_done(self, image_path: str, output: WorkerOutput):
        if not output.success:
            print(f"Error for {image_path}: {output.error}")
            return
            
        task_name = self.pipeline_manager._current_pipeline.name if self.pipeline_manager._current_pipeline else ""
        
        # Tagger
        if "tagger" in task_name:
             if output.result_text:
                 self.save_tagger_tags_for_image(image_path, output.result_text)
                 
             if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                 self.tagger_tags = self.load_tagger_tags_for_current_image()
                 self.refresh_tags_tab()
             
             if getattr(self, "_is_batch_to_txt", False) and output.result_text:
                  self.write_batch_result_to_txt(image_path, output.result_text, is_tagger=True)

        # LLM
        elif "llm" in task_name:
             content = output.result_text or ""
             final_content = extract_llm_content_and_postprocess(content, self.english_force_lowercase)
             
             if final_content:
                 self.save_nl_for_image(image_path, final_content)
                 if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                     if final_content not in self.nl_pages:
                        self.nl_pages.append(final_content)
                     self.nl_page_index = len(self.nl_pages) - 1
                     self.nl_latest = final_content
                     self.refresh_nl_tab()
                     self.update_nl_page_controls()
                     self.on_text_changed()
                     
                 if getattr(self, "_is_batch_to_txt", False):
                      self.write_batch_result_to_txt(image_path, final_content, is_tagger=False)

        # Unmask or Mask Text
        elif "unmask" in task_name or "mask_text" in task_name:
             if output.result_data and "original_path" in output.result_data:
                  old_path = output.result_data.get("original_path")
                  new_path = output.result_data.get("result_path") or image_path
                  self._replace_image_path_in_list(old_path, new_path)
             
             if self.current_image_path:
                 self.load_image()

                 
        # Restore
        elif "restore" in task_name:
             if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                 self.load_image()




    # ==========================
    # Tools: Unmask (Remove BG) / Stroke Eraser
    # ==========================
    def _unique_path(self, path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        for i in range(1, 9999):
            p2 = f"{base}_{i}{ext}"
            if not os.path.exists(p2):
                return p2
        return path

    def _replace_image_path_in_list(self, old_path: str, new_path: str):
        if not old_path or not new_path or os.path.abspath(old_path) == os.path.abspath(new_path):
            return
        
        # Update current path first if match
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(old_path):
            self.current_image_path = new_path

        found = False
        abs_old = os.path.abspath(old_path)
        for i, p in enumerate(self.image_files):
            if os.path.abspath(p) == abs_old:
                self.image_files[i] = new_path
                found = True
                break
        
        # If not found (rare), append? No, just ignore.

    def _tagger_has_background(self, image_path: str) -> bool:
        """檢查 tagger_tags 是否含有 background"""
        sidecar = load_image_sidecar(image_path)
        raw = sidecar.get("tagger_tags", "")
        if not raw:
            return False
        return re.search(r"background", raw, re.IGNORECASE) is not None

    def unmask_current_image(self):
        if not self.current_image_path:
            return
            
        if self.pipeline_manager.is_running():
             QMessageBox.warning(self, "Warning", "已有任務正在執行中")
             return

        try:
            self.pipeline_manager.run_unmask(create_image_data_from_path(self.current_image_path))
            self.statusBar().showMessage("開始去背...", 2000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Unmask 失敗: {e}")


    def on_unmask_single_done(self):
        self.hide_progress()
        self.load_image()
        # Restore logic might have changed file existence, so refresh list?
        self.refresh_file_list(current_path=self.current_image_path)
        self.statusBar().showMessage("Unmask 完成", 5000)

    def mask_text_current_image(self):
        if not self.current_image_path:
            return
        if not self.settings.get("mask_batch_detect_text_enabled", True):
             QMessageBox.information(self, "Info", "設定中已停用 OCR 偵測。")
             return
             
        if self.pipeline_manager.is_running():
             QMessageBox.warning(self, "Warning", "已有任務正在執行中")
             return

        try:
             self.pipeline_manager.run_mask_text(create_image_data_from_path(self.current_image_path))
             self.statusBar().showMessage("正在去字...", 2000)
        except Exception as e:
             QMessageBox.warning(self, "Mask Text Error", f"失敗: {e}")

    def restore_current_image(self):
        """還原當前圖片為原始備份 (從 raw_image 資料夾)"""
        if not self.current_image_path:
            return
        
        if not has_raw_backup(self.current_image_path):
            QMessageBox.information(self, "Restore", "找不到原圖備份紀錄\n(可能尚未進行任何去背/去文字處理)")
            return
            
        if self.pipeline_manager.is_running():
             QMessageBox.warning(self, "Warning", "已有任務正在執行中")
             return

        try:
            self.pipeline_manager.run_restore(create_image_data_from_path(self.current_image_path))
            self.statusBar().showMessage("正在還原...", 2000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"還原失敗: {e}")


    def run_batch_unmask_background(self):
        if not self.image_files:
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        # ✅ 修正：根據設定決定是否過濾
        only_bg = bool(self.settings.get("mask_batch_only_if_has_background_tag", False))
        targets = []
        if only_bg:
            targets = [p for p in self.image_files if self._image_has_background_tag(p)]
            if not targets:
                QMessageBox.information(self, "Batch Unmask", "找不到含有 'background' 標籤的圖片")
                return
        else:
            targets = self.image_files

        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(False)
            
        try:
             images = create_image_data_list(targets)
             self.pipeline_manager.run_batch_unmask(images)
        except Exception as e:
             self.on_pipeline_error(str(e))




    @staticmethod
    def _qimage_to_pil_l(qimg: QImage) -> Image.Image:
        # 轉換格式
        q = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
        
        # 使用 QBuffer 將 QImage 存為 PNG 格式的 Bytes
        # 這樣可以由 Qt 自動處理所有的 Stride/Padding/Alignment 問題
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        q.save(buf, "PNG")
        
        # 使用 PIL 從記憶體中讀取
        return Image.open(BytesIO(ba.data()))

    def stroke_erase_to_webp(self, image_path: str, mask_qimg: QImage) -> str:
        if not image_path:
            return ""

        src_dir = os.path.dirname(image_path)
        unmask_dir = os.path.join(src_dir, "unmask")
        os.makedirs(unmask_dir, exist_ok=True)

        ext = os.path.splitext(image_path)[1].lower()
        base_no_ext = os.path.splitext(image_path)[0]

        target_file = base_no_ext + ".webp"
        if os.path.exists(target_file) and os.path.abspath(target_file) != os.path.abspath(image_path):
            target_file = self._unique_path(target_file)

        moved_original = ""
        if ext == ".webp":
            moved_original = self._unique_path(os.path.join(unmask_dir, os.path.basename(image_path)))
            shutil.move(image_path, moved_original)
            src_for_processing = moved_original
            target_file = image_path
        else:
            src_for_processing = image_path

        from PIL import ImageChops

        with Image.open(src_for_processing) as img:
            img_rgba = img.convert("RGBA")
            mask_pil = self._qimage_to_pil_l(mask_qimg)
            # resize to original size (dialog is scaled)
            mask_pil = mask_pil.resize(img_rgba.size, Image.Resampling.NEAREST)

            alpha = img_rgba.getchannel("A")
            keep = Image.eval(mask_pil, lambda v: 0 if v > 0 else 255)  # painted => transparent
            new_alpha = ImageChops.multiply(alpha, keep)
            img_rgba.putalpha(new_alpha)
            img_rgba.save(target_file, "WEBP")

        if ext != ".webp":
            moved_original = self._unique_path(os.path.join(unmask_dir, os.path.basename(image_path)))
            shutil.move(image_path, moved_original)

        return target_file

    def open_stroke_eraser(self):
        if not self.current_image_path:
            return
        try:
            dlg = StrokeEraseDialog(self.current_image_path, self)
        except Exception as e:
            QMessageBox.warning(self, "Stroke Eraser", f"無法載入圖片: {e}")
            return

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        mask_qimg, _w = dlg.get_result()
        try:
            old_path = self.current_image_path
            new_path = self.stroke_erase_to_webp(old_path, mask_qimg)
            if not new_path:
                return
            self._replace_image_path_in_list(old_path, new_path)
            self.load_image()
            self.statusBar().showMessage("Stroke Eraser 完成", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Stroke Eraser", f"失敗: {e}")

    # ==========================
    # Batch: Tagger / LLM
    # ==========================
    def show_progress(self, current, total, name):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{name} ({current}/{total})")
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(True)
            self.btn_cancel_batch.setEnabled(True)

    def hide_progress(self):
        self.progress_bar.setVisible(False)
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        
    def on_batch_done(self, msg="Batch Process Completed"):
        self.hide_progress()
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(False)
            self.btn_cancel_batch.setEnabled(False)
        QMessageBox.information(self, "Batch", msg)
        unload_all_models()

    def run_batch_tagger(self):
        if not self.image_files:
            return
            
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
            
        self.btn_batch_tagger.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)
        self.btn_batch_tagger_to_txt.setEnabled(False)
        self._is_batch_to_txt = False

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_tagger(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_tagger_to_txt(self):
        if not self.image_files:
            return
            
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
        
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_tagger_to_txt.setEnabled(False)
        self.btn_batch_tagger.setEnabled(False)

        # 1. Check Sidecar for cache
        files_to_process = []
        already_done_count = 0

        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                tags_str = sidecar.get("tagger_tags", "")

                if tags_str:
                    # Cache hit: Write directly
                    self.write_batch_result_to_txt(img_path, tags_str, is_tagger=True)
                    already_done_count += 1
                else:
                    # Cache miss: Add to queue
                    files_to_process.append(img_path)

            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 Tagger 結果至 txt", 5000)

            # 2. Process missing files
            if not files_to_process:
                self.btn_batch_tagger_to_txt.setEnabled(True)
                self.btn_batch_tagger.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch Tagger to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 Tagger...", 5000)

            images = create_image_data_list(files_to_process)
            self.pipeline_manager.run_batch_tagger(images)

        except Exception as e:
            self.on_pipeline_error(str(e))


    def run_batch_llm_to_txt(self):
        if not self.image_files:
            return
            
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
            
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.btn_batch_llm.setEnabled(False)

        # 1. 檢查 Sidecar
        files_to_process = []
        already_done_count = 0
        
        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                nl = sidecar.get("nl_pages", [])
                content = nl[-1] if nl and isinstance(nl, list) else ""
                
                if content:
                    self.write_batch_result_to_txt(img_path, content, is_tagger=False)
                    already_done_count += 1
                else:
                    files_to_process.append(img_path)
            
            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 LLM 結果至 txt", 5000)

            # 2. Process missing
            if not files_to_process:
                self.btn_batch_llm_to_txt.setEnabled(True)
                self.btn_batch_llm.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch LLM to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 LLM...", 5000)
            
            user_prompt = self.prompt_edit.toPlainText()
            
            images = create_image_data_list(files_to_process)
            self.pipeline_manager.run_batch_llm(
                 images,
                 prompt_mode=self.current_prompt_mode,
                 user_prompt=user_prompt
            )

        except Exception as e:
            self.on_pipeline_error(str(e))


    def prompt_delete_chars(self) -> bool:
        """回傳 True=刪除, False=保留, None=取消"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch to txt")
        msg.setText("是否自動刪除特徵標籤 (Character Tags)？")
        msg.setInformativeText("將根據設定中的黑白名單過濾標籤或句子。")
        btn_yes = msg.addButton("自動刪除", QMessageBox.ButtonRole.YesRole)
        btn_no = msg.addButton("保留", QMessageBox.ButtonRole.NoRole)
        btn_cancel = msg.addButton(QMessageBox.StandardButton.Cancel)
        
        msg.exec()
        if msg.clickedButton() == btn_yes:
            return True
        elif msg.clickedButton() == btn_no:
            return False
        return None

    def write_batch_result_to_txt(self, image_path, content, is_tagger: bool):
        cfg = self.settings
        delete_chars = getattr(self, "_batch_delete_chars", False)
        mode = cfg.get("batch_to_txt_mode", "append")
        folder_trigger = cfg.get("batch_to_txt_folder_trigger", False)
        
        items = []
        if is_tagger:
            raw_list = [x.strip() for x in content.split(",") if x.strip()]
            if delete_chars:
                raw_list = [t for t in raw_list if not is_basic_character_tag(t, cfg)]
            items = raw_list
        else:
            # nl_content from LLMworker has \n, possibly translation lines in ()
            raw_lines = content.splitlines()
            sentences = []
            for line in raw_lines:
                line = line.strip()
                if not line: continue
                # Skip lines that are entirely in parentheses (translations)
                if (line.startswith("(") and line.endswith(")")) or (line.startswith("（") and line.endswith("）")):
                    continue
                # Remove any remaining content in parentheses/brackets just in case
                line = re.sub(r"[\(（].*?[\)）]", "", line).strip()
                # Remove trailing period and normalize
                line = line.rstrip(".").strip()
                if line:
                    sentences.append(line)
            
            if delete_chars:
                sentences = [s for s in sentences if not is_basic_character_tag(s, cfg)]
            items = sentences

        if folder_trigger:
            trigger = os.path.basename(os.path.dirname(image_path)).strip()
            if trigger and trigger not in items:
                items.insert(0, trigger)

        txt_path = os.path.splitext(image_path)[0] + ".txt"
        existing_content = ""
        if mode == "append" and os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    existing_content = f.read().strip()
            except Exception: pass

        # Deduplication: 若內容已存在於文中 (Word Boundary Check)，則不附加
        if mode == "append" and existing_content and items:
            # Normalize search text
            search_text = existing_content.lower().replace("_", " ").replace("\n", " ")
            search_text = re.sub(r"\s+", " ", search_text)
            
            unique_items = []
            for item in items:
                t_norm = item.strip().lower().replace("_", " ")
                t_norm = re.sub(r"\s+", " ", t_norm)
                if not t_norm: 
                    continue
                
                # Strict whole word check using regex lookbehind/lookahead
                # e.g. "hair" won't match "chair", but "blonde hair" matches "blonde hair girl"
                try:
                    pattern = r"(?<!\w)" + re.escape(t_norm) + r"(?!\w)"
                    if not re.search(pattern, search_text):
                        unique_items.append(item)
                except Exception:
                    # Fallback if regex fails (rare)
                    if t_norm not in search_text:
                        unique_items.append(item)
            
            items = unique_items
            # 如果全部都重複，items 為空，下面邏輯會寫入空字串或變成只寫入分隔符?
            # 最好直接 return 避免寫入多餘的逗號或空行
            if not items:
                return
        
        if is_tagger:
            new_part = ", ".join(items)
            force_lower = cfg.get("english_force_lowercase", True)
            if mode == "append" and existing_content:
                final = cleanup_csv_like_text(existing_content + ", " + new_part, force_lower)
            else:
                final = cleanup_csv_like_text(new_part, force_lower)
        else:
            # For LLM results, now joining with comma and no trailing period
            new_part = ", ".join(items)
            if mode == "append" and existing_content:
                # Use comma or space-comma as separator
                sep = ", "
                if existing_content.endswith(",") or existing_content.endswith("."):
                    sep = " "
                final = existing_content + sep + new_part
            else:
                final = new_part
                
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final)
            if image_path == self.current_image_path:
                self.txt_edit.setPlainText(final)
        except Exception as e:
            print(f"[BatchWriter] 寫入失敗 {txt_path}: {e}")



    
    def use_default_prompt(self):
        """Switch prompt editor to Default Prompt template."""
        self.current_prompt_mode = "default"
        try:
            self.prompt_edit.setPlainText(self.default_user_prompt_template)
        except Exception:
            pass

    def use_custom_prompt(self):
        """Switch prompt editor to Custom Prompt template."""
        self.current_prompt_mode = "custom"
        try:
            self.prompt_edit.setPlainText(self.custom_prompt_template)
        except Exception:
            pass

    def run_batch_llm(self):
        if not self.image_files:
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        user_prompt = self.prompt_edit.toPlainText()
        if "{角色名}" in user_prompt:
             reply = QMessageBox.question(self, "Warning", "Prompt 包含未替換的 '{角色名}'。\n這可能會導致生成結果不正確。\n請手動輸入角色名或調整提示。\n\n確定要繼續嗎？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 return

        self.btn_batch_llm.setEnabled(False)
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.btn_run_llm.setEnabled(False)
        self._is_batch_to_txt = False
        
        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_llm(
                 images,
                 prompt_mode=self.current_prompt_mode,
                 user_prompt=user_prompt
             )
        except Exception as e:
             self.on_pipeline_error(str(e))




    def on_batch_error(self, err):
        self.btn_batch_tagger.setEnabled(True)
        self.btn_batch_tagger_to_txt.setEnabled(True)
        self.btn_auto_tag.setEnabled(True)
        self.btn_batch_llm.setEnabled(True)
        self.btn_batch_llm_to_txt.setEnabled(True)
        self.btn_run_llm.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage(f"Batch Error: {err}", 8000)

    # ==========================
    # Find/Replace
    # ==========================
    def open_find_replace(self):
        dlg = AdvancedFindReplaceDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            settings = dlg.get_settings()
            find_str = settings['find']
            rep_str = settings['replace']
            if not find_str:
                return
            target_files = self.image_files if settings['scope_all'] else [self.current_image_path]
            count = 0
            for img_path in target_files:
                if not img_path:
                    continue
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    flags = 0 if settings['case_sensitive'] else re.IGNORECASE
                    try:
                        # 1. 先執行取代
                        if settings['regex']:
                            new_content, n = re.subn(find_str, rep_str, content, flags=flags)
                            count += n
                        else:
                            if not settings['case_sensitive']:
                                pattern = re.compile(re.escape(find_str), re.IGNORECASE)
                                new_content, n = pattern.subn(rep_str, content)
                                count += n
                            else:
                                n = content.count(find_str)
                                if n > 0:
                                    new_content = content.replace(find_str, rep_str)
                                    count += n
                        
                        # 2. 如果有變動，執行自動格式化 (Format Refresh)
                        if new_content != content:
                            # === 修改重點開始：格式重整 ===
                            # 用逗號分割 -> 去除前後空白 -> 過濾空字串 -> 用 ", " 接回
                            parts = [p.strip() for p in new_content.split(",") if p.strip()]
                            new_content = ", ".join(parts)
                            # === 修改重點結束 ===

                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)

                    except Exception as e:
                        print(f"Replace error in {img_path}: {e}")

            self.load_image() # 重新載入當前圖片以顯示結果
            
            # 嘗試將焦點放回編輯框並捲動到底部 (非必要，但體驗較好)
            try:
                self.txt_edit.moveCursor(QTextCursor.MoveOperation.End)
                self.txt_edit.setFocus()
                self.txt_edit.ensureCursorVisible()
            except Exception:
                pass
                
            QMessageBox.information(self, "Result", f"Replaced {count} occurrences and reformatted.")


    # ==========================
    # Tools: Batch Mask Text (OCR)
    # ==========================
    def _image_has_background_tag(self, image_path: str) -> bool:
        """
        判斷是否為「背景圖」。
        檢查 .txt 分類以及 sidecar JSON 標籤。
        """
        try:
            # 1. 優先檢查 Sidecar JSON (如果有 Tagger 結果就抓得到)
            sidecar = load_image_sidecar(image_path)
            # 檢查 tagger_tags (標籤器結果) 或 tags_context (原本的標籤)
            tags_all = (sidecar.get("tagger_tags", "") + " " + sidecar.get("tags_context", "")).lower()
            if "background" in tags_all:
                return True

            # 2. 檢查 .txt 檔案 (後備方案)
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                return "background" in content
        except Exception:
            pass
        return False

    def run_batch_mask_text(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_mask_text(images)
        except Exception as e:
             self.on_pipeline_error(str(e))


    def run_batch_restore(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        reply = QMessageBox.question(
            self, "Batch Restore",
            "是否確定還原所有圖片的原檔 (若存在)？\n這將會覆蓋/刪除目前的去背版本。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_restore(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def cancel_batch(self):
        self.statusBar().showMessage("正在中止...", 2000)
        # Stop Pipeline
        if self.pipeline_manager.is_running():
            self.pipeline_manager.stop()
            
        # Stop legacy threads
        for attr in ['batch_mask_text_thread']:
             thread = getattr(self, attr, None)
             if thread is not None and thread.isRunning():
                 thread.stop()



    # ==========================
    # Settings
    # ==========================
    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_cfg = dlg.get_cfg()
            self.settings = new_cfg
            save_app_settings(new_cfg)

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
                self.prompt_edit.setPlainText(self.default_user_prompt_template)

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


