# ============================================================
#  Caption 神器
# ============================================================

import sys
import os
import shutil
import csv
import base64
import re
import json
import traceback
import warnings
import inspect
import gc  # 新增垃圾回收支援

# silence some noisy third-party warnings
os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_fwd")
warnings.filterwarnings("ignore", message="Failed to import flet")
warnings.filterwarnings("ignore", message="Token indices sequence length")

# [GPU Fix] 嘗試載入 pip 安裝的 NVIDIA dll
if os.name == 'nt':
    try:
        import nvidia.cudnn
        import nvidia.cublas
        libs = [
            os.path.dirname(nvidia.cudnn.__file__),
            os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin"),
            os.path.dirname(nvidia.cublas.__file__),
            os.path.join(os.path.dirname(nvidia.cublas.__file__), "bin"),
        ]
        for lib in libs:
            if os.path.exists(lib):
                os.add_dll_directory(lib)
    except Exception:
        pass

from pathlib import Path
from io import BytesIO
from urllib.request import urlopen, Request
from natsort import natsorted

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QSplitter,
    QScrollArea, QLineEdit, QDialog, QFormLayout, QComboBox,
    QCheckBox, QMessageBox, QPlainTextEdit, QInputDialog,
    QRadioButton, QGroupBox, QSizePolicy, QTabWidget,
    QFrame, QProgressBar, QSlider, QSpinBox, QDoubleSpinBox,
    QMenu
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, 
    QBuffer, QIODevice, QByteArray, QTimer
)
from PyQt6.QtGui import (
    QPixmap, QKeySequence, QAction, QShortcut, QFont,
    QPalette, QBrush, QPainter, QPen, QColor, QImage, QTextCursor,
    QCursor, QDesktopServices, QIcon
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, 
    QBuffer, QIODevice, QByteArray, QTimer, QUrl
)

from PIL import Image

# optional: clip_anytokenizer for token counting
# 嘗試匯入 transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not found, falling back to regex token counting.")





# [Refactor Imports]
from lib.data import AppSettings, ImageContext, load_app_settings, save_app_settings
from lib.const import (
    DEFAULT_CUSTOM_TAGS, LOCALIZATION, THEME_STYLES, 
    DEFAULT_APP_SETTINGS, TEMPLATE_DEFAULT, DEFAULT_SYSTEM_PROMPT
)
from lib.workers.batch import GenericBatchWorker
from lib.processors.tagger import TaggerProcessor
from lib.processors.llm import LLMProcessor
from lib.processors.vision import UnmaskProcessor, TextMaskProcessor, RestoreProcessor
from lib.utils import (
    create_checkerboard_png_bytes, delete_matching_npz, image_sidecar_json_path,
    load_image_sidecar, save_image_sidecar, backup_raw_image, restore_raw_image,
    has_raw_backup, get_raw_image_dir, delete_raw_backup, ensure_tags_csv,
    load_translations, parse_boorutag_meta, smart_parse_tags, extract_bracket_content,
    is_basic_character_tag, cleanup_csv_like_text, split_csv_like_text,
    try_tags_to_text_list, remove_underline, DanbooruQueryFilter, normalize_for_match
)
from lib.services.tagger import call_wd14
from lib.services.common import unload_all_models

# Optional: OCR text detection (used in MainWindow.mask_text_current_image)
try:
    from imgutils.ocr import detect_text_with_ocr
except ImportError:
    detect_text_with_ocr = None

# Optional: CLIPTokenizer for token counting
try:
    from clip_anytokenizer import CLIPTokenizer
except ImportError:
    CLIPTokenizer = None


os.environ['ONNX_MODE'] = 'gpu'

# ==========================================
#  Configuration & Globals
# Note: Utility functions (load_app_settings, call_wd14, etc.) 
# have been moved to lib.utils, lib.services, and lib.data.



# Danbooru filter and Utilities have been moved to lib.utils


# ==========================================
#  Main Window
# ==========================================
# UI classes (StrokeCanvas, StrokeEraseDialog, TagButton, TagFlowWidget,
# AdvancedFindReplaceDialog, SettingsDialog) have been moved to lib.ui

# Import Mixins
from lib.ui.main_window.mixins.shortcuts_mixin import ShortcutsMixin
from lib.ui.main_window.mixins.theme_mixin import ThemeMixin
from lib.ui.main_window.mixins.nl_mixin import NLMixin
from lib.ui.main_window.mixins.dialogs_mixin import DialogsMixin
from lib.ui.main_window.mixins.progress_mixin import ProgressMixin
from lib.ui.main_window.mixins.file_mixin import FileMixin
from lib.ui.main_window.mixins.filter_mixin import FilterMixin
from lib.ui.main_window.mixins.navigation_mixin import NavigationMixin

class MainWindow(ShortcutsMixin, ThemeMixin, NLMixin, DialogsMixin, ProgressMixin, 
                 FileMixin, FilterMixin, NavigationMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Captioning Assistant")
        self._clip_tokenizer = None
        self.resize(1600, 1000)

        self.settings = load_app_settings()
        self.app_settings = AppSettings(self.settings)

        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", DEFAULT_APP_SETTINGS["llm_api_key"]))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        self.default_user_prompt_template = self.app_settings.user_prompt_template
        # self.custom_prompt_template = ... [Removed]
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

        self.batch_tagger_thread = None
        self.batch_llm_thread = None
        self.batch_unmask_thread = None
        self.batch_mask_text_thread = None
        self.tagger_thread = None # Add missing single threads
        self.llm_thread = None


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

    # tr() moved to ThemeMixin

    # apply_theme() moved to ThemeMixin

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

        self.btn_reset_prompt = QPushButton(self.tr("btn_reset_prompt"))
        self.btn_reset_prompt.setToolTip("重置為目前設定的 Prompt 模板內容")
        self.btn_reset_prompt.clicked.connect(self.on_reset_prompt)
        nl_toolbar.addWidget(self.btn_reset_prompt)
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

    # setup_shortcuts() moved to ShortcutsMixin

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
    # refresh_file_list() moved to Mixin


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

    # apply_filter() moved to Mixin


    # clear_filter() moved to Mixin


    # next_image() moved to Mixin


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

    # keyPressEvent() and keyReleaseEvent() moved to ShortcutsMixin

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

    # delete_current_image() moved to Mixin


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

    # refresh_nl_tab() moved to NLMixin


    # ==========================
    # NL paging / dynamic sizing
    # ==========================
    # set_current_nl_page() moved to NLMixin

    # update_nl_page_controls() moved to NLMixin

    # prev_nl_page() and next_nl_page() moved to NLMixin

    # update_nl_result_height() moved to NLMixin

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
        self.btn_auto_tag.setEnabled(False)
        self.btn_auto_tag.setText("Tagging...")

        # [Refactor] Use GenericBatchWorker for single image
        ctx = ImageContext(self.current_image_path)
        # Check settings for tagger? TaggerProcessor uses self.settings internally
        proc = TaggerProcessor(self.app_settings)
        
        self.tagger_thread = GenericBatchWorker([ctx], proc)
        # Adapt output to match legacy slot: item_done(ctx, result) -> on_tagger_finished(result)
        self.tagger_thread.item_done.connect(lambda ctx, res: self.on_tagger_finished(res))
        self.tagger_thread.error.connect(self.on_tagger_error_no_popup)
        # We don't need finished_all for single item logic if we use item_done, 
        # but cleanup happens in finished_all usually? 
        # Actually worker cleans up processor in run() after loop.
        self.tagger_thread.start()

    def run_llm_generation(self):
        if not self.current_image_path:
            return

        tags_text = self.build_llm_tags_context_for_image(self.current_image_path)
        user_prompt = self.prompt_edit.toPlainText()

        self.btn_run_llm.setEnabled(False)
        self.btn_run_llm.setText("Running LLM...")

        # Check empty tags logic
        if "{tags}" in user_prompt and not tags_text.strip():
            reply = QMessageBox.question(
                self, "Warning", 
                "Prompt 包含 {tags} 但目前沒有標籤資料。\n確定要繼續嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.btn_run_llm.setEnabled(True)
                self.btn_batch_llm.setEnabled(True)
                return

        # [Refactor] Use GenericBatchWorker for single image
        ctx = ImageContext(self.current_image_path)
        
        # Determine strict settings usage. Legacy used self.settings dict.
        # Now using self.app_settings and processor
        proc = LLMProcessor(self.app_settings, override_user_prompt=user_prompt, is_batch=False)
        
        self.llm_thread = GenericBatchWorker([ctx], proc)
        # item_done -> (ctx, result_content)
        self.llm_thread.item_done.connect(lambda ctx, res: self.on_llm_finished_latest_only(res))
        self.llm_thread.error.connect(self.on_llm_error_no_popup)
        self.llm_thread.start()

    def on_tagger_error_no_popup(self, e):
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText("Auto Tag (WD14)")
        self.statusBar().showMessage(f"Tagger error: {e}", 5000)

    def on_tagger_finished(self, raw_tags_str):
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText("Auto Tag (WD14)")

        self.save_tagger_tags_for_image(self.current_image_path, raw_tags_str)

        parts = [x.strip() for x in raw_tags_str.split(",") if x.strip()]
        parts = try_tags_to_text_list(parts)
        parts = [t.replace("_", " ").strip() for t in parts if t.strip()]

        self.tagger_tags = parts
        self.refresh_tags_tab()
        self.on_text_changed()



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



    @staticmethod
    def extract_llm_content_and_postprocess(full_text: str, force_lowercase: bool = True) -> str:
        pattern = r"===處理結果開始===(.*?)===處理結果結束==="
        match = re.search(pattern, full_text, re.DOTALL)
        if not match:
            return ""

        content = match.group(1).strip()
        lines = [l.rstrip() for l in content.splitlines() if l.strip()]

        out_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("(") or s.startswith("（"):
                out_lines.append(s)
                continue

            s = s.replace(", ", " ")
            s = s.rstrip(".").strip()
            if force_lowercase:
                s = s.lower()
            out_lines.append(s)

        return "\n".join(out_lines)

    def on_llm_finished_latest_only(self, full_text):
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText("Run LLM")

        content = self.extract_llm_content_and_postprocess(full_text, self.english_force_lowercase)
        if not content:
            self.statusBar().showMessage("LLM Parser: 找不到 === markers 或內容為空", 5000)
            print(full_text)
            return

        self.save_nl_for_image(self.current_image_path, content)

        if not self.nl_pages:
            self.nl_pages = []
        self.nl_pages.append(content)
        self.nl_page_index = len(self.nl_pages) - 1
        self.nl_latest = content
        self.refresh_nl_tab()
        self.update_nl_page_controls()
        self.on_text_changed()

    def on_llm_error_no_popup(self, err):
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText("Run LLM")
        self.statusBar().showMessage(f"LLM Error: {err}", 8000)


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
        
        try:
            from transparent_background import Remover
        except ImportError:
            QMessageBox.warning(self, "Unmask", "transparent_background.Remover not available")
            return
            
        # [Refactor] Use GenericBatchWorker for single image with is_batch=False
        ctx = ImageContext(self.current_image_path)
        proc = UnmaskProcessor(self.app_settings, is_batch=False)
        
        self.batch_unmask_thread = GenericBatchWorker([ctx], proc)
        self.batch_unmask_thread.progress.connect(self.show_progress)
        self.batch_unmask_thread.item_done.connect(self.on_batch_unmask_per_image)
        self.batch_unmask_thread.finished_all.connect(lambda: self.on_batch_done("單圖去背完成"))
        self.batch_unmask_thread.error.connect(lambda e: QMessageBox.warning(self, "Error", f"Unmask 失敗: {e}"))
        self.batch_unmask_thread.start()

    def mask_text_current_image(self):
        if not self.current_image_path:
            return
        if not bool(self.app_settings.get("mask_batch_detect_text_enabled", True)):
            QMessageBox.information(self, "Info", "OCR text detection is disabled in settings.")
            return

        if detect_text_with_ocr is None:
             QMessageBox.warning(self, "Mask Text", self.tr("setting_mask_ocr_hint"))
             return

        # [Refactor] Use GenericBatchWorker for single image with is_batch=False
        ctx = ImageContext(self.current_image_path)
        proc = TextMaskProcessor(self.app_settings, is_batch=False)
        
        self.batch_mask_text_thread = GenericBatchWorker([ctx], proc)
        self.batch_mask_text_thread.progress.connect(self.show_progress)
        self.batch_mask_text_thread.item_done.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.finished_all.connect(lambda: self.on_batch_done("Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Mask Text", e))
        self.batch_mask_text_thread.start()

    def restore_current_image(self):
        """還原當前圖片為原始備份 (從 raw_image 資料夾)"""
        if not self.current_image_path:
            return
        
        # [Refactor] Use GenericBatchWorker + RestoreProcessor
        ctx = ImageContext(self.current_image_path)
        proc = RestoreProcessor(self.app_settings)
        
        self.batch_restore_thread = GenericBatchWorker([ctx], proc)
        self.batch_restore_thread.item_done.connect(lambda c, r: self.load_image())
        self.batch_restore_thread.error.connect(lambda e: self.statusBar().showMessage(f"Restore failed: {e}", 3000))
        self.batch_restore_thread.start()

    def run_batch_unmask_background(self):
        if not self.image_files:
            return
        
        # Check basic import availability (fast)
        try:
            from transparent_background import Remover
        except ImportError:
            QMessageBox.warning(self, "Batch Unmask", "transparent_background.Remover not available")
            return

        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(False)
            
        # [Refactor] Use GenericBatchWorker + UnmaskProcessor
        # Processor handles filtering internally based on settings
        contexts = [ImageContext(p) for p in self.image_files]
        proc = UnmaskProcessor(self.app_settings)
        
        self.batch_unmask_thread = GenericBatchWorker(contexts, proc)
        self.batch_unmask_thread.progress.connect(self.show_progress)
        self.batch_unmask_thread.item_done.connect(self.on_batch_unmask_per_image)
        self.batch_unmask_thread.finished_all.connect(self.on_batch_unmask_done)
        self.batch_unmask_thread.error.connect(self.on_batch_error)
        self.batch_unmask_thread.start()

    def on_batch_unmask_per_image(self, old_path: str, new_path: str):
        self._replace_image_path_in_list(old_path, new_path)

    def on_batch_unmask_done(self):
        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(True)
        self.hide_progress()
        self.load_image()
        self.statusBar().showMessage("Batch Unmask 完成", 5000)

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

    # stroke_erase_to_webp() moved to DialogsMixin


    # show_progress() moved to ProgressMixin


    # hide_progress() moved to ProgressMixin


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

        self.btn_batch_tagger.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)

        # [Refactor] Use GenericBatchWorker + TaggerProcessor
        contexts = [ImageContext(p) for p in self.image_files]
        proc = TaggerProcessor(self.app_settings)
        
        self.batch_tagger_thread = GenericBatchWorker(contexts, proc)
        self.batch_tagger_thread.progress.connect(self.show_progress)
        self.batch_tagger_thread.item_done.connect(self.on_batch_tagger_per_image)
        self.batch_tagger_thread.finished_all.connect(self.on_batch_tagger_done)
        self.batch_tagger_thread.error.connect(self.on_batch_error)
        self.batch_tagger_thread.start()

    def run_batch_tagger_to_txt(self):
        if not self.image_files:
            return
        
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_tagger_to_txt.setEnabled(False)

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
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch Tagger to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 Tagger...", 5000)

            # [Refactor] Use GenericBatchWorker + TaggerProcessor
            contexts = [ImageContext(p) for p in files_to_process]
            proc = TaggerProcessor(self.app_settings)

            self.batch_tagger_thread = GenericBatchWorker(contexts, proc)
            self.batch_tagger_thread.progress.connect(self.show_progress)
            self.batch_tagger_thread.item_done.connect(self.on_batch_tagger_per_image)
            self.batch_tagger_thread.finished_all.connect(self.on_batch_tagger_done)
            self.batch_tagger_thread.error.connect(self.on_batch_error)
            self.batch_tagger_thread.start()

        except Exception as e:
            self.btn_batch_tagger_to_txt.setEnabled(True)
            self._is_batch_to_txt = False
            QMessageBox.warning(self, "Error", f"Batch Processing Error: {e}")

    def run_batch_llm_to_txt(self):
        if not self.image_files:
            return
            
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_llm_to_txt.setEnabled(False)

        # 1. 檢查 Sidecar，將已有結果者直接寫入 txt
        files_to_process = []
        already_done_count = 0
        
        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                nl = sidecar.get("nl_pages", [])
                
                content = ""
                # 使用最後一次結果 (User request: "LLM用最後一次結果")
                if nl and isinstance(nl, list):
                    content = nl[-1]
                
                if content:
                    # 已有結果 -> 直接寫入
                    self.write_batch_result_to_txt(img_path, content, is_tagger=False)
                    already_done_count += 1
                else:
                    # 無結果 -> 加入待處理清單
                    files_to_process.append(img_path)
            
            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 LLM 結果至 txt", 5000)

            # 2. 針對無結果的檔案，執行 Batch LLM
            if not files_to_process:
                # 全部都有結果，直接結束
                self.btn_batch_llm_to_txt.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch LLM to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            # 有缺漏 -> 跑 Batch LLM
            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 LLM...", 5000)
            
            user_prompt = self.prompt_edit.toPlainText()

            # 建立 Worker，只針對 files_to_process
            # [Refactor] Use GenericBatchWorker + LLMProcessor
            contexts = [ImageContext(p) for p in files_to_process]
            # Batch mode implies using template or stored prompt. 
            # The legacy code passed `user_prompt` from editor.
            # LLMProcessor supports override_user_prompt.
            proc = LLMProcessor(self.app_settings, override_user_prompt=user_prompt)

            self.batch_llm_thread = GenericBatchWorker(contexts, proc)
            self.batch_llm_thread.progress.connect(self.show_progress)
            self.batch_llm_thread.item_done.connect(self.on_batch_llm_per_image)
            self.batch_llm_thread.finished_all.connect(self.on_batch_llm_done)
            self.batch_llm_thread.error.connect(self.on_batch_error)
            self.batch_llm_thread.start()

        except Exception as e:
            self.btn_batch_llm_to_txt.setEnabled(True)
            self._is_batch_to_txt = False
            QMessageBox.warning(self, "Error", f"Batch Processing Error: {e}")
            return

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

    def on_batch_tagger_per_image(self, image_path, raw_tags_str):
        self.save_tagger_tags_for_image(image_path, raw_tags_str)

        if image_path == self.current_image_path:
            parts = [x.strip() for x in raw_tags_str.split(",") if x.strip()]
            parts = try_tags_to_text_list(parts)
            parts = [t.replace("_", " ").strip() for t in parts if t.strip()]
            self.tagger_tags = parts
            self.refresh_tags_tab()
            self.on_text_changed()
        
        if getattr(self, "_is_batch_to_txt", False):
            self.write_batch_result_to_txt(image_path, raw_tags_str, is_tagger=True)

    def on_batch_tagger_done(self):
        self.btn_batch_tagger.setEnabled(True)
        self.btn_batch_tagger_to_txt.setEnabled(True)
        self.btn_auto_tag.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage("Batch Tagger 完成", 5000)

    


    def run_batch_llm(self):
        if not self.image_files:
            return

        self.btn_batch_llm.setEnabled(False)
        self.btn_run_llm.setEnabled(False)

        user_prompt = self.prompt_edit.toPlainText()

        # Check for unreplaced placeholder {角色名}
        if "{角色名}" in user_prompt:
            reply = QMessageBox.question(
                self, "Warning", 
                "Prompt 包含未替換的 '{角色名}'。\n這可能會導致生成結果不正確。\n請手動輸入角色名或調整提示。\n\n確定要繼續嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.btn_batch_llm.setEnabled(True)
                self.btn_run_llm.setEnabled(True)
                return

        # [Refactor] Use GenericBatchWorker + LLMProcessor
        contexts = [ImageContext(p) for p in self.image_files]
        proc = LLMProcessor(self.app_settings, override_user_prompt=user_prompt)
        
        self.batch_llm_thread = GenericBatchWorker(contexts, proc)
        self.batch_llm_thread.progress.connect(self.show_progress)
        self.batch_llm_thread.item_done.connect(self.on_batch_llm_per_image)
        self.batch_llm_thread.finished_all.connect(self.on_batch_llm_done)
        self.batch_llm_thread.error.connect(self.on_batch_error)
        self.batch_llm_thread.start()

    def on_batch_llm_per_image(self, image_path, nl_content):
        if not nl_content:
            return
        self.save_nl_for_image(image_path, nl_content)
        if image_path == self.current_image_path:
            if not self.nl_pages:
                self.nl_pages = []
            self.nl_pages.append(nl_content)
            self.nl_page_index = len(self.nl_pages) - 1
            self.nl_latest = nl_content
            self.refresh_nl_tab()
            self.update_nl_page_controls()
            self.on_text_changed()

        if getattr(self, "_is_batch_to_txt", False):
            self.write_batch_result_to_txt(image_path, nl_content, is_tagger=False)

    def on_batch_llm_done(self):
        self.btn_batch_llm.setEnabled(True)
        self.btn_batch_llm_to_txt.setEnabled(True)
        self.btn_run_llm.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage("Batch LLM 完成", 5000)

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
    # open_find_replace() moved to DialogsMixin


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

    def on_batch_mask_text_per_image(self, old_path, new_path):
        if old_path != new_path:
            self._replace_image_path_in_list(old_path, new_path)
            # If current image is the one processed, reload it
            if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
                self.load_image()

    def run_batch_mask_text(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        # [Refactor] Use GenericBatchWorker + TextMaskProcessor
        contexts = [ImageContext(p) for p in self.image_files]
        proc = TextMaskProcessor(self.app_settings)
        
        # Check settings
        if not bool(self.app_settings.get("mask_batch_detect_text_enabled", True)):
             QMessageBox.warning(self, "Warning", "Batch Text detection is disabled in settings.")
             return

        self.batch_mask_text_thread = GenericBatchWorker(contexts, proc)
        self.batch_mask_text_thread.progress.connect(lambda i, t, name: self.show_progress(i, t, name))
        self.batch_mask_text_thread.item_done.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.finished_all.connect(lambda: self.on_batch_done("Batch Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Batch Mask Text", e))
        self.batch_mask_text_thread.start()

    def on_batch_restore_per_image(self, old_path, new_path):
        if old_path != new_path:
            self._replace_image_path_in_list(old_path, new_path)
            # Reload if current image is affected
            if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
                self.load_image()

    def run_batch_restore(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        reply = QMessageBox.question(
            self, "Batch Restore",
            "是否確定還原所有圖片的原檔 (若存在)？\n這將會覆蓋/刪除目前的去背版本。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # [Refactor] Use GenericBatchWorker + RestoreProcessor
        contexts = [ImageContext(p) for p in self.image_files]
        proc = RestoreProcessor(self.app_settings)

        self.batch_restore_thread = GenericBatchWorker(contexts, proc)
        # Using lambda for progress signature matching
        self.batch_restore_thread.progress.connect(lambda i, t, name: self.show_progress(i, t, name))
        self.batch_restore_thread.item_done.connect(lambda ctx, res: self.on_batch_restore_per_image(ctx.path, res))
        self.batch_restore_thread.finished_all.connect(lambda: self.on_batch_done("Batch Restore 完成"))
        self.batch_restore_thread.error.connect(lambda e: self.on_batch_error("Batch Restore", e))
        self.batch_restore_thread.start()

    # cancel_batch() moved to ProgressMixin


    def on_reset_prompt(self):
        """Reset prompt editor to the currently active template."""
        try:
            # Re-fetch from app_settings to ensure latest
            self.default_user_prompt_template = self.app_settings.user_prompt_template
            self.prompt_edit.setPlainText(self.default_user_prompt_template)
            self.statusBar().showMessage(self.tr("msg_prompt_reset"), 2000)
        except Exception:
            pass

    # retranslate_ui() moved to ThemeMixin

    # _setup_menus() moved to ThemeMixin


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())