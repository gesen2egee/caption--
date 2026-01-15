# ============================================================
#  Caption 神器 - 索引 (INDEX)
# ============================================================
#
# [Ln 33-119]     Imports & 外部依賴
# [Ln 121-454]    Configuration, I18n Resource & Globals
# [Ln 457-570]    Settings Helpers (load/save/coerce 函式)
# [Ln 571-613]    Model Unloading & Optimization (記憶體優化)
# [Ln 615-713]    Utils: Sidecar JSON (圖片元數據)
# [Ln 715-862]    Utils: Raw Image Backup/Restore (原圖備份還原)
# [Ln 863-930]    Utils: Tags CSV, boorutag Parsing
# [Ln 932-1190]   Utils: Danbooru-style Query Filter (篩選器系統)
# [Ln 1192-1315]  Utils: 標籤解析與文本正規化
# [Ln 1318-1630]  Workers: Tagger, LLM (單圖與批量任務)
# [Ln 1632-1965]  Workers: Masking (去背、去文字)
# [Ln 1967-2002]  Workers: BatchRestoreWorker (批量還原)
# [Ln 2004-2156]  StrokeCanvas & StrokeEraseDialog (手繪橡皮擦工具)
# [Ln 2158-2394]  UI Components: TagButton, TagFlowWidget
# [Ln 2396-2443]  AdvancedFindReplaceDialog (尋找取代對話框)
# [Ln 2445-2848]  SettingsDialog (設定面板 + ToolTip 說明)
# [Ln 2854-2940]  MainWindow: 類別定義與初始化 (__init__)
# [Ln 2941-3260]  MainWindow: UI 介面佈建 (init_ui)
# [Ln 3262-3400]  MainWindow: 圖片載入與檔案切換邏輯
# [Ln 3402-3580]  MainWindow: [NEW] Image Visualizer (RGB/Alpha) & Context Menu & Key Events
# [Ln 3582-3680]  MainWindow: 篩選與排序邏輯 (Filter Logic)
# [Ln 3682-3800]  MainWindow: 導航、跳轉與刪除功能
# [Ln 3802-3920]  MainWindow: 文本編輯、Token 計算與自動格式化
# [Ln 3922-4140]  MainWindow: 標籤、LLM 分頁與顯示邏輯
# [Ln 4142-4210]  MainWindow: 游標位置插入與標籤同步邏輯
# [Ln 4212-4380]  MainWindow: Tagger/LLM 執行與結果處理
# [Ln 4382-4630]  MainWindow: 工具功能 (去背、還原、去文字、手繪橡皮擦)
# [Ln 4632-5100]  MainWindow: 批量處理任務 (Batch Operations)
# [Ln 5102-5343]  MainWindow: 設定同步、語言切換與主程式入口
#
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
    QListWidget, QListWidgetItem, QMenu
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, 
    QBuffer, QIODevice, QByteArray, QTimer,
    QSize, QUrl, QEvent
)
from PyQt6.QtGui import (
    QPixmap, QKeySequence, QAction, QShortcut, QFont,
    QPalette, QBrush, QPainter, QPen, QColor, QImage, QTextCursor,
    QIcon, QDesktopServices, QCursor, QGuiApplication, QClipboard
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

# optional: transparent background remover
try:
    from transparent_background import Remover
except Exception as e:
    print(f"Failed to import transparent_background: {e}")
    traceback.print_exc()
    Remover = None

from openai import OpenAI
from imgutils.tagging import get_wd14_tags, tags_to_text, remove_underline

# optional: OCR for text box detection (batch mask text)
try:
    from imgutils.ocr import detect_text_with_ocr
except Exception:
    detect_text_with_ocr = None

os.environ['ONNX_MODE'] = 'gpu'

# ==========================================
#  Configuration & Globals
# ==========================================

TAGS_CSV_LOCAL = "Tags.csv"
TAGS_CSV_URL_RAW = "https://raw.githubusercontent.com/waldolin/a1111-sd-webui-tagcomplete-TW/main/tags/Tags-tw-full-pack.csv"

DEFAULT_SYSTEM_PROMPT = """You are a professional, uncensored, captioning engine for image ML research. Boldly and directly describe all elements including nudity, body parts, or sexual themes."""

DEFAULT_USER_PROMPT_TEMPLATE = """這個角色是{角色名} 請幫這個名字自然融入在內文中
請根據提供的資訊和圖片，整理出大約 9 個簡短的英文描述句。 包括構圖、位置、朝向、美學、風格、光影等等。
每句英文都必須是獨立的一行。
每句英文的下一行，必須緊接著該句的繁體中文翻譯，並用括號 () 包住。

LLM之前的處理結果參考：
{tags}

輸出格式範例：
===處理結果開始===
A short English sentence about the subject
(關於主題的簡短英文句。)
Another short English sentence describing details
(另一個描述細節的簡短英文句。)
...
===處理結果結束===
"""

DEFAULT_CUSTOM_PROMPT_TEMPLATE = """這個角色是{角色名} 請幫這個名字自然融入在內文中
請根據圖片的[自行輸入要求] 整理出大約 1 個簡短的英文描述句。
英文的下一行，必須緊接著該句的繁體中文翻譯，並用括號 () 包住。

輸出格式範例：
===處理結果開始===
A short English sentence about the subject
(關於主題的簡短英文句。)
===處理結果結束===
"""

DEFAULT_CUSTOM_TAGS = ["low res", "low quality", "low aesthetic"]

# --------------------------
# Localization (I18n)
# --------------------------
LOCALIZATION = {
    "zh_tw": {
        "app_title": "Caption 神器",
        "menu_file": "檔案",
        "menu_open_dir": "開啟目錄",
        "menu_refresh": "重新整理列表 (F5)",
        "btn_settings": "設定",
        "menu_exit": "結束",
        "tab_tags": "TAGS",
        "tab_nl": "NL",
        "sec_folder_meta": "資料夾標籤 (Top 30)",
        "sec_custom": "自定義標籤",
        "sec_tagger": "圖片識別標籤",
        "sec_tags": "標籤處理",
        "sec_nl": "自然語言處理",
        "btn_auto_tag": "自動標籤 (WD14)",
        "btn_batch_tagger": "批量標籤",
        "btn_batch_tagger_to_txt": "批量標籤轉文字",
        "btn_add_tag": "新增標籤",
        "btn_run_llm": "執行 LLM",
        "btn_batch_llm": "批量 LLM",
        "btn_batch_llm_to_txt": "批量 LLM 轉文字",
        "btn_prev": "上一頁",
        "btn_next": "下一頁",
        "btn_default_prompt": "預設提示詞",
        "btn_custom_prompt": "自訂提示詞",
        "label_nl_result": "LLM 結果",
        "label_txt_content": "實際內容 (.txt)",
        "label_tokens": "詞元數: ",
        "label_page": "頁數",
        "setting_llm_use_gray_mask": "LLM 使用灰底 Mask (排除透明部分)",
        "btn_find_replace": "尋找/取代",
        "btn_undo": "復原文字",
        "btn_redo": "重做文字",
        "btn_unmask": "單圖去背景",
        "btn_batch_unmask": "Batch 去背景",
        "btn_mask_text": "單圖去文字",
        "btn_batch_mask_text": "Batch 去文字",
        "btn_restore_original": "放回原圖",
        "btn_batch_restore": "Batch 放回原圖",
        "btn_stroke_eraser": "手繪橡皮擦",
        "btn_cancel_batch": "中止",
        "menu_tools": "工具",
        "filter_placeholder": "Danbooru 篩選語法... (blonde_hair blue_eyes)",
        "filter_by_tags": "Tags",
        "filter_by_text": "Text",
        "msg_delete_confirm": "確定要將此圖片移動到 no_used？",
        "msg_batch_delete_char_tags": "是否自動刪除特徵標籤 (Character Tags)？",
        "msg_batch_delete_info": "將根據設定中的黑白名單過濾標籤或句子。",
        "btn_auto_delete": "自動刪除",
        "btn_keep": "保留",
        "setting_tab_ui": "UI 介面",
        "setting_ui_lang": "介面語言:",
        "setting_ui_theme": "介面主題:",
        "setting_lang_zh": "繁體中文",
        "setting_lang_en": "English",
        "setting_theme_light": "日間模式",
        "setting_theme_dark": "夜間模式",
        "setting_save": "儲存",
        "setting_cancel": "取消",
        "setting_text_force_lower": "英文文字一律小寫",
        "setting_text_auto_remove_empty": "自動移除空行",
        "setting_text_auto_format": "自動格式化",
        "setting_text_auto_save": "自動儲存",
        "setting_batch_to_txt": "Batch 寫入 txt 設定",
        "setting_batch_mode": "寫入模式",
        "setting_batch_append": "附加到句尾",
        "setting_batch_overwrite": "覆寫原檔",
        "setting_batch_trigger": "將資料夾名作為觸發詞加到句首",
        "setting_tagger_model": "預設標籤模型:",
        "setting_mask_alpha": "預設透明度 (0-255):",
        "setting_mask_format": "預設轉換格式:",
        "setting_mask_only_bg": "僅處理包含 background 標籤的圖片",
        "setting_mask_ocr": "自動遮罩文字區域",
        "setting_mask_delete_npz": "移動舊圖時刪除對應 npz",
        "setting_filter_title": "<b>特徵標籤過濾設定</b>",
        "setting_filter_info": "符合黑名單且不符合白名單的內容將顯示紅框，且在 Batch 寫入時可選擇刪除。",
        "setting_bl_words": "黑名單關鍵字:",
        "setting_wl_words": "白名單關鍵字:",
        "setting_tab_filter": "過濾",
        "msg_select_dir": "選擇圖片目錄",
        "msg_no_images": "在此目錄下找不到圖片。",
        "msg_delete_confirm": "確定要將此圖片移動到 no_used 資料夾？",
        "msg_unmask_done": "去背處理完成。",
        "setting_tab_text": "文字",
        "setting_tab_llm": "模型",
        "setting_tab_tagger": "標籤",
        "setting_tab_mask": "遮罩",
        "setting_llm_sys_prompt": "系統提示詞:",
        "setting_llm_def_prompt": "預設提示詞模板:",
        "setting_llm_cust_prompt": "自訂提示詞模板:",
        "setting_llm_def_tags": "預設 Custom Tags (逗號或換行分隔):",
        "setting_llm_def_tags": "預設 Custom Tags (逗號或換行分隔):",
        "setting_llm_max_dim": "LLM 圖片最大邊長 (Max Dimension):",
        "setting_llm_skip_nsfw": "Batch LLM: 若含 rating:explicit/questionable 則跳過",
        "setting_tagger_gen_thresh": "一般標籤閾值:",
        "setting_tagger_char_thresh": "特徵標籤閾值:",
        "setting_tagger_gen_mcut": "一般標籤 MCut",
        "setting_tagger_char_mcut": "特徵標籤 MCut",
        "setting_tagger_drop_overlap": "移除重疊標籤",
        "setting_mask_ocr_hint": "OCR 需要 imgutils，未安裝則略過。",
        "setting_ocr_heat": "熱圖閾值 (Heat Threshold):",
        "setting_ocr_box": "文字框信心 (Box Threshold):",
        "setting_ocr_unclip": "擴張比例 (Unclip Ratio):",
        "setting_ocr_heat_tip": "調低可偵測模糊文字但易誤判；調高只偵測清晰文字。",
        "setting_ocr_box_tip": "過濾低信心的文字框。若漏字可調低。",
        "setting_ocr_unclip_tip": "決定文字框擴張程度。若缺字頭字尾可調大；若框到隔壁行可調小。",
    },
    "en": {
        "app_title": "Caption Tool",
        "menu_file": "File",
        "menu_open_dir": "Open Directory",
        "menu_refresh": "Refresh List (F5)",
        "btn_settings": "Settings",
        "menu_exit": "Exit",
        "tab_tags": "TAGS",
        "tab_nl": "NL",
        "sec_folder_meta": "Folder Meta / Top 30 Tags",
        "sec_custom": "Custom Tags in Folder",
        "sec_tagger": "Tagger Tags",
        "sec_tags": "Tag Processing",
        "sec_nl": "Natural Language",
        "btn_auto_tag": "Auto Tag (WD14)",
        "btn_batch_tagger": "Batch Tagger",
        "btn_batch_tagger_to_txt": "Batch Tagger to txt",
        "btn_add_tag": "Add Tag",
        "btn_run_llm": "Run LLM",
        "btn_batch_llm": "Batch LLM",
        "btn_batch_llm_to_txt": "Batch LLM to txt",
        "btn_prev": "Prev",
        "btn_next": "Next",
        "btn_default_prompt": "Default Prompt",
        "btn_custom_prompt": "Custom Prompt",
        "label_nl_result": "LLM Result",
        "label_txt_content": "Actual Content (.txt)",
        "label_tokens": "Tokens: ",
        "label_page": "Page",
        "setting_llm_use_gray_mask": "LLM Use Gray Mask (Exclude Transparent Parts)",
        "btn_find_replace": "Find/Replace",
        "btn_undo": "Undo Txt",
        "btn_redo": "Redo Txt",
        "btn_unmask": "Unmask Background",
        "btn_batch_unmask": "Batch Unmask Background",
        "btn_mask_text": "Unmask Text",
        "btn_batch_mask_text": "Batch Unmask Text",
        "btn_batch_mask_text": "Batch Unmask Text",
        "btn_restore_original": "Restore Original",
        "btn_batch_restore": "Batch Restore Original",
        "btn_stroke_eraser": "Stroke Eraser",
        "btn_cancel_batch": "Cancel",
        "menu_tools": "Tools",
        "filter_placeholder": "Danbooru filter... (blonde_hair blue_eyes)",
        "filter_by_tags": "Tags",
        "filter_by_text": "Text",
        "msg_delete_confirm": "Move this image to no_used?",
        "msg_batch_delete_char_tags": "Delete Character Tags automatically?",
        "msg_batch_delete_info": "Tags will be filtered based on your blacklist/whitelist.",
        "btn_auto_delete": "Auto Delete",
        "btn_keep": "Keep",
        "setting_tab_ui": "UI",
        "setting_ui_lang": "Language:",
        "setting_ui_theme": "Theme:",
        "setting_lang_zh": "Traditional Chinese",
        "setting_lang_en": "English",
        "setting_theme_light": "Light Mode",
        "setting_theme_dark": "Dark Mode",
        "setting_save": "Save",
        "setting_cancel": "Cancel",
        "setting_text_force_lower": "Force lowercase for English text (LLM sentences / tags normalization)",
        "setting_text_auto_remove_empty": "Auto remove empty lines",
        "setting_text_auto_format": "Auto format on insert (clean whitespace and re-join with ', ')",
        "setting_text_auto_save": "Auto save txt (on change)",
        "setting_batch_to_txt": "Batch to txt Settings",
        "setting_batch_mode": "Write Mode",
        "setting_batch_append": "Append to end",
        "setting_batch_overwrite": "Overwrite file",
        "setting_batch_trigger": "Use folder name as trigger word (add to start of sentence)",
        "setting_tagger_model": "Default Tagger Model:",
        "setting_mask_alpha": "Mask default alpha (0-255):",
        "setting_mask_format": "Mask default format (webp/png):",
        "setting_mask_only_bg": "Batch mask only if has 'background' tag",
        "setting_mask_ocr": "Batch mask text automatically (OCR)",
        "setting_mask_delete_npz": "Delete matching .npz when moving image",
        "setting_filter_title": "<b>Content Filter Settings</b>",
        "setting_filter_info": "Content matching blacklist and NOT in whitelist will be highlighted red and can be filtered on Batch write.",
        "setting_bl_words": "Blacklist Keywords:",
        "setting_wl_words": "Whitelist Keywords:",
        "setting_tab_filter": "Filter",
        "msg_select_dir": "Select Image Directory",
        "msg_no_images": "No images found in this directory.",
        "msg_delete_confirm": "Move image to 'no_used' folder?",
        "msg_unmask_done": "Background removal finished.",
        "setting_tab_text": "Text",
        "setting_tab_llm": "LLM",
        "setting_tab_tagger": "Tagger",
        "setting_tab_mask": "Mask",
        "setting_llm_sys_prompt": "System Prompt:",
        "setting_llm_def_prompt": "Default Prompt Template:",
        "setting_llm_cust_prompt": "Custom Prompt Template:",
        "setting_llm_def_tags": "Default Custom Tags (Comma or Newline):",
        "setting_llm_def_tags": "Default Custom Tags (Comma or Newline):",
        "setting_llm_max_dim": "LLM Max Image Dimension:",
        "setting_llm_skip_nsfw": "Batch LLM: Skip if tag contains rating:explicit/questionable",
        "setting_tagger_gen_thresh": "General Threshold:",
        "setting_tagger_char_thresh": "Character Threshold:",
        "setting_tagger_gen_mcut": "General MCut Enabled",
        "setting_tagger_char_mcut": "Character MCut Enabled",
        "setting_tagger_drop_overlap": "Drop Overlap",
        "setting_mask_ocr_hint": "OCR relies on imgutils.ocr.detect_text_with_ocr; skips if not installed.",
        "setting_ocr_heat": "Heat Threshold:",
        "setting_ocr_box": "Box Threshold:",
        "setting_ocr_unclip": "Unclip Ratio:",
        "setting_ocr_heat_tip": "Lower: detects faint text (more noise); Higher: strict check.",
        "setting_ocr_box_tip": "Confidence threshold. Lower if text is missed.",
        "setting_ocr_unclip_tip": "Expansion ratio. Increase if edges are cut; decrease if merging lines.",
    }
}

# --------------------------
# Themes (CSS)
# --------------------------
THEME_STYLES = {
    "light": "",  # Use system default
    "dark": """
        QMainWindow, QDialog {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QPlainTextEdit, QLineEdit, QTextEdit {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #3e3e42;
        }
        QPushButton {
            background-color: #333333;
            color: #d4d4d4;
            border: 1px solid #444444;
            padding: 5px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #444444;
            border: 1px solid #666666;
        }
        QPushButton:pressed {
            background-color: #222222;
        }
        QTabWidget::pane {
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #969696;
            padding: 8px 15px;
            border: 1px solid #3e3e42;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QScrollArea {
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        QLabel {
            color: #d4d4d4;
        }
        QGroupBox {
            border: 1px solid #3e3e42;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QSplitter::handle {
            background-color: #3e3e42;
        }
    """
}

# --------------------------
# App Settings (persisted)
# --------------------------
APP_SETTINGS_FILE = os.path.join(str(Path.home()), ".ai_captioning_settings.json")

DEFAULT_APP_SETTINGS = {
    # LLM
    "llm_base_url": "https://openrouter.ai/api/v1",
    "llm_api_key": os.getenv("OPENROUTER_API_KEY", "<OPENROUTER_API_KEY>"),
    "llm_model": "mistralai/mistral-large-2512",
    "llm_system_prompt": DEFAULT_SYSTEM_PROMPT,
    "llm_user_prompt_template": DEFAULT_USER_PROMPT_TEMPLATE,
    "llm_custom_prompt_template": DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    "llm_custom_prompt_template": DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    "default_custom_tags": list(DEFAULT_CUSTOM_TAGS),
    "llm_skip_nsfw_on_batch": False,
    "llm_use_gray_mask": True,
    "last_open_dir": "",

    # Tagger (WD14)
    "tagger_model": "EVA02_Large",
    "general_threshold": 0.2,
    "general_mcut_enabled": False,
    "character_threshold": 0.85,
    "character_mcut_enabled": True,
    "drop_overlap": True,

    # Text / normalization
    "english_force_lowercase": True,
    "text_auto_remove_empty_lines": True,  # 自動移除空行
    "text_auto_format": True,              # 插入時自動格式化
    "text_auto_save": True,                # 改動時自動儲存
    "batch_to_txt_mode": "append",         # append | overwrite
    "batch_to_txt_folder_trigger": False,  # 是否將資料夾名作為觸發詞加到句首

    # LLM Resolution (Advanced)
    "llm_max_image_dimension": 1024,

    # Character Tags Filter (simple word matching)
    # 黑名單：包含這些 word 的 tag/句子會被標記
    "char_tag_blacklist_words": ["hair", "eyes", "skin", "bun", "bangs", "sidelocks", "twintails", "braid", "ponytail", "beard", "mustache", "ear", "horn", "tail", "wing", "breast", "mole", "halo", "glasses", "fang", "heterochromia", "headband", "freckles", "lip", "eyebrows", "eyelashes"],
    # 白名單：若包含這些 word，即使符合黑名單也不標記
    "char_tag_whitelist_words": ["holding", "hand", "sitting", "covering", "playing", "background", "looking"],

    # Mask / batch mask text
    "mask_default_alpha": 64, # 1-254
    "mask_default_format": "webp",  # webp | png
    "mask_batch_only_if_has_background_tag": True,
    "mask_batch_detect_text_enabled": True,  # if off, never call detect_text_with_ocr
    "mask_delete_npz_on_move": True,         # 移動舊圖時刪除對應 npz
    
    "mask_padding": 1,        # Mask 內縮像素 (0=不內縮)
    "mask_blur_radius": 3,    # Mask 高斯模糊半徑 (0=不模糊)
    
    # Batch Mask Logic
    "mask_batch_min_foreground_ratio": 0.1,  # 最低主體佔比
    "mask_batch_max_foreground_ratio": 0.8,  # 最高主體佔比
    "mask_batch_skip_if_scenery_tag": True,  # 若包含 indoors/outdoors 則跳過

    # Advanced OCR Settings
    "mask_ocr_heat_threshold": 0.2,
    "mask_ocr_box_threshold": 0.6,
    "mask_ocr_unclip_ratio": 2.3,

    # UI / Theme
    "ui_language": "zh_tw",   # zh_tw | en
    "ui_theme": "light",      # light | dark
}

def load_app_settings() -> dict:
    cfg = dict(DEFAULT_APP_SETTINGS)
    try:
        if os.path.exists(APP_SETTINGS_FILE):
            with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if isinstance(data, dict):
                cfg.update(data)
    except Exception as e:
        print(f"[Settings] load failed: {e}")
    return cfg

def save_app_settings(cfg: dict) -> bool:
    try:
        safe = dict(DEFAULT_APP_SETTINGS)
        safe.update(cfg or {})
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Settings] save failed: {e}")
        return False

def _coerce_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return default

def _coerce_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def _coerce_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)

def call_wd14(img_pil, cfg: dict):
    """Call imgutils.tagging.get_wd14_tags with only supported kwargs."""
    kwargs = {
        "model_name": cfg.get("tagger_model", DEFAULT_APP_SETTINGS["tagger_model"]),
        "general_threshold": _coerce_float(cfg.get("general_threshold", 0.2), 0.2),
        "general_mcut_enabled": _coerce_bool(cfg.get("general_mcut_enabled", False), False),
        "character_threshold": _coerce_float(cfg.get("character_threshold", 0.85), 0.85),
        "character_mcut_enabled": _coerce_bool(cfg.get("character_mcut_enabled", True), True),
        "drop_overlap": _coerce_bool(cfg.get("drop_overlap", True), True),
    }
    try:
        sig = inspect.signature(get_wd14_tags)
        allowed = set(sig.parameters.keys())
        use = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        use = {
            "model_name": kwargs["model_name"],
            "general_threshold": kwargs["general_threshold"],
            "character_mcut_enabled": kwargs["character_mcut_enabled"],
            "drop_overlap": kwargs["drop_overlap"],
        }
    rating, features, chars = get_wd14_tags(img_pil, **use)
    
    # Normalize tags using remove_underline
    features = {remove_underline(k): v for k, v in features.items()}
    chars = {remove_underline(k): v for k, v in chars.items()}
    
    return rating, features, chars


def unload_all_models():
    """ 
    強制執行垃圾回收，並在支援 Torch 的環境下排空 CUDA 快取。
    這有助於在完成 WD14、OCR 或去背景任務後釋放記憶體。
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


NL_PAGE_DELIM = "\n\n=====NL_PAGE=====\n\n"

# ==========================================
#  Utils / Parsing
# ==========================================

def create_checkerboard_png_bytes():
    """
    生成 16x16 棋盤格 PNG bytes（不走 data URI，避免 Qt stylesheet pixmap 警告）
    """
    try:
        w, h = 16, 16
        img = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        pixels = img.load()
        color = (220, 220, 220, 255)

        for y in range(h):
            for x in range(w):
                if (x < 8 and y < 8) or (x >= 8 and y >= 8):
                    pixels[x, y] = color

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()
    except Exception as e:
        print(f"Error creating checkerboard: {e}")
        # 1x1 透明 PNG
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )


def delete_matching_npz(image_path: str) -> int:
    """
    刪除與圖檔名匹配的 npz 檔案。
    例如圖檔 '1b7f4f85fac7f8f7076fa528e95176fb.webp' 
    會匹配 '1b7f4f85fac7f8f7076fa528e95176fb_0849x0849_sdxl.npz'
    回傳刪除的檔案數量。
    """
    if not image_path:
        return 0
    
    try:
        src_dir = os.path.dirname(image_path)
        # 取得不含副檔名的完整檔名 (例如 1b7f4f85fac7f8f7076fa528e95176fb)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        deleted = 0
        for f in os.listdir(src_dir):
            if f.endswith(".npz") and f.startswith(base_name):
                npz_path = os.path.join(src_dir, f)
                try:
                    os.remove(npz_path)
                    deleted += 1
                    print(f"[NPZ] 已刪除: {f}")
                except Exception as e:
                    print(f"[NPZ] 刪除失敗 {f}: {e}")
        return deleted
    except Exception as e:
        print(f"[NPZ] delete_matching_npz 錯誤: {e}")
        return 0


def image_sidecar_json_path(image_path: str) -> str:
    """取得圖片對應的 sidecar JSON 路徑"""
    return os.path.splitext(image_path)[0] + ".json"


def load_image_sidecar(image_path: str) -> dict:
    """
    載入圖片對應的 sidecar JSON。
    結構: {
        "tagger_tags": "...",
        "nl_pages": [...],
        "masked_background": bool,
        "masked_text": bool
    }
    """
    p = image_sidecar_json_path(image_path)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[Sidecar] 載入失敗 {p}: {e}")
    return {}


def save_image_sidecar(image_path: str, data: dict):
    """儲存圖片對應的 sidecar JSON"""
    p = image_sidecar_json_path(image_path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Sidecar] 儲存失敗 {p}: {e}")

def backup_original_image(image_path: str) -> bool:
    """
    修改圖片前先備份原圖 (若為首次修改)。
    備份到同層級的 raw_image 資料夾。
    並在 sidecar 記錄 'raw_image_rel_path'。
    """
    try:
        sidecar = load_image_sidecar(image_path)
        
        # 若已經有備份紀錄，先檢查檔案是否存在
        if "raw_image_rel_path" in sidecar:
            rel_path = sidecar["raw_image_rel_path"]
            src_dir = os.path.dirname(image_path)
            abs_raw_path = os.path.normpath(os.path.join(src_dir, rel_path))
            if os.path.exists(abs_raw_path):
                # 已經有備份且檔案存在，不需再備份
                return True
            else:
                # 紀錄還在但檔案不見了，重新備份當前檔案視為原圖?
                # 照需求: "放回原圖去raw_image找原圖 如果已經放回 回覆處理前狀態"
                # 若檔案不見了可能被手動刪除，這裡視為需要重新備份
                pass

        # 執行備份
        src_dir = os.path.dirname(image_path)
        raw_dir = os.path.join(src_dir, "raw_image")
        os.makedirs(raw_dir, exist_ok=True)
        
        # 原檔名
        fname = os.path.basename(image_path)
        dest_path = os.path.join(raw_dir, fname)
        
        # 如果 raw_image 也可以有同名衝突? 需求說 "用原檔名 複製一份"
        # 假設 raw_image 裡就是最原始的
        if not os.path.exists(dest_path):
            shutil.copy2(image_path, dest_path)
        
        # 計算相對路徑存入 JSON
        rel_path = os.path.relpath(dest_path, src_dir)
        sidecar["raw_image_rel_path"] = rel_path
        
        # 標記 masked 狀態通常由 worker 更新，但這裡是備份邏輯
        save_image_sidecar(image_path, sidecar)
        return True
    except Exception as e:
        print(f"[Backup] 備份失敗 {image_path}: {e}")
        return False

def restore_original_image(image_path: str) -> bool:
    """
    嘗試從 sidecar 記錄的 raw_image 還原圖片。
    """
    try:
        sidecar = load_image_sidecar(image_path)
        if "raw_image_rel_path" not in sidecar:
            return False
            
        rel_path = sidecar["raw_image_rel_path"]
        src_dir = os.path.dirname(image_path)
        abs_raw_path = os.path.normpath(os.path.join(src_dir, rel_path))
        
        if os.path.exists(abs_raw_path):
            # "如果已經放回 回覆處理前狀態"
            # 覆蓋當前圖片
            shutil.copy2(abs_raw_path, image_path)
            # 是否要刪除 mask 標記? "如果已經放回 (設計) 回覆處理前狀態"
            # 通常這意味著不再是 masked 狀態
            if "masked_text" in sidecar: del sidecar["masked_text"]
            if "masked_background" in sidecar: del sidecar["masked_background"]
            save_image_sidecar(image_path, sidecar)
            return True
        return False
    except Exception as e:
        print(f"[Restore] 還原失敗 {image_path}: {e}")
        return False


# ==========================================
#  Raw Image Backup / Restore (原圖備份還原)
# ==========================================

def get_raw_image_dir(image_path: str) -> str:
    """取得 raw_image 備份資料夾路徑"""
    return os.path.join(os.path.dirname(image_path), "raw_image")


def has_raw_backup(image_path: str) -> bool:
    """
    檢查圖片是否已有原圖備份。
    檢查 sidecar JSON 中的 raw_backup_path 欄位。
    """
    sidecar = load_image_sidecar(image_path)
    raw_rel = sidecar.get("raw_backup_path", "")
    if not raw_rel:
        return False
    
    # 驗證備份檔案是否存在
    src_dir = os.path.dirname(image_path)
    raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
    return os.path.exists(raw_abs)


def backup_raw_image(image_path: str) -> bool:
    """
    備份原圖到 raw_image 資料夾。
    - 如果已有備份，不重複備份
    - 備份後在 sidecar JSON 中記錄相對路徑
    - 回傳 True 表示有執行備份，False 表示已存在備份
    """
    if not image_path or not os.path.exists(image_path):
        return False
    
    # 檢查是否已有備份
    if has_raw_backup(image_path):
        return False
    
    try:
        src_dir = os.path.dirname(image_path)
        raw_dir = get_raw_image_dir(image_path)
        os.makedirs(raw_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        dest_path = os.path.join(raw_dir, filename)
        
        # 避免檔名衝突
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            for i in range(1, 9999):
                dest_path = os.path.join(raw_dir, f"{base}_{i}{ext}")
                if not os.path.exists(dest_path):
                    break
        
        # 複製原檔 (不是移動，因為之後還要在原位置處理)
        shutil.copy2(image_path, dest_path)
        
        # 計算相對路徑並儲存到 sidecar
        rel_path = os.path.relpath(dest_path, src_dir)
        sidecar = load_image_sidecar(image_path)
        sidecar["raw_backup_path"] = rel_path
        save_image_sidecar(image_path, sidecar)
        
        print(f"[Backup] 已備份原圖: {filename} -> {rel_path}")
        return True
        
    except Exception as e:
        print(f"[Backup] 備份失敗 {image_path}: {e}")
        return False


def restore_raw_image(image_path: str) -> bool:
    """
    從 raw_image 還原原圖。
    - 如果沒有備份紀錄，回傳 False
    - 還原後清除 sidecar 中的 mask 標記
    - 回傳 True 表示還原成功
    """
    if not image_path:
        return False
    
    sidecar = load_image_sidecar(image_path)
    raw_rel = sidecar.get("raw_backup_path", "")
    
    if not raw_rel:
        print(f"[Restore] 找不到備份紀錄: {image_path}")
        return False
    
    src_dir = os.path.dirname(image_path)
    raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
    
    if not os.path.exists(raw_abs):
        print(f"[Restore] 備份檔案不存在: {raw_abs}")
        return False
    
    try:
        # 複製備份回原位置 (覆蓋目前的處理版本)
        shutil.copy2(raw_abs, image_path)
        
        # 清除 sidecar 中的 mask 標記，但保留備份路徑
        sidecar["masked_background"] = False
        sidecar["masked_text"] = False
        save_image_sidecar(image_path, sidecar)
        
        print(f"[Restore] 已還原: {os.path.basename(image_path)}")
        return True
        
    except Exception as e:
        print(f"[Restore] 還原失敗 {image_path}: {e}")
        return False


def delete_raw_backup(image_path: str) -> bool:
    """
    刪除原圖備份（當使用者確認不需要還原時）。
    - 刪除 raw_image 中的備份檔案
    - 清除 sidecar 中的備份路徑
    """
    if not image_path:
        return False
    
    sidecar = load_image_sidecar(image_path)
    raw_rel = sidecar.get("raw_backup_path", "")
    
    if not raw_rel:
        return False
    
    src_dir = os.path.dirname(image_path)
    raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
    
    try:
        if os.path.exists(raw_abs):
            os.remove(raw_abs)
            print(f"[Backup] 已刪除備份: {raw_rel}")
        
        # 清除 sidecar 中的備份路徑
        if "raw_backup_path" in sidecar:
            del sidecar["raw_backup_path"]
        save_image_sidecar(image_path, sidecar)
        
        return True
        
    except Exception as e:
        print(f"[Backup] 刪除備份失敗 {image_path}: {e}")
        return False


def ensure_tags_csv(csv_path=TAGS_CSV_LOCAL):
    if os.path.exists(csv_path):
        return True
    try:
        req = Request(TAGS_CSV_URL_RAW, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
        with open(csv_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"[Tags.csv] 下載失敗: {e}")
        return False


def load_translations(csv_path=TAGS_CSV_LOCAL):
    translations = {}
    if not os.path.exists(csv_path):
        ensure_tags_csv(csv_path)

    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        # 使用 remove_underline 統一格式
                        key = remove_underline(row[0].strip())
                        translations[key] = row[1].strip()
        except Exception as e:
            print(f"Error loading translations: {e}")
    return translations


def parse_boorutag_meta(meta_path):
    tags_meta = []
    hint_info = []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            lines += [''] * (20 - len(lines))

            if len(lines) >= 19 and lines[18]:
                tags_meta = [t.strip() for t in lines[18].split(',') if t.strip()]

            if len(lines) >= 7 and lines[6] != "by artstyle" and lines[6]:
                artist_val = lines[6].replace('by ', '').replace(' artstyle', '')
                hint_info.append(f"the artist of this image: {{{artist_val}}}")

            if len(lines) >= 10 and lines[9]:
                sources = [s.strip() for s in lines[9].split(',') if s.strip()]
                if len(sources) >= 3:
                    hint_info.append("the copyright of this image: {{crossover}}")
                else:
                    hint_info.append("the copyright of this image: {{" + ', '.join(sources) + '}}')

            if len(lines) >= 13 and lines[12]:
                characters = [c.strip() for c in lines[12].split(',') if c.strip()]
                if characters and len(characters) < 4:
                    hint_info.append("the characters of this image: {{" + ', '.join(characters) + '}}')

    except Exception as e:
        print(f"[boorutag] 解析出錯 {meta_path}: {e}")
    return tags_meta, hint_info


# ==========================================
#  Danbooru-style Query Filter
# ==========================================
import fnmatch

class DanbooruQueryFilter:
    """
    Danbooru-style query parser and matcher.
    Supports: AND (space), OR, NOT (-), grouping (()), wildcards (*), rating shortcuts, order.
    """

    def __init__(self, query: str):
        self.query = query.strip()
        self.order_mode = None  # 'landscape' or 'portrait'
        self._parse_order()

    def _parse_order(self):
        """Extract order: directive from query."""
        import re
        match = re.search(r'\border:(landscape|portrait)\b', self.query, re.IGNORECASE)
        if match:
            self.order_mode = match.group(1).lower()
            self.query = re.sub(r'\border:(landscape|portrait)\b', '', self.query, flags=re.IGNORECASE).strip()

    def _normalize(self, text: str) -> str:
        """Normalize text: lowercase, underscores to spaces."""
        return text.lower().replace("_", " ").strip()

    def _expand_rating(self, term: str) -> str:
        """Expand rating shortcuts like rating:e -> rating:explicit."""
        rating_map = {
            "rating:e": "rating:explicit",
            "rating:q": "rating:questionable",
            "rating:s": "rating:sensitive",
            "rating:g": "rating:general",
        }
        lower = term.lower()
        return rating_map.get(lower, term)

    def _term_matches(self, term: str, content: str) -> bool:
        """Check if a single term matches the content."""
        term = self._expand_rating(term)
        term_norm = self._normalize(term)
        content_norm = self._normalize(content)

        # Handle wildcards
        if "*" in term_norm:
            # fnmatch style: * matches any characters
            pattern = term_norm.replace(" ", "*")  # Allow flexible spacing
            # Check each word in content
            words = content_norm.split()
            for word in words:
                if fnmatch.fnmatch(word, pattern):
                    return True
            # Also check whole content
            if fnmatch.fnmatch(content_norm, f"*{pattern}*"):
                return True
            return False

        # Handle rating:q,s format (multiple ratings)
        if term_norm.startswith("rating:") and "," in term_norm:
            ratings = term_norm.replace("rating:", "").split(",")
            for r in ratings:
                r = r.strip()
                expanded = self._expand_rating(f"rating:{r}")
                if self._normalize(expanded) in content_norm:
                    return True
            return False

        # Simple substring match for normalized content
        return term_norm in content_norm

    def _tokenize(self, query: str) -> list:
        """Tokenize the query into terms and operators."""
        import re
        # Tokens: (, ), or, ~term, -term, -(, term
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            if query[i].isspace():
                i += 1
                continue
            
            # Grouping
            if query[i] == '(':
                tokens.append('(')
                i += 1
            elif query[i] == ')':
                tokens.append(')')
                i += 1
            # Negation with group
            elif query[i:i+2] == '-(':
                tokens.append('-')
                tokens.append('(')
                i += 2
            # Negation prefix
            elif query[i] == '-':
                # Find the term after -
                i += 1
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term:
                    tokens.append(('-', term))
            # Legacy OR prefix
            elif query[i] == '~':
                i += 1
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term:
                    tokens.append(('~', term))
            # OR keyword
            elif query[i:i+2].lower() == 'or' and (i+2 >= len(query) or query[i+2].isspace()):
                tokens.append('or')
                i += 2
            # Regular term
            else:
                term_start = i
                while i < len(query) and not query[i].isspace() and query[i] not in '()':
                    i += 1
                term = query[term_start:i]
                if term and term.lower() != 'or':
                    tokens.append(term)
        
        return tokens

    def _evaluate(self, tokens: list, content: str) -> bool:
        """Evaluate tokenized query against content."""
        if not tokens:
            return True

        # Handle legacy OR (~term ~term)
        tilde_terms = [t[1] for t in tokens if isinstance(t, tuple) and t[0] == '~']
        if tilde_terms:
            # Any of the tilde terms must match
            other_tokens = [t for t in tokens if not (isinstance(t, tuple) and t[0] == '~')]
            tilde_result = any(self._term_matches(term, content) for term in tilde_terms)
            if other_tokens:
                return tilde_result and self._evaluate(other_tokens, content)
            return tilde_result

        # Split by OR
        or_groups = []
        current_group = []
        paren_depth = 0
        
        for token in tokens:
            if token == '(':
                paren_depth += 1
                current_group.append(token)
            elif token == ')':
                paren_depth -= 1
                current_group.append(token)
            elif token == 'or' and paren_depth == 0:
                if current_group:
                    or_groups.append(current_group)
                current_group = []
            else:
                current_group.append(token)
        
        if current_group:
            or_groups.append(current_group)

        # If we have OR groups, any must match
        if len(or_groups) > 1:
            return any(self._evaluate_and_group(group, content) for group in or_groups)
        
        return self._evaluate_and_group(tokens, content)

    def _evaluate_and_group(self, tokens: list, content: str) -> bool:
        """Evaluate an AND group (all must match, except negations)."""
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                # Find matching )
                paren_depth = 1
                j = i + 1
                while j < len(tokens) and paren_depth > 0:
                    if tokens[j] == '(':
                        paren_depth += 1
                    elif tokens[j] == ')':
                        paren_depth -= 1
                    j += 1
                sub_tokens = tokens[i+1:j-1]
                if not self._evaluate(sub_tokens, content):
                    return False
                i = j
            elif token == ')':
                i += 1
            elif token == '-':
                # Next token is negated
                i += 1
                if i < len(tokens):
                    next_token = tokens[i]
                    if next_token == '(':
                        # Negated group
                        paren_depth = 1
                        j = i + 1
                        while j < len(tokens) and paren_depth > 0:
                            if tokens[j] == '(':
                                paren_depth += 1
                            elif tokens[j] == ')':
                                paren_depth -= 1
                            j += 1
                        sub_tokens = tokens[i+1:j-1]
                        if self._evaluate(sub_tokens, content):
                            return False
                        i = j
                    else:
                        i += 1
            elif isinstance(token, tuple):
                op, term = token
                if op == '-':
                    if self._term_matches(term, content):
                        return False
                elif op == '~':
                    pass  # Handled above
                i += 1
            elif isinstance(token, str) and token not in ('(', ')', 'or'):
                if not self._term_matches(token, content):
                    return False
                i += 1
            else:
                i += 1
        
        return True

    def matches(self, content: str) -> bool:
        """Check if content matches the query."""
        if not self.query:
            return True
        tokens = self._tokenize(self.query)
        return self._evaluate(tokens, content)

    def sort_images(self, image_paths: list) -> list:
        """Sort images by order mode (landscape/portrait)."""
        if not self.order_mode:
            return image_paths
        
        def get_aspect(path):
            try:
                img = Image.open(path)
                return img.width / img.height
            except Exception:
                return 1.0
        
        if self.order_mode == 'landscape':
            return sorted(image_paths, key=lambda p: -get_aspect(p))
        elif self.order_mode == 'portrait':
            return sorted(image_paths, key=lambda p: get_aspect(p))
        return image_paths


def extract_bracket_content(text):
    return re.findall(r'\{(.*?)\}', text)


def smart_parse_tags(text):
    """
    Parses text into a list of dictionaries {'text': str, 'trans': str}.
    """
    if not text:
        return []

    clean_text = text.strip()
    if not clean_text:
        return []

    parsed_items = []
    lines = [l.strip() for l in clean_text.split('\n') if l.strip()]

    is_sentence_mode = False
    if len(lines) > 1:
        for line in lines:
            if (line.startswith("(") and line.endswith(")")) or \
               (line.startswith("（") and line.endswith("）")):
                is_sentence_mode = True
                break
    elif "." in clean_text and "," not in clean_text and len(clean_text) > 50:
        is_sentence_mode = True
        lines = [l.strip() for l in clean_text.replace(". ", ".\n").split('\n') if l.strip()]

    if is_sentence_mode:
        i = 0
        while i < len(lines):
            current_line = lines[i]
            if (current_line.startswith("(") and current_line.endswith(")")) or \
               (current_line.startswith("（") and current_line.endswith("）")):
                i += 1
                continue

            trans = None
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if (next_line.startswith("(") and next_line.endswith(")")) or \
                   (next_line.startswith("（") and next_line.endswith("）")):
                    trans = next_line[1:-1].strip()
                    i += 1

            parsed_items.append({'text': current_line, 'trans': trans})
            i += 1

    else:
        segments = clean_text.replace("\n", ",").split(",")
        for s in segments:
            if s.strip():
                parsed_items.append({'text': s.strip(), 'trans': None})

    return parsed_items


def is_basic_character_tag(text: str, cfg: dict) -> bool:
    """
    判定一段文字（tag 或句子）是否為特徵內容。
    邏輯：任何黑名單 word 包含在文字中，且沒有任何白名單 word 包含在文字中。
    """
    if not text:
        return False
    
    # 正規化：小寫，保持空格
    t = text.strip().lower()
    
    # 取得黑白名單 words（以逗號分隔）
    bl_words = [w.strip().lower() for w in cfg.get("char_tag_blacklist_words", []) if w.strip()]
    wl_words = [w.strip().lower() for w in cfg.get("char_tag_whitelist_words", []) if w.strip()]
    
    # 如果沒有黑名單，直接回傳 False
    if not bl_words:
        return False
    
    # 檢查是否包含任何黑名單 word
    has_blacklist = any(bw in t for bw in bl_words)
    if not has_blacklist:
        return False
    
    # 檢查是否包含任何白名單 word（若有則不算符合）
    has_whitelist = any(ww in t for ww in wl_words)
    if has_whitelist:
        return False
    
    return True


def normalize_for_match(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace(", ", "").replace(",", "")
    t = t.strip()
    t = t.rstrip(".")
    return t.strip()


def cleanup_csv_like_text(text: str, force_lower: bool = False) -> str:
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    result = ", ".join(parts)
    if force_lower:
        result = result.lower()
    return result


def split_csv_like_text(text: str):
    return [p.strip() for p in text.split(",") if p.strip()]


def try_tags_to_text_list(tags_list):
    """
    先 tags_to_text，再拆回 list；若失敗就 fallback
    """
    try:
        s = tags_to_text(tags_list)
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts
    except Exception:
        return [t.strip() for t in tags_list if str(t).strip()]

# ==========================================
#  Workers
# ==========================================

class TaggerWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image_path, cfg: dict):
        super().__init__()
        self.image_path = image_path
        self.cfg = dict(cfg or {})

    def run(self):
        try:
            img = Image.open(self.image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            rating, features, chars = call_wd14(img, self.cfg)
            rating_tag = f"rating:{max(rating, key=rating.get)}"
            tags_list = [rating_tag] + list(chars.keys()) + list(features.keys())
            tags_str = ", ".join(tags_list)
            self.finished.emit(tags_str)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            unload_all_models()


class LLMWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model_name, system_prompt, user_prompt, image_path, tags_context, max_dim=1024, settings=None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_path = image_path
        self.tags_context = tags_context
        self.max_dim = max_dim
        self.settings = settings or {}

    def run(self):
        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            # --- 指定使用原圖或灰底 ---
            use_gray = self.settings.get("llm_use_gray_mask", True)
            img_path_to_open = self.image_path
            
            if not use_gray:
                # 嘗試尋找 unmask 裡的原檔
                src_dir = os.path.dirname(self.image_path)
                stem = os.path.splitext(os.path.basename(self.image_path))[0]
                unmask_dir = os.path.join(src_dir, "unmask")
                if os.path.exists(unmask_dir):
                    for f in os.listdir(unmask_dir):
                        if os.path.splitext(f)[0] == stem:
                            img_path_to_open = os.path.join(unmask_dir, f)
                            break
            
            img = Image.open(img_path_to_open)
            
            # 讀取 Sidecar 判斷是否曾被處理過 (去背景/去文字)
            sidecar = load_image_sidecar(self.image_path)
            is_masked_in_app = sidecar.get("masked_text", False) or sidecar.get("masked_background", False)
            
            has_alpha = False
            if img.mode == 'RGBA':
                has_alpha = True
                if use_gray:
                    # 強制變全灰 (無論 alpha 多少，只要有透明度就變灰)
                    canvas = Image.new("RGB", img.size, (136, 136, 136))
                    alpha = img.getchannel('A')
                    # Binary: alpha < 255 -> 0 (use gray), alpha == 255 -> 255 (use pixel)
                    mask = alpha.point(lambda p: 255 if p == 255 else 0)
                    canvas.paste(img, mask=mask)
                    img = canvas
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 移除不再需要的 legacy 變數
            should_warn = False 

            # Use self.max_dim for resizing
            target_size = self.max_dim
            ratio = min(target_size / img.width, target_size / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{img_str}"

            # 決定提示詞前綴
            prompt_prefix = ""
            if use_gray:
                if has_alpha or is_masked_in_app:
                    prompt_prefix = "這是一張經過去背處理的圖像，背景已填滿灰色，請忽視灰色區域並針對主體進行描述。\n"
            
            final_user_content = prompt_prefix + self.user_prompt.replace("{LLM處理結果}", self.tags_context)
            final_user_content = final_user_content.replace("{tags}", self.tags_context)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_user_content},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]
                }
            ]

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
            )

            self.finished.emit(response.choices[0].message.content)

        except Exception as e:
            self.error.emit(str(e))


class BatchTaggerWorker(QThread):
    progress = pyqtSignal(int, int, str)  # i, total, filename
    per_image = pyqtSignal(str, str)      # image_path, tags_str
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_paths, cfg: dict):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self.image_paths)
            for i, p in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                self.progress.emit(i, total, os.path.basename(p))
                try:
                    img = Image.open(p)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    rating, features, chars = call_wd14(img, self.cfg)
                    rating_tag = f"rating:{max(rating, key=rating.get)}"
                    tags_list = [rating_tag] + list(chars.keys()) + list(features.keys())
                    tags_str = ", ".join(tags_list)
                    self.per_image.emit(p, tags_str)
                except Exception as e:
                    print(f"[BatchTagger] {p} 失敗: {e}")
            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            unload_all_models()


class BatchLLMWorker(QThread):
    progress = pyqtSignal(int, int, str)
    per_image = pyqtSignal(str, str)  # image_path, nl_content
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model_name, system_prompt, user_prompt, image_paths, tags_context_getter, max_dim=1024, skip_nsfw=False, settings=None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_paths = list(image_paths)
        self.tags_context_getter = tags_context_getter
        self.max_dim = max_dim
        self.skip_nsfw = skip_nsfw
        self.settings = settings or {}
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            total = len(self.image_paths)
            for i, p in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                self.progress.emit(i, total, os.path.basename(p))

                try:
                    tags_context = self.tags_context_getter(p)

                    # Check NSFW skip
                    if self.skip_nsfw:
                        t_lower = tags_context.lower()
                        if "explicit" in t_lower or "questionable" in t_lower:
                            # Skip this image
                            self.per_image.emit(p, "")
                            continue

                    # --- 指定使用原圖或灰底 ---
                    use_gray = self.settings.get("llm_use_gray_mask", True)
                    img_path_to_open = p
                    
                    if not use_gray:
                        # 嘗試尋找 unmask 裡的原檔
                        src_dir = os.path.dirname(p)
                        stem = os.path.splitext(os.path.basename(p))[0]
                        unmask_dir = os.path.join(src_dir, "unmask")
                        if os.path.exists(unmask_dir):
                            for f in os.listdir(unmask_dir):
                                if os.path.splitext(f)[0] == stem:
                                    img_path_to_open = os.path.join(unmask_dir, f)
                                    break
                    
                    img = Image.open(img_path_to_open)
                    
                    # 讀取 Sidecar 判斷是否曾被處理過 (去背景/去文字)
                    sidecar = load_image_sidecar(p)
                    is_masked_in_app = sidecar.get("masked_text", False) or sidecar.get("masked_background", False)
                    
                    has_alpha = False
                    if img.mode == 'RGBA':
                        has_alpha = True
                        if use_gray:
                            # 強制變全灰
                            canvas = Image.new("RGB", img.size, (136, 136, 136))
                            alpha = img.getchannel('A')
                            # Binary: alpha < 255 -> 0, alpha == 255 -> 255
                            mask = alpha.point(lambda p: 255 if p == 255 else 0)
                            canvas.paste(img, mask=mask)
                            img = canvas
                        else:
                            img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 移除不再需要的 legacy 變數
                    should_warn = False
                    
                    target_size = self.max_dim
                    ratio = min(target_size / img.width, target_size / img.height)
                    if ratio < 1:
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=90)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_url = f"data:image/jpeg;base64,{img_str}"

                    # 決定提示詞前綴
                    prompt_prefix = ""
                    if use_gray:
                        if has_alpha or is_masked_in_app:
                            prompt_prefix = "這是一張經過去背處理的圖像，背景已填滿灰色，請忽視灰色區域並針對主體進行描述。\n"
                    
                    final_user_content = prompt_prefix + self.user_prompt.replace("{LLM處理結果}", tags_context)
                    final_user_content = final_user_content.replace("{tags}", tags_context)

                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": final_user_content},
                                {"type": "image_url", "image_url": {"url": img_url}}
                            ]
                        }
                    ]

                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False
                    )

                    full_text = response.choices[0].message.content
                    content = MainWindow.extract_llm_content_and_postprocess(full_text, self.settings.get('english_force_lowercase', True))
                    self.per_image.emit(p, content)

                except Exception as e:
                    print(f"[BatchLLM] {p} 失敗: {e}")

            self.done.emit()

        except Exception:
            self.error.emit(traceback.format_exc())


# ==========================================
#  Extra Tools: Background Remover & Stroke Eraser
# ==========================================


class BatchMaskTextWorker(QThread):
    progress = pyqtSignal(int, int, str)   # i, total, filename
    per_image = pyqtSignal(str, str)       # old_path, new_path
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_paths, cfg: dict, background_tag_checker=None, is_batch=True):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self.background_tag_checker = background_tag_checker
        self.is_batch = is_batch
        self._stop = False

    def stop(self):
        self._stop = True

    def _should_process(self, image_path: str) -> bool:
        # Batch 時檢查是否已處理過去文字
        if self.is_batch:
            sidecar = load_image_sidecar(image_path)
            if sidecar.get("masked_text", False):
                return False

        only_bg = bool(self.cfg.get("mask_batch_only_if_has_background_tag", False))
        if not only_bg:
            return True
        if self.background_tag_checker is None:
            return True
        try:
            return bool(self.background_tag_checker(image_path))
        except Exception:
            return True

    def _detect_text_boxes(self, image_path: str):
        if detect_text_with_ocr is None:
            return []
        if not bool(self.cfg.get("mask_batch_detect_text_enabled", True)):
            return []
        try:
            heat = float(self.cfg.get("mask_ocr_heat_threshold", 0.2))
            box = float(self.cfg.get("mask_ocr_box_threshold", 0.6))
            unclip = float(self.cfg.get("mask_ocr_unclip_ratio", 2.3))
            
            results = detect_text_with_ocr(
                image_path,
                max_candidates=100,
                heat_threshold=heat,
                box_threshold=box,
                unclip_ratio=unclip
            )
            boxes = []
            for item in results or []:
                if not item:
                    continue
                box = item[0]
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    boxes.append(tuple(box))
            return boxes
        except Exception as e:
            print(f"[OCR] {image_path} failed: {e}")
            return []

    def run(self):
        try:
            import numpy as np
            total = len(self.image_paths)
            alpha_val = int(self.cfg.get("mask_default_alpha", 0))
            alpha_val = max(0, min(255, alpha_val))
            fmt = str(self.cfg.get("mask_default_format", "webp")).lower().strip(".")
            if fmt not in ("webp", "png"):
                fmt = "webp"
            
            for i, pth in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                if self._stop:
                    break
                self.progress.emit(i, total, os.path.basename(pth))

                # Batch Mode Check: Skip if already masked
                if self.is_batch:
                    sidecar = load_image_sidecar(pth)
                    if sidecar.get("masked_text", False):
                        continue
                
                if not self._should_process(pth):
                    continue

                boxes = self._detect_text_boxes(pth)
                if not boxes:
                    continue
                
                # Backup Original FIRST
                backup_original_image(pth)

                # ========== 新備份機制 ==========
                backup_raw_image(pth)

                base_no_ext = os.path.splitext(pth)[0]
                ext = os.path.splitext(pth)[1].lower()
                out_path = base_no_ext + f".{fmt}"

                with Image.open(pth) as img:
                    img_rgba = img.convert("RGBA")
                    a = np.array(img_rgba.getchannel("A"), dtype=np.uint8)
                    for (x1, y1, x2, y2) in boxes:
                        x1 = max(0, int(x1)); y1 = max(0, int(y1))
                        x2 = min(a.shape[1], int(x2)); y2 = min(a.shape[0], int(y2))
                        if x2 > x1 and y2 > y1:
                            a[y1:y2, x1:x2] = alpha_val
                    img_rgba.putalpha(Image.fromarray(a, mode="L"))
                    
                    # Save
                    if fmt == "png":
                        img_rgba.save(out_path, "PNG")
                    else:
                        img_rgba.save(out_path, "WEBP")

                # 如果格式不同，刪除原檔 (已備份)
                if ext != f".{fmt}" and os.path.abspath(out_path) != os.path.abspath(pth):
                    try:
                        os.remove(pth)
                        # 處理 sidecar JSON
                        old_json = image_sidecar_json_path(pth)
                        new_json = image_sidecar_json_path(out_path)
                        if os.path.exists(old_json) and old_json != new_json:
                            shutil.move(old_json, new_json)
                        # 刪除對應 npz
                        if self.cfg.get("mask_delete_npz_on_move", True):
                            delete_matching_npz(pth)
                    except Exception:
                        pass

                # 記錄 masked_text 到 JSON sidecar
                sidecar = load_image_sidecar(out_path)
                sidecar["masked_text"] = True
                save_image_sidecar(out_path, sidecar)

                self.per_image.emit(pth, out_path)

            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            unload_all_models()


class BatchUnmaskWorker(QThread):
    progress = pyqtSignal(int, int, str)   # i, total, filename
    per_image = pyqtSignal(str, str)       # old_path, new_path
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_paths, cfg: dict = None, background_tag_checker=None, is_batch=True):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self.background_tag_checker = background_tag_checker
        self.is_batch = is_batch
        self._stop = False
        self.is_batch = is_batch # Added for batch vs single image processing distinction

    def stop(self):
        self._stop = True

    @staticmethod
    def _unique_path(path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        for i in range(1, 9999):
            p2 = f"{base}_{i}{ext}"
            if not os.path.exists(p2):
                return p2
        return path

    @staticmethod
    def remove_background_to_webp(image_path: str, remover, cfg: dict = None, is_batch=False) -> str:
        """
        去背處理主邏輯。
        回傳 (new_path, old_path) 或 (None, None) 表示跳過。
        """
        if not image_path:
            return None, None

        cfg = cfg or {}
        
        # Settings
        alpha_threshold = int(cfg.get("mask_default_alpha", 0))
        padding = int(cfg.get("mask_padding", 3))
        blur_radius = int(cfg.get("mask_blur_radius", 10))

        # Batch Only: Check Scenery Tags
        if is_batch and cfg.get("mask_batch_skip_if_scenery_tag", True):
            sidecar = load_image_sidecar(image_path)
            tags = sidecar.get("tagger_tags", "")
            t_lower = tags.lower()
            if "indoors" in t_lower or "outdoors" in t_lower:
                return None, None

        # ========== 新備份機制 ==========
        # 第一次處理時備份原圖到 raw_image
        backup_raw_image(image_path)
        
        ext = os.path.splitext(image_path)[1].lower()
        base_no_ext = os.path.splitext(image_path)[0]

        # 輸出檔案：統一為 WEBP
        target_file = base_no_ext + ".webp"
        
        # 讀取來源 (直接從原位置讀取，因為已備份)
        src_for_processing = image_path

        import numpy as np
        from PIL import ImageFilter

        with Image.open(src_for_processing) as img:
            # (1) 處理輸入 Alpha：alpha=0 的像素設為白色
            img_rgba_input = img.convert('RGBA')
            input_arr = np.array(img_rgba_input)
            
            alpha_channel_input = input_arr[:, :, 3]
            zero_indices_input = alpha_channel_input == 0
            
            input_arr[zero_indices_input, 0] = 255
            input_arr[zero_indices_input, 1] = 255
            input_arr[zero_indices_input, 2] = 255
            
            img_corrected = Image.fromarray(input_arr, 'RGBA')
            
            # (2) 生成遮罩
            img_rm = remover.process(img_corrected.convert('RGB'), type='rgba')
            rm_arr = np.array(img_rm)
            mask_arr_remover = rm_arr[:, :, 3]

            # 結合原始 Alpha
            combined_alpha = np.minimum(mask_arr_remover, alpha_channel_input)

            # (3) Batch Only: 主體佔比檢查
            if is_batch:
                min_r = float(cfg.get("mask_batch_min_foreground_ratio", 0.1))
                max_r = float(cfg.get("mask_batch_max_foreground_ratio", 0.8))
                
                fg_count = np.sum(combined_alpha == 255)
                ratio = fg_count / combined_alpha.size
                
                if ratio < min_r or ratio > max_r:
                    return None, None

            # (4) Padding -> Blur -> Clamp
            mask_img = Image.fromarray(combined_alpha)
            
            if padding > 0:
                mask_img = mask_img.filter(ImageFilter.MinFilter(padding * 2 + 1))
            
            if blur_radius > 0:
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
            
            processed_alpha_float = np.array(mask_img).astype(np.float32)
            
            if alpha_threshold > 0:
                processed_alpha_float = np.maximum(processed_alpha_float, alpha_threshold)
            
            processed_alpha = np.clip(processed_alpha_float, 0, 255).astype(np.uint8)
            
            # 重組最終影像
            final_r = rm_arr[:, :, 0]
            final_g = rm_arr[:, :, 1]
            final_b = rm_arr[:, :, 2]
            
            final_img_arr = np.dstack((final_r, final_g, final_b, processed_alpha))
            final_img = Image.fromarray(final_img_arr, 'RGBA')
            
            # 儲存為 WEBP
            final_img.save(target_file, 'WEBP', quality=100)

        # 如果原檔不是 WEBP，刪除原檔 (已備份)
        if ext != ".webp" and os.path.abspath(target_file) != os.path.abspath(image_path):
            try:
                os.remove(image_path)
                # 處理 sidecar JSON
                old_json = image_sidecar_json_path(image_path)
                new_json = image_sidecar_json_path(target_file)
                if os.path.exists(old_json) and old_json != new_json:
                    shutil.move(old_json, new_json)
            except Exception:
                pass

        # 更新 Sidecar
        sidecar = load_image_sidecar(target_file)
        sidecar["masked_background"] = True
        save_image_sidecar(target_file, sidecar)

        return target_file, image_path

    def run(self):
        try:
            if Remover is None:
                self.error.emit("transparent_background.Remover not available")
                return

            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            remover = Remover(device=device)

            total = len(self.image_paths)
            for i, p in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                self.progress.emit(i, total, os.path.basename(p))
                
                # Batch 時檢查是否已處理過去背
                if self.is_batch:
                    sidecar = load_image_sidecar(p)
                    if sidecar.get("masked_background", False):
                        continue

                # 遵循設定：僅處理包含 background 標籤的圖片
                only_bg = bool(self.cfg.get("mask_batch_only_if_has_background_tag", False))
                if self.is_batch and only_bg and self.background_tag_checker:
                    if not self.background_tag_checker(p):
                        continue

                try:
                    new_path, old_path = BatchUnmaskWorker.remove_background_to_webp(
                        p, 
                        remover, 
                        cfg=self.cfg,
                        is_batch=self.is_batch
                    )
                    if new_path:
                        # 刪除對應 npz
                        if self.cfg.get("mask_delete_npz_on_move", True) and old_path:
                            delete_matching_npz(old_path)
                        self.per_image.emit(p, new_path)
                except Exception as e:
                    print(f"[BatchUnmask] {p} 失敗: {e}")

            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if 'remover' in locals():
                del remover
            unload_all_models()



class BatchRestoreWorker(QThread):
    """批量還原原圖 (從 raw_image 資料夾)"""
    progress = pyqtSignal(int, int, str)
    per_image = pyqtSignal(str, str)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = list(image_paths)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self.image_paths)
            for i, pth in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                
                self.progress.emit(i, total, os.path.basename(pth))

                # 使用新的還原機制
                if has_raw_backup(pth):
                    try:
                        success = restore_raw_image(pth)
                        if success:
                            self.per_image.emit(pth, pth)  # 路徑不變，只是內容還原
                    except Exception as e:
                        print(f"[BatchRestore] Failed {pth}: {e}")

            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


class StrokeCanvas(QLabel):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.base_pixmap = pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mask = QImage(self.base_pixmap.size(), QImage.Format.Format_Grayscale8)
        self.mask.fill(0)

        self.preview = QPixmap(self.base_pixmap.size())
        self.preview.fill(Qt.GlobalColor.transparent)

        self.pen_width = 30
        self.drawing = False
        self.last_pos = None

        self._update_display()

    def set_pen_width(self, w: int):
        self.pen_width = max(1, int(w))

    def clear_mask(self):
        self.mask.fill(0)
        self.preview.fill(Qt.GlobalColor.transparent)
        self._update_display()

    def _draw_line(self, p1, p2):
        # draw to mask
        painter = QPainter(self.mask)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(self.pen_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()

        # draw preview overlay
        painter2 = QPainter(self.preview)
        pen2 = QPen(QColor(255, 0, 0, 160))
        pen2.setWidth(self.pen_width)
        pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen2.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter2.setPen(pen2)
        painter2.drawLine(p1, p2)
        painter2.end()

    def _update_display(self):
        pm = QPixmap(self.base_pixmap)
        painter = QPainter(pm)
        painter.drawPixmap(0, 0, self.preview)
        painter.end()
        self.setPixmap(pm)

    def _to_image_pos(self, widget_pos: QPoint):
        """Map a widget (label) position to image pixel coords, respecting centered pixmap."""
        pm_w = self.base_pixmap.width()
        pm_h = self.base_pixmap.height()
        off_x = int((self.width() - pm_w) / 2)
        off_y = int((self.height() - pm_h) / 2)
        x = int(widget_pos.x() - off_x)
        y = int(widget_pos.y() - off_y)
        if x < 0 or y < 0 or x >= pm_w or y >= pm_h:
            return None
        return QPoint(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            p = self._to_image_pos(event.position().toPoint())
            if p is None:
                event.ignore()
                return
            self.drawing = True
            self.last_pos = p
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            p = self._to_image_pos(event.position().toPoint())
            if p is None:
                event.ignore()
                return
            if self.last_pos is not None:
                self._draw_line(self.last_pos, p)
                self.last_pos = p
                self._update_display()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def get_mask(self) -> QImage:
        return QImage(self.mask)

class StrokeEraseDialog(QDialog):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stroke Eraser")
        self.image_path = image_path
        self._mask = None

        layout = QVBoxLayout(self)

        # load image (fit to a reasonable size)
        pm = QPixmap(image_path)
        if pm.isNull():
            raise RuntimeError("Cannot load image")

        max_w, max_h = 1200, 800
        if pm.width() > max_w or pm.height() > max_h:
            pm = pm.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.canvas = StrokeCanvas(pm)
        layout.addWidget(self.canvas, 1)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("筆畫粗細:"))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(120)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(lambda v: self.canvas.set_pen_width(v))
        ctrl.addWidget(self.slider, 1)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.canvas.clear_mask)
        ctrl.addWidget(self.btn_clear)

        layout.addLayout(ctrl)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_apply = QPushButton("Apply")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

    def get_result(self):
        return self.canvas.get_mask(), int(self.slider.value())

# ==========================================
#  UI Components
# ==========================================

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
                    # \b matches word boundary.
                    # Note: traditional \b might fail on non-ascii if not configured?
                    # Python's re handles unicode word boundaries by default? 
                    # Actually standard \w in python 3 re matches unicode.
                    if re.search(rf"\b{esc}\b", search_text):
                        is_active = True
                except Exception:
                    pass
            
            btn.setChecked(is_active)
            btn.blockSignals(False)


class AdvancedFindReplaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Find & Replace")
        self.setMinimumWidth(400)
        self.layout = QVBoxLayout(self)
        form = QFormLayout()
        self.find_edit = QLineEdit()
        self.replace_edit = QLineEdit()
        form.addRow("Find:", self.find_edit)
        form.addRow("Replace:", self.replace_edit)
        self.layout.addLayout(form)
        self.grp_scope = QGroupBox("Scope")
        scope_layout = QHBoxLayout()
        self.rb_current = QRadioButton("Current Image Only")
        self.rb_all = QRadioButton("All Images")
        self.rb_current.setChecked(True)
        scope_layout.addWidget(self.rb_current)
        scope_layout.addWidget(self.rb_all)
        self.grp_scope.setLayout(scope_layout)
        self.layout.addWidget(self.grp_scope)
        self.grp_mode = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        self.chk_case = QCheckBox("Case Sensitive")
        self.chk_regex = QCheckBox("Regular Expression")
        mode_layout.addWidget(self.chk_case)
        mode_layout.addWidget(self.chk_regex)
        self.grp_mode.setLayout(mode_layout)
        self.layout.addWidget(self.grp_mode)
        btn_layout = QHBoxLayout()
        self.btn_replace = QPushButton("Replace")
        self.btn_cancel = QPushButton("Cancel")
        btn_layout.addWidget(self.btn_replace)
        btn_layout.addWidget(self.btn_cancel)
        self.layout.addLayout(btn_layout)
        self.btn_replace.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def get_settings(self):
        return {
            'find': self.find_edit.text(),
            'replace': self.replace_edit.text(),
            'scope_all': self.rb_all.isChecked(),
            'case_sensitive': self.chk_case.isChecked(),
            'regex': self.chk_regex.isChecked()
        }



class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = dict(cfg or {})
        self.setWindowTitle(self.tr("btn_settings"))
        self.setMinimumWidth(640)

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs, 1)
        
        # 初始化 UI
        self._init_ui()

    def tr(self, key: str) -> str:
        lang = self.cfg.get("ui_language", "zh_tw")
        return LOCALIZATION.get(lang, LOCALIZATION["zh_tw"]).get(key, key)

    def _init_ui(self):
        # ---- UI ----
        tab_ui = QWidget()
        ui_layout = QVBoxLayout(tab_ui)
        ui_form = QFormLayout()
        
        self.cb_lang = QComboBox()
        self.cb_lang.addItem(self.tr("setting_lang_zh"), "zh_tw")
        self.cb_lang.addItem(self.tr("setting_lang_en"), "en")
        idx_lang = self.cb_lang.findData(self.cfg.get("ui_language", "zh_tw"))
        self.cb_lang.setCurrentIndex(idx_lang if idx_lang >= 0 else 0)
        
        ui_form.addRow(self.tr("setting_ui_lang"), self.cb_lang)
        
        self.rb_light = QRadioButton(self.tr("setting_theme_light"))
        self.rb_dark = QRadioButton(self.tr("setting_theme_dark"))
        if self.cfg.get("ui_theme", "light") == "dark":
            self.rb_dark.setChecked(True)
        else:
            self.rb_light.setChecked(True)
        
        ui_theme_lay = QHBoxLayout()
        ui_theme_lay.addWidget(self.rb_light)
        ui_theme_lay.addWidget(self.rb_dark)
        ui_form.addRow(self.tr("setting_ui_theme"), ui_theme_lay)
        
        ui_layout.addLayout(ui_form)
        ui_layout.addStretch(1)
        self.tabs.addTab(tab_ui, self.tr("setting_tab_ui"))

        # ---- LLM ----
        tab_llm = QWidget()
        llm_layout = QVBoxLayout(tab_llm)
        form = QFormLayout()

        self.ed_base_url = QLineEdit(str(self.cfg.get("llm_base_url", "")))
        self.ed_api_key = QLineEdit(str(self.cfg.get("llm_api_key", "")))
        self.ed_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_model = QLineEdit(str(self.cfg.get("llm_model", "")))

        form.addRow("LLM Base URL:", self.ed_base_url)
        form.addRow("API Key:", self.ed_api_key)
        form.addRow("API Key:", self.ed_api_key)
        form.addRow("Model:", self.ed_model)
        
        self.spin_llm_dim = QSpinBox()
        self.spin_llm_dim.setRange(256, 4096)
        self.spin_llm_dim.setSingleStep(128)
        self.spin_llm_dim.setValue(int(self.cfg.get("llm_max_image_dimension", 1024)))
        self.spin_llm_dim.setToolTip("傳給 LLM 的圖片最大邊長。\n調大：細節更多但 API 費用較高、速度較慢\n調小：處理更快且省費用，但可能遺漏細節")
        form.addRow(self.tr("setting_llm_max_dim"), self.spin_llm_dim)
        
        self.chk_llm_skip_nsfw = QCheckBox(self.tr("setting_llm_skip_nsfw"))
        self.chk_llm_skip_nsfw.setChecked(bool(self.cfg.get("llm_skip_nsfw_on_batch", False)))
        self.chk_llm_skip_nsfw.setToolTip("勾選後，批量 LLM 會自動跳過含 explicit/questionable 標籤的圖片")
        form.addRow("", self.chk_llm_skip_nsfw)
        
        self.chk_llm_use_gray_mask = QCheckBox(self.tr("setting_llm_use_gray_mask"))
        self.chk_llm_use_gray_mask.setChecked(bool(self.cfg.get("llm_use_gray_mask", True)))
        self.chk_llm_use_gray_mask.setToolTip("勾選後，去背後的透明區域會填灰色再傳給 LLM，\n讓 AI 專注描述主體而不是背景")
        form.addRow("", self.chk_llm_use_gray_mask)

        llm_layout.addLayout(form)

        llm_layout.addWidget(QLabel(self.tr("setting_llm_sys_prompt")))
        self.ed_system_prompt = QPlainTextEdit()
        self.ed_system_prompt.setPlainText(str(self.cfg.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT)))
        self.ed_system_prompt.setMinimumHeight(90)
        llm_layout.addWidget(self.ed_system_prompt)

        llm_layout.addWidget(QLabel(self.tr("setting_llm_def_prompt")))
        self.ed_user_template = QPlainTextEdit()
        self.ed_user_template.setPlainText(str(self.cfg.get("llm_user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)))
        self.ed_user_template.setMinimumHeight(200)
        llm_layout.addWidget(self.ed_user_template, 1)


        llm_layout.addWidget(QLabel(self.tr("setting_llm_cust_prompt")))
        self.ed_custom_template = QPlainTextEdit()
        self.ed_custom_template.setPlainText(str(self.cfg.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
        self.ed_custom_template.setMinimumHeight(200)
        llm_layout.addWidget(self.ed_custom_template, 1)

        llm_layout.addWidget(QLabel(self.tr("setting_llm_def_tags")))
        self.ed_default_custom_tags = QPlainTextEdit()
        tags = self.cfg.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS))
        if isinstance(tags, list):
            self.ed_default_custom_tags.setPlainText("\n".join([str(t) for t in tags]))
        else:
            self.ed_default_custom_tags.setPlainText(str(tags))
        self.ed_default_custom_tags.setMinimumHeight(80)
        llm_layout.addWidget(self.ed_default_custom_tags)

        self.tabs.addTab(tab_llm, self.tr("setting_tab_llm"))

        # ---- Tagger ----
        tab_tagger = QWidget()
        tagger_layout = QVBoxLayout(tab_tagger)
        form2 = QFormLayout()
        self.ed_tagger_model = QLineEdit(str(self.cfg.get("tagger_model", "EVA02_Large")))
        self.ed_tagger_model.setToolTip("WD14 標籤模型名稱。常用: EVA02_Large (最準)、SwinV2 (較快)")
        
        self.ed_general_threshold = QLineEdit(str(self.cfg.get("general_threshold", 0.2)))
        self.ed_general_threshold.setToolTip("一般標籤的信心閾值 (0.0~1.0)\n調低：標籤更多但可能有誤判\n調高：標籤更精準但可能遺漏")
        
        self.chk_general_mcut = QCheckBox(self.tr("setting_tagger_gen_mcut"))
        self.chk_general_mcut.setChecked(bool(self.cfg.get("general_mcut_enabled", False)))
        self.chk_general_mcut.setToolTip("啟用 MCut 演算法自動決定閾值，會覆蓋上方的手動閾值設定")

        self.ed_character_threshold = QLineEdit(str(self.cfg.get("character_threshold", 0.85)))
        self.ed_character_threshold.setToolTip("角色/特徵標籤的信心閾值 (0.0~1.0)\n建議設較高 (0.8+) 以避免誤判角色")
        
        self.chk_character_mcut = QCheckBox(self.tr("setting_tagger_char_mcut"))
        self.chk_character_mcut.setChecked(bool(self.cfg.get("character_mcut_enabled", True)))
        self.chk_character_mcut.setToolTip("啟用 MCut 演算法自動決定閾值，會覆蓋上方的手動閾值設定")

        self.chk_drop_overlap = QCheckBox(self.tr("setting_tagger_drop_overlap"))
        self.chk_drop_overlap.setChecked(bool(self.cfg.get("drop_overlap", True)))
        self.chk_drop_overlap.setToolTip("移除重疊標籤，例如同時有 'long hair' 和 'hair' 時只保留更具體的")

        form2.addRow(self.tr("setting_tagger_model"), self.ed_tagger_model)
        form2.addRow(self.tr("setting_tagger_gen_thresh"), self.ed_general_threshold)
        form2.addRow("", self.chk_general_mcut)
        form2.addRow(self.tr("setting_tagger_char_thresh"), self.ed_character_threshold)
        form2.addRow("", self.chk_character_mcut)
        form2.addRow("", self.chk_drop_overlap)

        tagger_layout.addLayout(form2)
        tagger_layout.addStretch(1)
        self.tabs.addTab(tab_tagger, self.tr("setting_tab_tagger"))

        # ---- Text ----
        tab_text = QWidget()
        text_layout = QVBoxLayout(tab_text)
        self.chk_force_lower = QCheckBox(self.tr("setting_text_force_lower"))
        self.chk_force_lower.setChecked(bool(self.cfg.get("english_force_lowercase", True)))
        self.chk_force_lower.setToolTip("勾選後，所有英文標籤和句子會自動轉為小寫，\n符合 Stable Diffusion 訓練資料的常見格式")
        text_layout.addWidget(self.chk_force_lower)

        self.chk_auto_remove_empty = QCheckBox(self.tr("setting_text_auto_remove_empty"))
        self.chk_auto_remove_empty.setChecked(bool(self.cfg.get("text_auto_remove_empty_lines", True)))
        self.chk_auto_remove_empty.setToolTip("自動移除文字檔中的空白行，保持內容整潔")
        text_layout.addWidget(self.chk_auto_remove_empty)

        self.chk_auto_format = QCheckBox(self.tr("setting_text_auto_format"))
        self.chk_auto_format.setChecked(bool(self.cfg.get("text_auto_format", True)))
        self.chk_auto_format.setToolTip("自動整理標籤格式：移除多餘空格、統一用 ', ' 分隔")
        text_layout.addWidget(self.chk_auto_format)

        self.chk_auto_save = QCheckBox(self.tr("setting_text_auto_save"))
        self.chk_auto_save.setChecked(bool(self.cfg.get("text_auto_save", True)))
        self.chk_auto_save.setToolTip("編輯內容時自動儲存到 .txt 檔案，無需手動按儲存")
        text_layout.addWidget(self.chk_auto_save)

        # Batch to txt options
        text_layout.addWidget(self.make_hline())
        text_layout.addWidget(QLabel(f"<b>{self.tr('setting_batch_to_txt')}</b>"))
        
        mode_grp = QGroupBox(self.tr("setting_batch_mode"))
        mode_grp.setToolTip("決定批量處理時如何寫入 .txt 檔案")
        mode_lay = QHBoxLayout()
        self.rb_batch_append = QRadioButton(self.tr("setting_batch_append"))
        self.rb_batch_append.setToolTip("將新內容附加到現有文字的後面 (推薦)")
        self.rb_batch_overwrite = QRadioButton(self.tr("setting_batch_overwrite"))
        self.rb_batch_overwrite.setToolTip("完全覆蓋原有文字，請謹慎使用")
        if self.cfg.get("batch_to_txt_mode", "append") == "overwrite":
            self.rb_batch_overwrite.setChecked(True)
        else:
            self.rb_batch_append.setChecked(True)
        mode_lay.addWidget(self.rb_batch_append)
        mode_lay.addWidget(self.rb_batch_overwrite)
        mode_grp.setLayout(mode_lay)
        text_layout.addWidget(mode_grp)
        
        self.chk_folder_trigger = QCheckBox(self.tr("setting_batch_trigger"))
        self.chk_folder_trigger.setChecked(bool(self.cfg.get("batch_to_txt_folder_trigger", False)))
        self.chk_folder_trigger.setToolTip("勾選後，會把資料夾名稱當作觸發詞加到句子最前面\n例如資料夾 '1girl_miku' 會在開頭加上 'miku'")
        text_layout.addWidget(self.chk_folder_trigger)

        text_layout.addStretch(1)
        self.tabs.addTab(tab_text, self.tr("setting_tab_text"))

        # ---- Mask ----
        tab_mask = QWidget()
        mask_layout = QVBoxLayout(tab_mask)
        form3 = QFormLayout()

        self.ed_mask_alpha = QLineEdit(str(self.cfg.get("mask_default_alpha", 0)))
        self.ed_mask_alpha.setToolTip("去除部分的殘留透明度 (1-254)\n調低：去得更乾淨 (接近全透明)\n調高：保留更多半透明效果")
        self.ed_mask_format = QLineEdit(str(self.cfg.get("mask_default_format", "webp")))
        self.ed_mask_format.setToolTip("輸出格式：webp (檔案小) 或 png (相容性好)")
        form3.addRow(self.tr("setting_mask_alpha"), self.ed_mask_alpha)
        
        # New Settings
        self.spin_mask_padding = QSpinBox()
        self.spin_mask_padding.setRange(0, 50)
        self.spin_mask_padding.setValue(int(self.cfg.get("mask_padding", 3)))
        self.spin_mask_padding.setToolTip("主體邊緣內縮的像素數\n調大：邊緣更乾淨，但可能切到主體\n調小：保留更多邊緣細節")
        form3.addRow("Mask Padding (內縮像素):", self.spin_mask_padding)

        self.spin_mask_blur = QSpinBox()
        self.spin_mask_blur.setRange(0, 50)
        self.spin_mask_blur.setValue(int(self.cfg.get("mask_blur_radius", 10)))
        self.spin_mask_blur.setToolTip("邊緣模糊半徑 (高斯模糊)\n調大：邊緣更柔和自然\n調小：邊緣更銳利")
        form3.addRow("Mask Blur (模糊半徑):", self.spin_mask_blur)

        form3.addRow(self.tr("setting_mask_format"), self.ed_mask_format)

        self.chk_mask_bg_only = QCheckBox(self.tr("setting_mask_only_bg"))
        self.chk_mask_bg_only.setChecked(bool(self.cfg.get("mask_batch_only_if_has_background_tag", False)))
        self.chk_mask_bg_only.setToolTip("勾選後，批量去背只處理標籤含 'background' 的圖片\n避免誤處理不需要去背的圖")
        form3.addRow("", self.chk_mask_bg_only)

        self.chk_mask_ocr = QCheckBox(self.tr("setting_mask_ocr"))
        self.chk_mask_ocr.setChecked(bool(self.cfg.get("mask_batch_detect_text_enabled", True)))
        self.chk_mask_ocr.setToolTip("啟用 OCR 自動偵測並遮蔽圖片中的文字區域")
        form3.addRow("", self.chk_mask_ocr)

        # OCR Advanced
        self.spin_ocr_heat = QDoubleSpinBox()
        self.spin_ocr_heat.setRange(0.01, 1.0)
        self.spin_ocr_heat.setSingleStep(0.05)
        self.spin_ocr_heat.setValue(float(self.cfg.get("mask_ocr_heat_threshold", 0.2)))
        self.spin_ocr_heat.setToolTip(self.tr("setting_ocr_heat_tip"))
        form3.addRow(self.tr("setting_ocr_heat"), self.spin_ocr_heat)

        self.spin_ocr_box = QDoubleSpinBox()
        self.spin_ocr_box.setRange(0.01, 1.0)
        self.spin_ocr_box.setSingleStep(0.05)
        self.spin_ocr_box.setValue(float(self.cfg.get("mask_ocr_box_threshold", 0.6)))
        self.spin_ocr_box.setToolTip(self.tr("setting_ocr_box_tip"))
        form3.addRow(self.tr("setting_ocr_box"), self.spin_ocr_box)

        self.spin_ocr_unclip = QDoubleSpinBox()
        self.spin_ocr_unclip.setRange(1.0, 5.0)
        self.spin_ocr_unclip.setSingleStep(0.1)
        self.spin_ocr_unclip.setValue(float(self.cfg.get("mask_ocr_unclip_ratio", 2.3)))
        self.spin_ocr_unclip.setToolTip(self.tr("setting_ocr_unclip_tip"))
        form3.addRow(self.tr("setting_ocr_unclip"), self.spin_ocr_unclip)

        self.chk_mask_del_npz = QCheckBox(self.tr("setting_mask_delete_npz"))
        self.chk_mask_del_npz.setChecked(bool(self.cfg.get("mask_delete_npz_on_move", True)))
        self.chk_mask_del_npz.setToolTip("移動原圖時自動刪除對應的 .npz 快取檔案 (SD 訓練用)")
        form3.addRow("", self.chk_mask_del_npz)

        mask_layout.addLayout(form3)

        # Batch Ratio Limits
        ratio_box = QGroupBox("Batch Mask 主體佔比限制")
        ratio_box.setToolTip("根據去背後主體佔畫面的比例來決定是否套用去背")
        ratio_lay = QFormLayout()
        
        self.spin_mask_min_ratio = QDoubleSpinBox()
        self.spin_mask_min_ratio.setRange(0.0, 1.0)
        self.spin_mask_min_ratio.setSingleStep(0.05)
        self.spin_mask_min_ratio.setValue(float(self.cfg.get("mask_batch_min_foreground_ratio", 0.1)))
        self.spin_mask_min_ratio.setToolTip("主體佔比下限。若主體太小 (佔比低於此值)，可能是誤判，跳過不處理")
        ratio_lay.addRow("Min Ratio (主體過小跳過):", self.spin_mask_min_ratio)

        self.spin_mask_max_ratio = QDoubleSpinBox()
        self.spin_mask_max_ratio.setRange(0.0, 1.0)
        self.spin_mask_max_ratio.setSingleStep(0.05)
        self.spin_mask_max_ratio.setValue(float(self.cfg.get("mask_batch_max_foreground_ratio", 0.8)))
        self.spin_mask_max_ratio.setToolTip("主體佔比上限。若主體佔滿畫面 (無背景可去)，跳過不處理")
        ratio_lay.addRow("Max Ratio (主體過大跳過):", self.spin_mask_max_ratio)
        
        self.chk_skip_scenery = QCheckBox("跳過場景圖 (含 indoors/outdoors 標籤)")
        self.chk_skip_scenery.setChecked(bool(self.cfg.get("mask_batch_skip_if_scenery_tag", True)))
        self.chk_skip_scenery.setToolTip("勾選後，若標籤含 indoors 或 outdoors (場景圖)，則跳過去背")
        ratio_lay.addRow("", self.chk_skip_scenery)

        ratio_box.setLayout(ratio_lay)
        mask_layout.addWidget(ratio_box)

        # The original hint label is removed as the tooltip is now on chk_mask_ocr
        mask_layout.addStretch(1)
        self.tabs.addTab(tab_mask, self.tr("setting_tab_mask"))

        # ---- Tags Filter (Character Tags) ----
        tab_filter = QWidget()
        filter_layout = QVBoxLayout(tab_filter)
        filter_layout.addWidget(QLabel(self.tr("setting_filter_title")))
        filter_layout.addWidget(QLabel(self.tr("setting_filter_info")))
        
        # f_form = QFormLayout()  <-- Remove FormLayout to stacking
        
        bl_label = QLabel(self.tr("setting_bl_words"))
        bl_label.setToolTip("包含這些關鍵字的標籤會被標記為『特徵標籤』(紅框)，\n批量寫入 txt 時可選擇自動刪除")
        filter_layout.addWidget(bl_label)
        self.ed_bl_words = QPlainTextEdit()
        self.ed_bl_words.setPlainText(", ".join(self.cfg.get("char_tag_blacklist_words", [])))
        self.ed_bl_words.setMinimumHeight(120)
        self.ed_bl_words.setToolTip("例如: hair, eyes, skin 等通用外觀描述\n這些標籤適合用於 LoRA 訓練時過濾")
        filter_layout.addWidget(self.ed_bl_words)

        wl_label = QLabel(self.tr("setting_wl_words"))
        wl_label.setToolTip("包含這些關鍵字的標籤即使符合黑名單也不會被標記")
        filter_layout.addWidget(wl_label)
        self.ed_wl_words = QPlainTextEdit()
        self.ed_wl_words.setPlainText(", ".join(self.cfg.get("char_tag_whitelist_words", [])))
        self.ed_wl_words.setMinimumHeight(80)
        self.ed_wl_words.setToolTip("例如: holding hair, background 等動作或情境描述\n這些標籤不是角色固有特徵，應該保留")
        filter_layout.addWidget(self.ed_wl_words)
        
        # filter_layout.addLayout(f_form)
        
        filter_layout.addStretch(1)
        
        self.tabs.addTab(tab_filter, self.tr("setting_tab_filter"))

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QPushButton(self.tr("setting_save"))
        self.btn_cancel = QPushButton(self.tr("setting_cancel"))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        self.layout.addLayout(btns)

    def _parse_tags(self, s: str):
        raw = (s or "").strip()
        if not raw:
            return []
        if "\n" in raw:
            parts = [x.strip() for x in raw.splitlines() if x.strip()]
        else:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
        return [p.replace("_", " ").strip() for p in parts if p.strip()]

    def make_hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def get_cfg(self) -> dict:
        cfg = dict(self.cfg)

        cfg["llm_base_url"] = self.ed_base_url.text().strip() or DEFAULT_APP_SETTINGS["llm_base_url"]
        cfg["llm_api_key"] = self.ed_api_key.text().strip()
        cfg["llm_model"] = self.ed_model.text().strip() or DEFAULT_APP_SETTINGS["llm_model"]
        cfg["llm_system_prompt"] = self.ed_system_prompt.toPlainText()
        cfg["llm_user_prompt_template"] = self.ed_user_template.toPlainText()
        cfg["llm_custom_prompt_template"] = self.ed_custom_template.toPlainText()
        cfg["llm_max_image_dimension"] = self.spin_llm_dim.value()
        cfg["llm_skip_nsfw_on_batch"] = self.chk_llm_skip_nsfw.isChecked()
        cfg["llm_use_gray_mask"] = self.chk_llm_use_gray_mask.isChecked()
        cfg["default_custom_tags"] = self._parse_tags(self.ed_default_custom_tags.toPlainText())

        cfg["tagger_model"] = self.ed_tagger_model.text().strip() or DEFAULT_APP_SETTINGS["tagger_model"]
        cfg["general_threshold"] = _coerce_float(self.ed_general_threshold.text(), DEFAULT_APP_SETTINGS["general_threshold"])
        cfg["general_mcut_enabled"] = self.chk_general_mcut.isChecked()
        cfg["character_threshold"] = _coerce_float(self.ed_character_threshold.text(), DEFAULT_APP_SETTINGS["character_threshold"])
        cfg["character_mcut_enabled"] = self.chk_character_mcut.isChecked()
        cfg["drop_overlap"] = self.chk_drop_overlap.isChecked()

        cfg["english_force_lowercase"] = self.chk_force_lower.isChecked()
        cfg["text_auto_remove_empty_lines"] = self.chk_auto_remove_empty.isChecked()
        cfg["text_auto_format"] = self.chk_auto_format.isChecked()
        cfg["text_auto_save"] = self.chk_auto_save.isChecked()
        cfg["batch_to_txt_mode"] = "overwrite" if self.rb_batch_overwrite.isChecked() else "append"
        cfg["batch_to_txt_folder_trigger"] = self.chk_folder_trigger.isChecked()

        a = _coerce_int(self.ed_mask_alpha.text(), DEFAULT_APP_SETTINGS["mask_default_alpha"])
        # Rule: 1-254 (USER request: "1-254 才對 (以防RGB丟失)")
        a = max(1, min(254, a))
        
        fmt = (self.ed_mask_format.text().strip().lower() or DEFAULT_APP_SETTINGS["mask_default_format"]).strip(".")
        if fmt not in ("webp", "png"):
            fmt = DEFAULT_APP_SETTINGS["mask_default_format"]

        cfg["mask_default_alpha"] = a
        cfg["mask_default_format"] = fmt
        cfg["mask_padding"] = self.spin_mask_padding.value()
        cfg["mask_blur_radius"] = self.spin_mask_blur.value()
        cfg["mask_batch_only_if_has_background_tag"] = self.chk_mask_bg_only.isChecked()
        cfg["mask_batch_detect_text_enabled"] = self.chk_mask_ocr.isChecked()
        cfg["mask_delete_npz_on_move"] = self.chk_mask_del_npz.isChecked()
        cfg["mask_ocr_heat_threshold"] = float(f"{self.spin_ocr_heat.value():.2f}")
        cfg["mask_ocr_box_threshold"] = float(f"{self.spin_ocr_box.value():.2f}")
        cfg["mask_ocr_unclip_ratio"] = float(f"{self.spin_ocr_unclip.value():.2f}")
        cfg["mask_batch_min_foreground_ratio"] = float(f"{self.spin_mask_min_ratio.value():.2f}")
        cfg["mask_batch_max_foreground_ratio"] = float(f"{self.spin_mask_max_ratio.value():.2f}")
        cfg["mask_batch_skip_if_scenery_tag"] = self.chk_skip_scenery.isChecked()

        # Tags Filter
        cfg["char_tag_blacklist_words"] = self._parse_tags(self.ed_bl_words.toPlainText())
        cfg["char_tag_whitelist_words"] = self._parse_tags(self.ed_wl_words.toPlainText())

        cfg["ui_language"] = self.cb_lang.currentData()
        cfg["ui_theme"] = "dark" if self.rb_dark.isChecked() else "light"

        return cfg


# ==========================================
#  Main Window
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Captioning Assistant")
        self._clip_tokenizer = None
        self.resize(1600, 1000)

        self.settings = load_app_settings()

        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", DEFAULT_APP_SETTINGS["llm_api_key"]))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        self.default_user_prompt_template = str(self.settings.get("llm_user_prompt_template", DEFAULT_APP_SETTINGS["llm_user_prompt_template"]))
        self.custom_prompt_template = str(self.settings.get("llm_custom_prompt_template", DEFAULT_APP_SETTINGS.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
        self.current_prompt_mode = "default"
        self.default_custom_tags_global = list(self.settings.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
        self.english_force_lowercase = bool(self.settings.get("english_force_lowercase", True))

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

        # Image View State
        self.current_pil_image = None  # Store original PIL image
        self.view_mode = 0             # 0: Original, 1: RGB, 2: Alpha
        self.temp_view_mode = -1       # For Key Press override

        self.batch_tagger_thread = None
        self.batch_llm_thread = None
        self.batch_unmask_thread = None
        self.batch_mask_text_thread = None

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
        return LOCALIZATION.get(lang, LOCALIZATION["zh_tw"]).get(key, key)

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

        # ✅ View Mode Dropdown
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems(["圖片 (Original)", "RGB Only", "Alpha Channel"])
        self.combo_view_mode.setToolTip("顯示模式切換\n(按住 N 顯示 RGB, 按住 M 顯示 Alpha)")
        self.combo_view_mode.currentIndexChanged.connect(self.on_view_mode_changed)
        filter_bar.addWidget(self.combo_view_mode)
        
        left_layout.addLayout(filter_bar)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_image_context_menu)
        self.image_label.setMinimumSize(400, 400)

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
    def refresh_file_list(self):
        if not self.root_dir_path or not os.path.exists(self.root_dir_path):
            return
        
        dir_path = self.root_dir_path
        # Keep track of current file to restore selection
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

            # ✅ Load PIL Image & Update View
            try:
                self.current_pil_image = Image.open(self.current_image_path)
                self.current_pil_image = ImageOps.exif_transpose(self.current_pil_image)
            except Exception as e:
                print(f"Error loading image: {e}")
                self.current_pil_image = None
            
            self.update_current_view_pixmap()

    def update_current_view_pixmap(self):
        """根據 View Mode 生成顯示用的 Pixmap"""
        if self.current_pil_image is None:
            self.image_label.clear()
            self.current_pixmap = QPixmap()
            return

        mode = self.view_mode
        # Key override
        if self.temp_view_mode != -1:
            mode = self.temp_view_mode
        
        try:
            display_img = self.current_pil_image
            
            if mode == 1: # RGB Only
                display_img = self.current_pil_image.convert("RGB")
            elif mode == 2: # Alpha Channel
                if self.current_pil_image.mode in ('RGBA', 'LA') or 'A' in self.current_pil_image.getbands():
                    display_img = self.current_pil_image.getchannel('A')
                else:
                    # No alpha, show white
                    display_img = Image.new("L", self.current_pil_image.size, 255)

            # Convert to QPixmap
            # Handle Grayscale 'L' mode -> convert to RGB for consistency in QT
            if display_img.mode == 'L':
                display_img = display_img.convert("RGB")
            
            im_qt = ImageQt.ImageQt(display_img)
            self.current_pixmap = QPixmap.fromImage(im_qt)
            
            if not self.current_pixmap.isNull():
                self.update_image_display()
            else:
                self.image_label.clear()
        
        except Exception as e:
            print(f"Error updating view pixmap: {e}")

    def keyPressEvent(self, event):
        # 如果焦點在輸入框，不攔截按鍵
        if self.focusWidget() in [self.filter_input, self.index_input, self.prompt_edit, self.txt_edit, self.ed_bl_words, self.ed_wl_words]:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key.Key_N:
            if self.temp_view_mode != 1:
                self.temp_view_mode = 1
                self.update_current_view_pixmap()
        elif event.key() == Qt.Key.Key_M:
             if self.temp_view_mode != 2:
                self.temp_view_mode = 2
                self.update_current_view_pixmap()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return

        if event.key() == Qt.Key.Key_N and self.temp_view_mode == 1:
            self.temp_view_mode = -1
            self.update_current_view_pixmap()
        elif event.key() == Qt.Key.Key_M and self.temp_view_mode == 2:
            self.temp_view_mode = -1
            self.update_current_view_pixmap()
        else:
            super().keyReleaseEvent(event)

    def on_view_mode_changed(self, index):
        self.view_mode = index
        self.update_current_view_pixmap()

    def show_image_context_menu(self, pos):
        if not self.current_image_path:
            return

        menu = QMenu(self)
        
        action_copy_img = QAction(QIcon(), "複製圖片 (Copy Image)", self)
        action_copy_img.triggered.connect(self.copy_image_to_clipboard)
        menu.addAction(action_copy_img)

        action_copy_path = QAction(QIcon(), "複製路徑 (Copy Path)", self)
        action_copy_path.triggered.connect(self.copy_image_path)
        menu.addAction(action_copy_path)

        action_open_explorer = QAction(QIcon(), "打開檔案位置 (Show in Explorer)", self)
        action_open_explorer.triggered.connect(self.open_file_explorer)
        menu.addAction(action_open_explorer)

        menu.exec(self.image_label.mapToGlobal(pos))

    def copy_image_to_clipboard(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.current_pixmap)
            self.statusBar().showMessage("圖片已複製到剪貼簿", 2000)

    def copy_image_path(self):
        if self.current_image_path:
            QApplication.clipboard().setText(self.current_image_path)
            self.statusBar().showMessage("路徑已複製到剪貼簿", 2000)

    def open_file_explorer(self):
        if self.current_image_path:
            # 針對 Windows 的選取檔案功能
            import subprocess
            path = os.path.normpath(self.current_image_path)
            try:
                subprocess.Popen(f'explorer /select,"{path}"')
            except Exception as e:
                print(f"Open explorer failed: {e}")
                # Fallback
                folder = os.path.dirname(path)
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

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
        scaled = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

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
        self.btn_auto_tag.setEnabled(False)
        self.btn_auto_tag.setText("Tagging...")

        self.tagger_thread = TaggerWorker(self.current_image_path, self.settings)
        self.tagger_thread.finished.connect(self.on_tagger_finished)
        self.tagger_thread.error.connect(self.on_tagger_error_no_popup)
        self.tagger_thread.start()

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

        self.llm_thread = LLMWorker(
            self.llm_base_url,
            self.api_key,
            self.model_name,
            self.llm_system_prompt,
            user_prompt,
            self.current_image_path,
            tags_text,
            max_dim=int(self.settings.get("llm_max_image_dimension", 1024)),
            settings=self.settings
        )
        self.llm_thread.finished.connect(self.on_llm_finished_latest_only)
        self.llm_thread.error.connect(self.on_llm_error_no_popup)
        self.llm_thread.start()

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
        if Remover is None:
            QMessageBox.warning(self, "Unmask", "transparent_background.Remover not available")
            return
            
        # Use BatchWorker for single image to support progress bar & async
        self.btn_batch_unmask_thread = BatchUnmaskWorker(
            [self.current_image_path], 
            self.settings,
            background_tag_checker=None,  # Force process for single image
            is_batch=False
        )
        self.btn_batch_unmask_thread.progress.connect(self.show_progress)
        self.btn_batch_unmask_thread.per_image.connect(self.on_batch_unmask_per_image)
        self.btn_batch_unmask_thread.done.connect(self.on_unmask_single_done)
        self.btn_batch_unmask_thread.error.connect(lambda e: QMessageBox.warning(self, "Error", f"Unmask 失敗: {e}"))
        self.btn_batch_unmask_thread.start()

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
            QMessageBox.information(self, "Info", "OCR text detection is disabled in settings.")
            return

        image_path = self.current_image_path
        # Use BatchMaskTextWorker method logic manually
        # Needs to detect text and process
        # For single image, we can just instantiate a worker for 1 item
        if detect_text_with_ocr is None:
             QMessageBox.warning(self, "Mask Text", self.tr("setting_mask_ocr_hint"))
             return

        # Simple approach: reuse logic by making a list of 1
        # reusing worker might be complex due to threading, let's run logic directly?
        # Re-using worker for single image is safer to keep logic consistent.
        
        self.batch_mask_text_thread = BatchMaskTextWorker(
            [image_path], 
            self.settings, 
            background_tag_checker=None,
            is_batch=False  # 單圖強制執行
        )
        self.batch_mask_text_thread.progress.connect(self.show_progress)
        self.batch_mask_text_thread.per_image.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.done.connect(lambda: self.on_batch_done("Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Mask Text", e))
        self.batch_mask_text_thread.start()

    def restore_current_image(self):
        """還原當前圖片為原始備份 (從 raw_image 資料夾)"""
        if not self.current_image_path:
            return
        
        if not has_raw_backup(self.current_image_path):
            QMessageBox.information(self, "Restore", "找不到原圖備份紀錄\n(可能尚未進行任何去背/去文字處理)")
            return
            
        try:
            success = restore_raw_image(self.current_image_path)
            if success:
                self.load_image()
                self.statusBar().showMessage("已還原原圖", 3000)
            else:
                QMessageBox.warning(self, "Restore", "還原失敗：備份檔案可能已遺失")
        except Exception as e:
            QMessageBox.warning(self, "Restore", f"還原失敗: {e}")

    def run_batch_unmask_background(self):
        if not self.image_files:
            return
        if Remover is None:
            QMessageBox.warning(self, "Batch Unmask", "transparent_background.Remover not available")
            return

        # ✅ 修正：根據設定決定是否過濾
        only_bg = bool(self.settings.get("mask_batch_only_if_has_background_tag", False))
        if only_bg:
            targets = [p for p in self.image_files if self._image_has_background_tag(p)]
            if not targets:
                QMessageBox.information(self, "Batch Unmask", "找不到含有 'background' 標籤的圖片")
                return
        else:
            targets = self.image_files

        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(False)
            
        self.batch_unmask_thread = BatchUnmaskWorker(
            targets, 
            self.settings,
            background_tag_checker=self._image_has_background_tag
        )
        self.batch_unmask_thread.progress.connect(self.show_progress)
        self.batch_unmask_thread.per_image.connect(self.on_batch_unmask_per_image)
        self.batch_unmask_thread.done.connect(self.on_batch_unmask_done)
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

        self.btn_batch_tagger.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)

        self.batch_tagger_thread = TaggerWorker(self.image_files, self.settings) if isinstance(self.image_files, str) else BatchTaggerWorker(self.image_files, self.settings)
        self.batch_tagger_thread.progress.connect(self.show_progress)
        self.batch_tagger_thread.per_image.connect(self.on_batch_tagger_per_image)
        self.batch_tagger_thread.done.connect(self.on_batch_tagger_done)
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

            self.batch_tagger_thread = BatchTaggerWorker(files_to_process, self.settings)
            self.batch_tagger_thread.progress.connect(lambda i, t, n: self.show_progress(i, t, n)) # Re-bind if progress uses 3 args
            self.batch_tagger_thread.per_image.connect(self.on_batch_tagger_per_image)
            self.batch_tagger_thread.done.connect(self.on_batch_tagger_done)
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
            self.batch_llm_thread = BatchLLMWorker(
                self.llm_base_url,
                self.api_key,
                self.model_name,
                self.llm_system_prompt,
                user_prompt,
                files_to_process,  # List of missing paths
                self.build_llm_tags_context_for_image,
                max_dim=int(self.settings.get("llm_max_image_dimension", 1024)),
                skip_nsfw=bool(self.settings.get("llm_skip_nsfw_on_batch", False))
            )
            self.batch_llm_thread.progress.connect(self.show_progress)
            self.batch_llm_thread.per_image.connect(self.on_batch_llm_per_image)
            self.batch_llm_thread.done.connect(self.on_batch_llm_done)
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

        self.batch_llm_thread = BatchLLMWorker(
            self.llm_base_url,
            self.api_key,
            self.model_name,
            self.llm_system_prompt,
            user_prompt,
            self.image_files,
            self.build_llm_tags_context_for_image,
            max_dim=int(self.settings.get("llm_max_image_dimension", 1024)),
            skip_nsfw=bool(self.settings.get("llm_skip_nsfw_on_batch", False)),
            settings=self.settings
        )
        self.batch_llm_thread.progress.connect(self.show_progress)
        self.batch_llm_thread.per_image.connect(self.on_batch_llm_per_image)
        self.batch_llm_thread.done.connect(self.on_batch_llm_done)
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

        self.batch_mask_text_thread = BatchMaskTextWorker(
            self.image_files,
            self.settings,
            background_tag_checker=self._image_has_background_tag
        )
        self.batch_mask_text_thread.progress.connect(lambda i, t, name: self.show_progress(i, t, name))
        self.batch_mask_text_thread.per_image.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.done.connect(lambda: self.on_batch_done("Batch Mask Text 完成"))
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

        self.batch_restore_thread = BatchRestoreWorker(self.image_files)
        # Using lambda for progress signature matching
        self.batch_restore_thread.progress.connect(lambda i, t, name: self.show_progress(i, t, name))
        self.batch_restore_thread.per_image.connect(self.on_batch_restore_per_image)
        self.batch_restore_thread.done.connect(lambda: self.on_batch_done("Batch Restore 完成"))
        self.batch_restore_thread.error.connect(lambda e: self.on_batch_error("Batch Restore", e))
        self.batch_restore_thread.start()

    def cancel_batch(self):
        self.statusBar().showMessage("正在中止...", 2000)
        for attr in ['batch_unmask_thread', 'batch_mask_text_thread', 'batch_restore_thread', 'batch_tagger_thread', 'batch_llm_thread']:
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())