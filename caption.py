# ============================================================
#  Caption 神器 - 索引 (INDEX)
# ============================================================
#
# [Ln 32-97]    Imports & 外部依賴
# [Ln 98-300]   Configuration, I18n Resource & Globals
# [Ln 302-380]  Settings Helpers (load/save/coerce 函式)
# [Ln 382-530]  Utils: delete_matching_npz、JSON sidecar、checkerboard
# [Ln 532-700]  Utils / Parsing / Tag Logic (含 is_basic_character_tag)
# [Ln 702-1000] Workers (TaggerWorker, LLMWorker, BatchTaggerWorker, BatchLLMWorker)
# [Ln 1002-1160] BatchMaskTextWorker (OCR 批次遮罩)
# [Ln 1162-1260] BatchUnmaskWorker (批次去背)
# [Ln 1262-1360] StrokeCanvas & StrokeEraseDialog (手繪橡皮擦)
# [Ln 1362-1590] UI Components (TagButton 多主題適配、TagFlowWidget、Advanced Find/Replace)
# [Ln 1592-1880] SettingsDialog (UI 分頁新增語言/主題、I18n 覆蓋)
# [Ln 1882-1940] MainWindow.__init__ (主視窗初始化、tr 輔助函式)
# [Ln 1942-2190] MainWindow.init_ui (UI 佈局、分頁、按鈕)
# [Ln 2192-2260] MainWindow 快捷鍵 & 滾輪事件
# [Ln 2262-2430] MainWindow 檔案讀取 & on_text_changed
# [Ln 2432-2500] MainWindow Token 計數
# [Ln 2502-2680] MainWindow TAGS/NL (JSON sidecar 整合)
# [Ln 2682-2770] MainWindow NL Paging
# [Ln 2772-2920] MainWindow Tag 插入/移除 & Tagger
# [Ln 2922-3060] MainWindow LLM 生成
# [Ln 3062-3150] MainWindow Tools: Unmask / Stroke Eraser
# [Ln 3152-3290] MainWindow Batch Tagger / LLM to txt (含寫入與過濾邏輯)
# [Ln 3292-3560] MainWindow Settings 儲存、retranslate_ui & main 入口
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

# silence some noisy third-party warnings
warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_fwd")
warnings.filterwarnings("ignore", message="Failed to import flet")

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
    QFrame, QProgressBar, QSlider
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, 
    QBuffer, QIODevice, QByteArray
)
from PyQt6.QtGui import (
    QPixmap, QKeySequence, QAction, QShortcut, QFont,
    QPalette, QBrush, QPainter, QPen, QColor, QImage, QTextCursor
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
except Exception:
    Remover = None

from openai import OpenAI
from imgutils.tagging import get_wd14_tags, tags_to_text, remove_underline

# optional: OCR for text box detection (batch mask text)
try:
    from imgutils.ocr import detect_text_with_ocr
except Exception:
    detect_text_with_ocr = None

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
        "btn_find_replace": "尋找/取代",
        "btn_undo": "復原文字",
        "btn_redo": "重做文字",
        "btn_unmask": "單圖去背景",
        "btn_batch_unmask": "Batch 去背景",
        "btn_mask_text": "單圖去文字",
        "btn_batch_mask_text": "Batch 去文字",
        "btn_restore_original": "放回原檔",
        "btn_stroke_eraser": "手繪橡皮擦",
        "btn_cancel_batch": "中止",
        "menu_tools": "工具",
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
        "setting_tagger_gen_thresh": "一般標籤閾值:",
        "setting_tagger_char_thresh": "特徵標籤閾值:",
        "setting_tagger_gen_mcut": "一般標籤 MCut",
        "setting_tagger_char_mcut": "特徵標籤 MCut",
        "setting_tagger_drop_overlap": "移除重疊標籤",
        "setting_mask_ocr_hint": "OCR 需要 imgutils，未安裝則略過。",
    },
    "en": {
        "app_title": "Caption Tool",
        "menu_file": "File",
        "menu_open_dir": "Open Directory",
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
        "btn_find_replace": "Find/Replace",
        "btn_undo": "Undo Txt",
        "btn_redo": "Redo Txt",
        "btn_unmask": "Unmask Background",
        "btn_batch_unmask": "Batch Unmask Background",
        "btn_mask_text": "Unmask Text",
        "btn_batch_mask_text": "Batch Unmask Text",
        "btn_restore_original": "Restore Original",
        "btn_stroke_eraser": "Stroke Eraser",
        "btn_cancel_batch": "Cancel",
        "menu_tools": "Tools",
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
        "setting_tagger_gen_thresh": "General Threshold:",
        "setting_tagger_char_thresh": "Character Threshold:",
        "setting_tagger_gen_mcut": "General MCut Enabled",
        "setting_tagger_char_mcut": "Character MCut Enabled",
        "setting_tagger_drop_overlap": "Drop Overlap",
        "setting_mask_ocr_hint": "OCR relies on imgutils.ocr.detect_text_with_ocr; skips if not installed.",
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
    "default_custom_tags": list(DEFAULT_CUSTOM_TAGS),

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

    # Character Tags Filter (simple word matching)
    # 黑名單：包含這些 word 的 tag/句子會被標記
    "char_tag_blacklist_words": ["hair", "eyes", "skin", "bun", "bangs", "sidelocks", "twintails", "braid", "ponytail", "beard", "mustache", "ear", "horn", "tail", "wing", "breast", "mole", "halo", "glasses", "fang", "heterochromia", "headband", "freckles", "lip", "eyebrows", "eyelashes"],
    # 白名單：若包含這些 word，即使符合黑名單也不標記
    "char_tag_whitelist_words": ["holding", "hand", "sitting", "covering", "playing", "background", "looking"],

    # Mask / batch mask text
    "mask_default_alpha": 0,  # 0-255, 0 = fully transparent
    "mask_default_format": "webp",  # webp | png
    "mask_batch_only_if_has_background_tag": False,
    "mask_batch_detect_text_enabled": True,  # if off, never call detect_text_with_ocr
    "mask_delete_npz_on_move": True,         # 移動舊圖時刪除對應 npz

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


class LLMWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model_name, system_prompt, user_prompt, image_path, tags_context):
        super().__init__()
        self.base_url = base_url
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_path = image_path
        self.tags_context = tags_context

    def run(self):
        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            img = Image.open(self.image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            ratio = min(1024 / img.width, 1024 / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{img_str}"

            final_user_content = self.user_prompt.replace("{LLM處理結果}", self.tags_context)
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


class BatchLLMWorker(QThread):
    progress = pyqtSignal(int, int, str)
    per_image = pyqtSignal(str, str)  # image_path, nl_content
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model_name, system_prompt, user_prompt, image_paths, tags_context_getter):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.image_paths = list(image_paths)
        self.tags_context_getter = tags_context_getter
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

                    img = Image.open(p)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    ratio = min(1024 / img.width, 1024 / img.height)
                    if ratio < 1:
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=90)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_url = f"data:image/jpeg;base64,{img_str}"

                    final_user_content = self.user_prompt.replace("{LLM處理結果}", tags_context)
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

    def __init__(self, image_paths, cfg: dict, background_tag_checker=None):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self.background_tag_checker = background_tag_checker
        self._stop = False

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

    def _should_process(self, image_path: str) -> bool:
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
            results = detect_text_with_ocr(image_path)
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
                self.progress.emit(i, total, os.path.basename(pth))

                if not self._should_process(pth):
                    continue

                boxes = self._detect_text_boxes(pth)
                if not boxes:
                    continue

                src_dir = os.path.dirname(pth)
                out_dir = os.path.join(src_dir, "unmask")
                os.makedirs(out_dir, exist_ok=True)

                base_no_ext = os.path.splitext(pth)[0]
                out_path = base_no_ext + f".{fmt}"
                if os.path.exists(out_path) and os.path.abspath(out_path) != os.path.abspath(pth):
                    out_path = self._unique_path(out_path)

                ext = os.path.splitext(pth)[1].lower()
                if ext == f".{fmt}":
                    moved_original = self._unique_path(os.path.join(out_dir, os.path.basename(pth)))
                    shutil.move(pth, moved_original)
                    # 刪除對應 npz
                    if self.cfg.get("mask_delete_npz_on_move", True):
                        delete_matching_npz(pth)
                    src_for_processing = moved_original
                    out_path = pth
                else:
                    src_for_processing = pth

                with Image.open(src_for_processing) as img:
                    img_rgba = img.convert("RGBA")
                    a = np.array(img_rgba.getchannel("A"), dtype=np.uint8)
                    for (x1, y1, x2, y2) in boxes:
                        x1 = max(0, int(x1)); y1 = max(0, int(y1))
                        x2 = min(a.shape[1], int(x2)); y2 = min(a.shape[0], int(y2))
                        if x2 > x1 and y2 > y1:
                            a[y1:y2, x1:x2] = alpha_val
                    img_rgba.putalpha(Image.fromarray(a, mode="L"))
                    if fmt == "png":
                        img_rgba.save(out_path, "PNG")
                    else:
                        img_rgba.save(out_path, "WEBP")

                if ext != f".{fmt}":
                    moved_original = self._unique_path(os.path.join(out_dir, os.path.basename(pth)))
                    shutil.move(pth, moved_original)
                    # 刪除對應 npz
                    if self.cfg.get("mask_delete_npz_on_move", True):
                        delete_matching_npz(pth)

                # 記錄 masked_text 到 JSON sidecar
                sidecar = load_image_sidecar(out_path)
                sidecar["masked_text"] = True
                save_image_sidecar(out_path, sidecar)

                self.per_image.emit(pth, out_path)

            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchUnmaskWorker(QThread):
    progress = pyqtSignal(int, int, str)   # i, total, filename
    per_image = pyqtSignal(str, str)       # old_path, new_path
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_paths, cfg: dict = None):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self._stop = False

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
    def remove_background_to_webp(image_path: str, remover) -> str:
        if not image_path:
            return ""

        src_dir = os.path.dirname(image_path)
        unmask_dir = os.path.join(src_dir, "unmask")
        os.makedirs(unmask_dir, exist_ok=True)

        ext = os.path.splitext(image_path)[1].lower()
        base_no_ext = os.path.splitext(image_path)[0]

        # target file: always WEBP
        target_file = base_no_ext + ".webp"
        if os.path.exists(target_file) and os.path.abspath(target_file) != os.path.abspath(image_path):
            target_file = BatchUnmaskWorker._unique_path(target_file)

        # move original first if it is already WEBP (avoid overwrite)
        moved_original = ""
        if ext == ".webp":
            moved_original = BatchUnmaskWorker._unique_path(os.path.join(unmask_dir, os.path.basename(image_path)))
            shutil.move(image_path, moved_original)
            src_for_processing = moved_original
            target_file = image_path  # write back to original path
        else:
            src_for_processing = image_path

        with Image.open(src_for_processing) as img:
            img = img.convert('RGB')
            img_rm = remover.process(img, type='rgba')
            img_rm.save(target_file, 'WEBP')

        # move original (non-webp) after success
        if ext != ".webp":
            moved_original = BatchUnmaskWorker._unique_path(os.path.join(unmask_dir, os.path.basename(image_path)))
            shutil.move(image_path, moved_original)

        return target_file, image_path  # 回傳 (new_path, old_path) 以便外部處理 npz 刪除

    def run(self):
        try:
            if Remover is None:
                self.error.emit("transparent_background.Remover not available")
                return

            remover = Remover()

            total = len(self.image_paths)
            for i, p in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                self.progress.emit(i, total, os.path.basename(p))
                
                # Check sidecar to skip
                sidecar = load_image_sidecar(p)
                if sidecar.get("masked_background", False):
                    continue

                try:
                    result = self.remove_background_to_webp(p, remover)
                    if result:
                        new_path, old_path = result
                        # 刪除對應 npz
                        if self.cfg.get("mask_delete_npz_on_move", True):
                            delete_matching_npz(old_path)
                        # 記錄 masked_background 到 JSON sidecar
                        sidecar = load_image_sidecar(new_path)
                        sidecar["masked_background"] = True
                        save_image_sidecar(new_path, sidecar)
                        self.per_image.emit(p, new_path)
                except Exception as e:
                    print(f"[BatchUnmask] {p} 失敗: {e}")

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
        current_tokens = split_csv_like_text(active_text_content)
        current_norm = set(normalize_for_match(t) for t in current_tokens)

        for tag, btn in self.buttons.items():
            btn.blockSignals(True)
            is_active = normalize_for_match(tag) in current_norm
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
        form.addRow("Model:", self.ed_model)
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
        self.ed_general_threshold = QLineEdit(str(self.cfg.get("general_threshold", 0.2)))
        self.chk_general_mcut = QCheckBox(self.tr("setting_tagger_gen_mcut"))
        self.chk_general_mcut.setChecked(bool(self.cfg.get("general_mcut_enabled", False)))

        self.ed_character_threshold = QLineEdit(str(self.cfg.get("character_threshold", 0.85)))
        self.chk_character_mcut = QCheckBox(self.tr("setting_tagger_char_mcut"))
        self.chk_character_mcut.setChecked(bool(self.cfg.get("character_mcut_enabled", True)))

        self.chk_drop_overlap = QCheckBox(self.tr("setting_tagger_drop_overlap"))
        self.chk_drop_overlap.setChecked(bool(self.cfg.get("drop_overlap", True)))

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
        text_layout.addWidget(self.chk_force_lower)

        self.chk_auto_remove_empty = QCheckBox(self.tr("setting_text_auto_remove_empty"))
        self.chk_auto_remove_empty.setChecked(bool(self.cfg.get("text_auto_remove_empty_lines", True)))
        text_layout.addWidget(self.chk_auto_remove_empty)

        self.chk_auto_format = QCheckBox(self.tr("setting_text_auto_format"))
        self.chk_auto_format.setChecked(bool(self.cfg.get("text_auto_format", True)))
        text_layout.addWidget(self.chk_auto_format)

        self.chk_auto_save = QCheckBox(self.tr("setting_text_auto_save"))
        self.chk_auto_save.setChecked(bool(self.cfg.get("text_auto_save", True)))
        text_layout.addWidget(self.chk_auto_save)

        # Batch to txt options
        text_layout.addWidget(self.make_hline())
        text_layout.addWidget(QLabel(f"<b>{self.tr('setting_batch_to_txt')}</b>"))
        
        mode_grp = QGroupBox(self.tr("setting_batch_mode"))
        mode_lay = QHBoxLayout()
        self.rb_batch_append = QRadioButton(self.tr("setting_batch_append"))
        self.rb_batch_overwrite = QRadioButton(self.tr("setting_batch_overwrite"))
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
        text_layout.addWidget(self.chk_folder_trigger)

        text_layout.addStretch(1)
        self.tabs.addTab(tab_text, self.tr("setting_tab_text"))

        # ---- Mask ----
        tab_mask = QWidget()
        mask_layout = QVBoxLayout(tab_mask)
        form3 = QFormLayout()

        self.ed_mask_alpha = QLineEdit(str(self.cfg.get("mask_default_alpha", 0)))
        self.ed_mask_format = QLineEdit(str(self.cfg.get("mask_default_format", "webp")))
        form3.addRow(self.tr("setting_mask_alpha"), self.ed_mask_alpha)
        form3.addRow(self.tr("setting_mask_format"), self.ed_mask_format)

        self.chk_only_bg = QCheckBox(self.tr("setting_mask_only_bg"))
        self.chk_only_bg.setChecked(bool(self.cfg.get("mask_batch_only_if_has_background_tag", False)))

        self.chk_detect_text = QCheckBox(self.tr("setting_mask_ocr"))
        self.chk_detect_text.setChecked(bool(self.cfg.get("mask_batch_detect_text_enabled", True)))

        mask_layout.addLayout(form3)
        mask_layout.addWidget(self.chk_only_bg)
        mask_layout.addWidget(self.chk_detect_text)

        self.chk_delete_npz = QCheckBox(self.tr("setting_mask_delete_npz"))
        self.chk_delete_npz.setChecked(bool(self.cfg.get("mask_delete_npz_on_move", True)))
        mask_layout.addWidget(self.chk_delete_npz)

        hint = QLabel(self.tr("setting_mask_ocr_hint"))
        hint.setStyleSheet("color:#666;")
        mask_layout.addWidget(hint)
        mask_layout.addStretch(1)
        self.tabs.addTab(tab_mask, self.tr("setting_tab_mask"))

        # ---- Tags Filter (Character Tags) ----
        tab_filter = QWidget()
        filter_layout = QVBoxLayout(tab_filter)
        filter_layout.addWidget(QLabel(self.tr("setting_filter_title")))
        filter_layout.addWidget(QLabel(self.tr("setting_filter_info")))
        
        # f_form = QFormLayout()  <-- Remove FormLayout to stacking
        
        filter_layout.addWidget(QLabel(self.tr("setting_bl_words")))
        self.ed_bl_words = QPlainTextEdit()
        self.ed_bl_words.setPlainText(", ".join(self.cfg.get("char_tag_blacklist_words", [])))
        self.ed_bl_words.setMinimumHeight(120)
        filter_layout.addWidget(self.ed_bl_words)

        filter_layout.addWidget(QLabel(self.tr("setting_wl_words")))
        self.ed_wl_words = QPlainTextEdit()
        self.ed_wl_words.setPlainText(", ".join(self.cfg.get("char_tag_whitelist_words", [])))
        self.ed_wl_words.setMinimumHeight(80)
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
        a = max(0, min(255, a))
        fmt = (self.ed_mask_format.text().strip().lower() or DEFAULT_APP_SETTINGS["mask_default_format"]).strip(".")
        if fmt not in ("webp", "png"):
            fmt = DEFAULT_APP_SETTINGS["mask_default_format"]

        cfg["mask_default_alpha"] = a
        cfg["mask_default_format"] = fmt
        cfg["mask_batch_only_if_has_background_tag"] = self.chk_only_bg.isChecked()
        cfg["mask_batch_detect_text_enabled"] = self.chk_detect_text.isChecked()
        cfg["mask_delete_npz_on_move"] = self.chk_delete_npz.isChecked()

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

        self.nl_pages = []
        self.nl_page_index = 0
        self.nl_latest = ""

        self.batch_tagger_thread = None
        self.batch_llm_thread = None
        self.batch_unmask_thread = None
        self.batch_mask_text_thread = None

        self.init_ui()
        self.apply_theme()
        self.setup_shortcuts()
        self._hf_tokenizer = None

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
        self.img_info_label = QLabel("No Image Loaded")
        self.img_info_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        left_layout.addWidget(self.img_info_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self.btn_auto_tag.clicked.connect(self.run_tagger)
        tags_toolbar.addWidget(self.btn_auto_tag)

        self.btn_batch_tagger = QPushButton(self.tr("btn_batch_tagger"))
        self.btn_batch_tagger.clicked.connect(self.run_batch_tagger)
        tags_toolbar.addWidget(self.btn_batch_tagger)

        self.btn_batch_tagger_to_txt = QPushButton(self.tr("btn_batch_tagger_to_txt"))
        self.btn_batch_tagger_to_txt.clicked.connect(self.run_batch_tagger_to_txt)
        tags_toolbar.addWidget(self.btn_batch_tagger_to_txt)

        self.btn_add_custom_tag = QPushButton(self.tr("btn_add_tag"))
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
        self.btn_run_llm.clicked.connect(self.run_llm_generation)
        nl_toolbar.addWidget(self.btn_run_llm)

        # ✅ Batch 按鍵保留在上方
        self.btn_batch_llm = QPushButton(self.tr("btn_batch_llm"))
        self.btn_batch_llm.clicked.connect(self.run_batch_llm)
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

        self.btn_default_prompt = QPushButton(self.tr("btn_default_prompt"))
        self.btn_default_prompt.clicked.connect(self.use_default_prompt)
        nl_toolbar.addWidget(self.btn_default_prompt)

        self.btn_custom_prompt = QPushButton(self.tr("btn_custom_prompt"))
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
        bot_toolbar.addWidget(self.txt_token_label)
        bot_toolbar.addStretch(1)

        self.btn_find_replace = QPushButton(self.tr("btn_find_replace"))
        self.btn_find_replace.clicked.connect(self.open_find_replace)
        bot_toolbar.addWidget(self.btn_find_replace)

        self.btn_txt_undo = QPushButton(self.tr("btn_undo"))
        self.btn_txt_redo = QPushButton(self.tr("btn_redo"))
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
    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("msg_select_dir"))
        if dir_path:
            self.root_dir_path = dir_path
            self.image_files = []
            valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
            ignore_dirs = {"no_used", "unmask"}

            # ✅ 根目錄檔案
            try:
                for entry in os.scandir(dir_path):
                    if entry.is_file() and entry.name.lower().endswith(valid_exts):
                        if any(part.lower() in ignore_dirs for part in Path(entry.path).parts):
                            continue
                        self.image_files.append(entry.path)
            except Exception:
                pass

            # ✅ 第一級子資料夾（不往下遞迴）
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
            if self.image_files:
                self.current_index = 0
                self.load_image()
            else:
                QMessageBox.information(self, "Info", self.tr("msg_no_images"))

    def load_image(self):
        if 0 <= self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
            self.current_folder_path = str(Path(self.current_image_path).parent)

            self.img_info_label.setText(
                f"{self.current_index + 1} / {len(self.image_files)} : {os.path.basename(self.current_image_path)}"
            )

            pixmap = QPixmap(self.current_image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)

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

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

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
        
        # === 修改開始：強制游標邏輯 ===
        # 如果游標在位置 0 (開頭) 且文字不為空，或者編輯框當前沒有焦點
        # 我們假設使用者是想要 "Append" (附加) 到最後面，而不是插在開頭
        if (cursor.position() == 0 and len(text) > 0) or not edit.hasFocus():
            cursor.movePosition(QTextCursor.MoveOperation.End)
            edit.setTextCursor(cursor) # 更新編輯框的游標狀態
        # === 修改結束 ===

        pos = cursor.position()

        if not text.strip():
            # 如果原本是空的，直接取代
            new_text = token
            edit.blockSignals(True)
            edit.setPlainText(new_text)
            edit.blockSignals(False)
            
            # 設定游標到最後
            c = edit.textCursor()
            c.movePosition(QTextCursor.MoveOperation.End)
            edit.setTextCursor(c)
            edit.ensureCursorVisible() # 確保視窗捲動到游標處
            return

        before = text[:pos]
        after = text[pos:]

        prefix = ""
        suffix = ""

        before_strip = before.rstrip()
        after_strip = after.lstrip()

        # 智慧判斷逗號：如果是接在後面，前面補逗號
        if before_strip and not before_strip.endswith(","):
            prefix = ", "
        # 如果是插在中間，後面補逗號
        if after_strip and not after_strip.startswith(","):
            suffix = ", "

        inserted = before + prefix + token + suffix + after
        inserted = cleanup_csv_like_text(inserted)

        edit.blockSignals(True)
        edit.setPlainText(inserted)
        edit.blockSignals(False)

        # 計算新的游標位置：原本位置 + 前綴長度 + token長度
        # 這裡做一個優化：通常加完 Tag 後，使用者希望游標在該 Tag 的後面
        new_pos_target = len(before) + len(prefix) + len(token)
        
        # 如果我們原本就是在最後面加入，直接把游標移到全文字尾
        if pos >= len(text): 
             new_pos = len(inserted)
        else:
             new_pos = min(len(inserted), new_pos_target)

        c = edit.textCursor()
        c.setPosition(new_pos)
        edit.setTextCursor(c)
        edit.ensureCursorVisible() # 關鍵：確保視窗會捲動到游標所在位置

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

        self.llm_thread = LLMWorker(
            self.llm_base_url,
            self.api_key,
            self.model_name,
            self.llm_system_prompt,
            user_prompt,
            self.current_image_path,
            tags_text
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
        try:
            remover = Remover()
            old_path = self.current_image_path
            result = BatchUnmaskWorker.remove_background_to_webp(old_path, remover)
            if not result:
                return
            new_path, _ = result
            # 刪除對應 npz
            if self.settings.get("mask_delete_npz_on_move", True):
                delete_matching_npz(old_path)
            # 記錄 masked_background 到 JSON sidecar
            sidecar = load_image_sidecar(new_path)
            sidecar["masked_background"] = True
            save_image_sidecar(new_path, sidecar)
            self._replace_image_path_in_list(old_path, new_path)
            self.load_image()
            self.statusBar().showMessage("Unmask 完成", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Unmask", f"失敗: {e}")

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
            background_tag_checker=None # Force run for single image
        )
        self.batch_mask_text_thread.progress.connect(self.show_progress)
        self.batch_mask_text_thread.per_image.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.done.connect(lambda: self.on_batch_done("Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Mask Text", e))
        self.batch_mask_text_thread.start()

    def restore_current_image(self):
        if not self.current_image_path:
            return
        
        current_path = self.current_image_path
        # Check if original exists in "unmask" subfolder
        src_dir = os.path.dirname(current_path)
        unmask_dir = os.path.join(src_dir, "unmask")
        filename = os.path.basename(current_path)
        
        # The worker moves original to "unmask/filename"
        # BUT it might have been renamed with _unique_path if conflict.
        # We try to find the best match? Or just the exact name?
        # Usually exact name matches the original.
        
        original_restore_path = os.path.join(unmask_dir, filename)
        
        # If current file is .webp, original might be .jpg/.png
        # We need to find if there is a file in unmask that matches base name?
        # The worker logic:
        # if ext == .webp: moves original (if .webp) to unmask/.
        # if ext != .webp: moves original (if !.webp) to unmask/ AND saves .webp to current.
        
        # So we look for any file in unmask/ with same stem?
        stem = os.path.splitext(filename)[0]
        
        candidate = None
        if os.path.exists(unmask_dir):
            for f in os.listdir(unmask_dir):
                if os.path.splitext(f)[0] == stem:
                    candidate = os.path.join(unmask_dir, f)
                    break
        
        if not candidate:
            QMessageBox.information(self, "Restore", "找不到位於 unmask/ 資料夾的原檔備份")
            return
            
        try:
            # Move candidate back to current_path or replace current_path
            # If current_path is the webp result, we should remove it and put original back?
            # Or just overwrite?
            
            # Case 1: result is .webp, original was .jpg
            # We want to restore .jpg to current folder.
            # And Remove .webp? Yes, usually.
            
            # Destination: src_dir + candidate_filename
            dest_path = os.path.join(src_dir, os.path.basename(candidate))
            
            # If dest_path != current_path, we might have 2 files now.
            # We should probably remove current_path if it was generated.
            
            # Move back
            shutil.move(candidate, dest_path)
            
            # Update sidecar: remove masked flags
            sidecar = load_image_sidecar(dest_path)
            if "masked_background" in sidecar: del sidecar["masked_background"]
            if "masked_text" in sidecar: del sidecar["masked_text"]
            save_image_sidecar(dest_path, sidecar)
            
            # If we restored a file that has different name/ext than current_path,
            # we should update list.
            # AND if current_path was a generated webp, we should delete it?
            if os.path.abspath(dest_path) != os.path.abspath(current_path):
                 # Ask user or just delete?
                 # Assuming we want to swap back.
                 try:
                     os.remove(current_path)
                 except: pass
            
            self._replace_image_path_in_list(current_path, dest_path)
            self.load_image()
            self.statusBar().showMessage("已還原原檔", 3000)
            
        except Exception as e:
             QMessageBox.warning(self, "Restore", f"還原失敗: {e}")

    def run_batch_unmask_background(self):
        if not self.image_files:
            return
        if Remover is None:
            QMessageBox.warning(self, "Batch Unmask", "transparent_background.Remover not available")
            return

        targets = [p for p in self.image_files if self._tagger_has_background(p)]
        if not targets:
            QMessageBox.information(self, "Batch Unmask", "找不到 tagger 含 background 的圖片")
            return

        self.batch_unmask_thread = BatchUnmaskWorker(targets, self.settings)
        self.batch_unmask_thread.progress.connect(self.show_progress)
        self.batch_unmask_thread.per_image.connect(self.on_batch_unmask_per_image)
        self.batch_unmask_thread.done.connect(self.on_batch_unmask_done)
        self.batch_unmask_thread.error.connect(self.on_batch_error)
        self.batch_unmask_thread.start()

    def on_batch_unmask_per_image(self, old_path: str, new_path: str):
        self._replace_image_path_in_list(old_path, new_path)

    def on_batch_unmask_done(self):
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
        self.run_batch_tagger()

    def run_batch_llm_to_txt(self):
        if not self.image_files:
            return
            
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.run_batch_llm()

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
            sentences = [s.strip() for s in content.replace("\n", ". ").split(". ") if s.strip()]
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
            new_part = ". ".join(items)
            if items and not new_part.endswith("."): 
                new_part += "."
            if mode == "append" and existing_content:
                sep = " " if not existing_content.endswith(".") else " "
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

        self.batch_llm_thread = BatchLLMWorker(
            self.llm_base_url,
            self.api_key,
            self.model_name,
            self.llm_system_prompt,
            user_prompt,
            self.image_files,
            self.build_llm_tags_context_for_image
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
        try:
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

    def cancel_batch(self):
        self.statusBar().showMessage("正在中止...", 2000)
        if hasattr(self, 'batch_unmask_thread') and self.batch_unmask_thread.isRunning():
            self.batch_unmask_thread.stop()
        if hasattr(self, 'batch_mask_text_thread') and self.batch_mask_text_thread.isRunning():
            self.batch_mask_text_thread.stop()
        # Add logic for other batch threads if needed


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
        settings_action = QAction(self.tr("btn_settings"), self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        tools_menu = menubar.addMenu(self.tr("menu_tools"))
        
        unmask_action = QAction(self.tr("btn_unmask"), self)
        unmask_action.triggered.connect(self.unmask_current_image)
        tools_menu.addAction(unmask_action)

        mask_text_action = QAction(self.tr("btn_mask_text"), self)
        mask_text_action.triggered.connect(self.mask_text_current_image)
        tools_menu.addAction(mask_text_action)

        tools_menu.addSeparator()
        
        batch_unmask_action = QAction(self.tr("btn_batch_unmask"), self)
        batch_unmask_action.triggered.connect(self.run_batch_unmask_background)
        tools_menu.addAction(batch_unmask_action)

        batch_mask_text_action = QAction(self.tr("btn_batch_mask_text"), self)
        batch_mask_text_action.triggered.connect(self.run_batch_mask_text)
        tools_menu.addAction(batch_mask_text_action)

        tools_menu.addSeparator()

        restore_action = QAction(self.tr("btn_restore_original"), self) 
        restore_action.triggered.connect(self.restore_current_image)
        tools_menu.addAction(restore_action)

        stroke_action = QAction(self.tr("btn_stroke_eraser"), self)
        stroke_action.triggered.connect(self.open_stroke_eraser)
        tools_menu.addAction(stroke_action)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())