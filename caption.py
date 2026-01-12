# ============================================================
#  Caption 神器 - 索引 (INDEX)
# ============================================================
#
# [Ln 32-97]    Imports & 外部依賴
# [Ln 98-185]   Configuration & Globals (含 Batch to txt 與 Char Tag 過濾設定)
# [Ln 187-255]  Settings Helpers (load/save/coerce 函式)
# [Ln 257-410]  Utils: delete_matching_npz、JSON sidecar、checkerboard
# [Ln 412-580]  Utils / Parsing / Tag Logic (含 is_basic_character_tag)
# [Ln 582-840]  Workers (TaggerWorker, LLMWorker, BatchTaggerWorker, BatchLLMWorker)
# [Ln 842-1000] BatchMaskTextWorker (OCR 批次遮罩)
# [Ln 1002-1130] BatchUnmaskWorker (批次去背)
# [Ln 1132-1270] StrokeCanvas & StrokeEraseDialog (手繪橡皮擦)
# [Ln 1272-1530] UI Components (TagButton 紅框、TagFlowWidget、Advanced Find/Replace)
# [Ln 1532-1780] SettingsDialog (Text 分頁新增 Batch 選項 + Tags Filter 分頁)
# [Ln 1782-1850] MainWindow.__init__ (主視窗初始化)
# [Ln 1852-2150] MainWindow.init_ui (新增 Batch to txt 按鈕)
# [Ln 2152-2200] MainWindow 快捷鍵 & 滾輪事件
# [Ln 2202-2370] MainWindow 檔案/圖片載入 & on_text_changed
# [Ln 2372-2440] MainWindow Token 計數
# [Ln 2442-2610] MainWindow TAGS/NL (JSON sidecar 整合)
# [Ln 2612-2700] MainWindow NL Paging
# [Ln 2702-2850] MainWindow Tag 插入/移除 & Tagger
# [Ln 2852-2990] MainWindow LLM 生成
# [Ln 2992-3080] MainWindow Tools: Unmask / Stroke Eraser
# [Ln 3082-3220] MainWindow Batch Tagger / LLM to txt (含寫入與過濾邏輯)
# [Ln 3222-3235] MainWindow Settings 儲存 & main 入口
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
    QScrollArea, QLineEdit, QDialog, QFormLayout,
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
from imgutils.tagging import get_wd14_tags, tags_to_text

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

    # Character Tags Filter (based on imgutils)
    "char_tag_whitelist_suffixes": ['anal_hair', 'anal_tail', 'arm_behind_head', 'arm_hair', 'arm_under_breasts', 'arms_behind_head', 'bird_on_head', 'blood_in_hair', 'breasts_on_glass', 'breasts_on_head', 'cat_on_head', 'closed_eyes', 'clothed_female_nude_female', 'clothed_female_nude_male', 'clothed_male_nude_female', 'clothes_between_breasts', 'cream_on_face', 'drying_hair', 'empty_eyes', 'face_to_breasts', 'facial', 'food_on_face', 'food_on_head', 'game_boy', "grabbing_another's_hair", 'grabbing_own_breast', 'gun_to_head', 'half-closed_eyes', 'head_between_breasts', 'heart_in_eye', 'multiple_boys', 'multiple_girls', 'object_on_breast', 'object_on_head', 'paint_splatter_on_face', 'parted_lips', 'penis_on_face', 'person_on_head', 'pokemon_on_head', 'pubic_hair', 'rabbit_on_head', 'rice_on_face', 'severed_head', 'star_in_eye', 'sticker_on_face', 'tentacles_on_male', 'tying_hair'],
    "char_tag_whitelist_prefixes": ['holding', 'hand on', 'hands on', 'hand to', 'hands to', 'hand in', 'hands in', 'hand over', 'hands over', 'futa with', 'futa on', 'cum on', 'covering', 'adjusting', 'rubbing', 'sitting', 'shading', 'playing', 'cutting'],
    "char_tag_whitelist_words": ['drill'],
    "char_tag_blacklist_suffixes": ['eyes', 'skin', 'hair', 'bun', 'bangs', 'cut', 'sidelocks', 'twintails', 'braid', 'braids', 'afro', 'ahoge', 'drill', 'drills', 'bald', 'dreadlocks', 'side up', 'ponytail', 'updo', 'beard', 'mustache', 'pointy ears', 'ear', 'horn', 'tail', 'wing', 'ornament', 'hairband', 'pupil', 'bow', 'eyewear', 'headwear', 'ribbon', 'crown', 'cap', 'hat', 'hairclip', 'breast', 'mole', 'halo', 'earrings', 'animal ear fluff', 'hair flower', 'glasses', 'fang', 'female', 'girl', 'boy', 'male', 'beret', 'heterochromia', 'headdress', 'headgear', 'eyepatch', 'headphones', 'eyebrows', 'eyelashes', 'sunglasses', 'hair intakes', 'scrunchie', 'ear_piercing', 'head', 'on face', 'on head', 'on hair', 'headband', 'hair rings', 'under_mouth', 'freckles', 'lip', 'eyeliner', 'eyeshadow', 'tassel', 'over one eye', 'drill', 'drill hair'],
    "char_tag_blacklist_prefixes": ['hair over', 'hair between', 'facial'],

    # Mask / batch mask text
    "mask_default_alpha": 0,  # 0-255, 0 = fully transparent
    "mask_default_format": "webp",  # webp | png
    "mask_batch_only_if_has_background_tag": False,
    "mask_batch_detect_text_enabled": True,  # if off, never call detect_text_with_ocr
    "mask_delete_npz_on_move": True,         # 移動舊圖時刪除對應 npz
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
    return get_wd14_tags(img_pil, **use)


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
                        translations[row[0].strip()] = row[1].strip()
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


def is_basic_character_tag(tag: str, cfg: dict) -> bool:
    """
    判定一個 tag 是否為特徵標籤 (Basic Character Tag)。
    邏輯：(匹配黑名單前綴/呼綴) AND (不匹配白名單單字/前綴/後綴)。
    """
    if not tag:
        return False
    
    # 正規化標籤以進行匹配
    t = tag.strip().lower().replace("_", " ")
    
    # 1. 檢查白名單 (優先)
    wl_words = cfg.get("char_tag_whitelist_words", [])
    if t in [w.strip().lower().replace("_", " ") for w in wl_words]:
        return False
        
    wl_prefixes = cfg.get("char_tag_whitelist_prefixes", [])
    for p in wl_prefixes:
        pre = p.strip().lower().replace("_", " ")
        if t.startswith(pre + " ") or t == pre:
            return False
            
    wl_suffixes = cfg.get("char_tag_whitelist_suffixes", [])
    for s in wl_suffixes:
        suf = s.strip().lower().replace("_", " ")
        if t.endswith(" " + suf) or t == suf:
            return False
            
    # 2. 檢查黑名單
    bl_prefixes = cfg.get("char_tag_blacklist_prefixes", [])
    for p in bl_prefixes:
        pre = p.strip().lower().replace("_", " ")
        if t.startswith(pre + " ") or t == pre:
            return True
            
    bl_suffixes = cfg.get("char_tag_blacklist_suffixes", [])
    for s in bl_suffixes:
        suf = s.strip().lower().replace("_", " ")
        if t.endswith(" " + suf) or t == suf:
            return True
            
    return False


def normalize_for_match(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace(", ", "").replace(",", "")
    t = t.strip()
    t = t.rstrip(".")
    return t.strip()


def cleanup_csv_like_text(text: str) -> str:
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    return ", ".join(parts)


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
        border_color = "red" if self.is_character else "#ccc"
        border_width = "2px" if self.is_character else "1px"
        checked_border = "red" if self.is_character else "#0055aa"
        
        self.setStyleSheet(f"""
            QPushButton {{
                border: {border_width} solid {border_color};
                border-radius: 4px;
                background-color: #f0f0f0;
            }}
            QPushButton:checked {{
                background-color: #d0e8ff;
                border: {border_width} solid {checked_border};
            }}
            QPushButton:hover {{
                border: {border_width} solid #999;
            }}
        """)

    def update_label(self):
        safe_text = str(self.raw_text).replace("<", "&lt;").replace(">", "&gt;")
        content = f"<span style='font-size:13px; font-weight:bold; color:#000;'>{safe_text}</span>"

        if self.translation:
            safe_trans = str(self.translation).replace("<", "&lt;").replace(">", "&gt;")
            content += f"<br><span style='color:#666; font-size:11px;'>{safe_trans}</span>"

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
                trans = self.translations_csv.get(text)

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
        self.setWindowTitle("Settings")
        self.setMinimumWidth(640)
        self.cfg = dict(cfg or {})

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

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

        llm_layout.addWidget(QLabel("System Prompt:"))
        self.ed_system_prompt = QPlainTextEdit()
        self.ed_system_prompt.setPlainText(str(self.cfg.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT)))
        self.ed_system_prompt.setMinimumHeight(90)
        llm_layout.addWidget(self.ed_system_prompt)

        llm_layout.addWidget(QLabel("Default Prompt:"))
        self.ed_user_template = QPlainTextEdit()
        self.ed_user_template.setPlainText(str(self.cfg.get("llm_user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)))
        self.ed_user_template.setMinimumHeight(200)
        llm_layout.addWidget(self.ed_user_template, 1)


        llm_layout.addWidget(QLabel("Custom Prompt:"))
        self.ed_custom_template = QPlainTextEdit()
        self.ed_custom_template.setPlainText(str(self.cfg.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
        self.ed_custom_template.setMinimumHeight(200)
        llm_layout.addWidget(self.ed_custom_template, 1)


        llm_layout.addWidget(QLabel("預設 Custom Tags（逗號或每行一個）:"))
        self.ed_default_custom_tags = QPlainTextEdit()
        tags = self.cfg.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS))
        if isinstance(tags, list):
            self.ed_default_custom_tags.setPlainText("\n".join([str(t) for t in tags]))
        else:
            self.ed_default_custom_tags.setPlainText(str(tags))
        self.ed_default_custom_tags.setMinimumHeight(80)
        llm_layout.addWidget(self.ed_default_custom_tags)

        self.tabs.addTab(tab_llm, "LLM")

        # ---- Tagger ----
        tab_tagger = QWidget()
        tagger_layout = QVBoxLayout(tab_tagger)
        form2 = QFormLayout()
        self.ed_tagger_model = QLineEdit(str(self.cfg.get("tagger_model", "EVA02_Large")))
        self.ed_general_threshold = QLineEdit(str(self.cfg.get("general_threshold", 0.2)))
        self.chk_general_mcut = QCheckBox("general_mcut_enabled")
        self.chk_general_mcut.setChecked(bool(self.cfg.get("general_mcut_enabled", False)))

        self.ed_character_threshold = QLineEdit(str(self.cfg.get("character_threshold", 0.85)))
        self.chk_character_mcut = QCheckBox("character_mcut_enabled")
        self.chk_character_mcut.setChecked(bool(self.cfg.get("character_mcut_enabled", True)))

        self.chk_drop_overlap = QCheckBox("drop_overlap")
        self.chk_drop_overlap.setChecked(bool(self.cfg.get("drop_overlap", True)))

        form2.addRow("Tagger 預設 Model:", self.ed_tagger_model)
        form2.addRow("general_threshold:", self.ed_general_threshold)
        form2.addRow("", self.chk_general_mcut)
        form2.addRow("character_threshold:", self.ed_character_threshold)
        form2.addRow("", self.chk_character_mcut)
        form2.addRow("", self.chk_drop_overlap)

        tagger_layout.addLayout(form2)
        tagger_layout.addStretch(1)
        self.tabs.addTab(tab_tagger, "Tagger")

        # ---- Text ----
        tab_text = QWidget()
        text_layout = QVBoxLayout(tab_text)
        self.chk_force_lower = QCheckBox("英文文字是否一律小寫（LLM 英文句 / tags 正規化）")
        self.chk_force_lower.setChecked(bool(self.cfg.get("english_force_lowercase", True)))
        text_layout.addWidget(self.chk_force_lower)

        self.chk_auto_remove_empty = QCheckBox("自動移除空行（text 裡面有空行或全空白自動移除）")
        self.chk_auto_remove_empty.setChecked(bool(self.cfg.get("text_auto_remove_empty_lines", True)))
        text_layout.addWidget(self.chk_auto_remove_empty)

        self.chk_auto_format = QCheckBox("自動格式化（插入時用 , 分割去除空白，用 ', ' 重組）")
        self.chk_auto_format.setChecked(bool(self.cfg.get("text_auto_format", True)))
        text_layout.addWidget(self.chk_auto_format)

        self.chk_auto_save = QCheckBox("自動儲存 txt（有改動時自動儲存）")
        self.chk_auto_save.setChecked(bool(self.cfg.get("text_auto_save", True)))
        text_layout.addWidget(self.chk_auto_save)

        # Batch to txt options
        text_layout.addWidget(self.make_hline())
        text_layout.addWidget(QLabel("<b>Batch to txt 設定</b>"))
        
        mode_grp = QGroupBox("寫入模式")
        mode_lay = QHBoxLayout()
        self.rb_batch_append = QRadioButton("附加到句尾")
        self.rb_batch_overwrite = QRadioButton("覆寫原檔")
        if self.cfg.get("batch_to_txt_mode", "append") == "overwrite":
            self.rb_batch_overwrite.setChecked(True)
        else:
            self.rb_batch_append.setChecked(True)
        mode_lay.addWidget(self.rb_batch_append)
        mode_lay.addWidget(self.rb_batch_overwrite)
        mode_grp.setLayout(mode_lay)
        text_layout.addWidget(mode_grp)
        
        self.chk_folder_trigger = QCheckBox("將資料夾名作為觸發詞加到句首")
        self.chk_folder_trigger.setChecked(bool(self.cfg.get("batch_to_txt_folder_trigger", False)))
        text_layout.addWidget(self.chk_folder_trigger)

        text_layout.addStretch(1)
        self.tabs.addTab(tab_text, "Text")

        # ---- Mask ----
        tab_mask = QWidget()
        mask_layout = QVBoxLayout(tab_mask)
        form3 = QFormLayout()

        self.ed_mask_alpha = QLineEdit(str(self.cfg.get("mask_default_alpha", 0)))
        self.ed_mask_format = QLineEdit(str(self.cfg.get("mask_default_format", "webp")))
        form3.addRow("mask 預設透明度 alpha (0-255):", self.ed_mask_alpha)
        form3.addRow("mask 預設轉換格式 (webp/png):", self.ed_mask_format)

        self.chk_only_bg = QCheckBox("mask batch 是否只針對有 background 的 tag")
        self.chk_only_bg.setChecked(bool(self.cfg.get("mask_batch_only_if_has_background_tag", False)))

        self.chk_detect_text = QCheckBox("batch 自動 mask 有文字的地方（OCR）")
        self.chk_detect_text.setChecked(bool(self.cfg.get("mask_batch_detect_text_enabled", True)))

        mask_layout.addLayout(form3)
        mask_layout.addWidget(self.chk_only_bg)
        mask_layout.addWidget(self.chk_detect_text)

        self.chk_delete_npz = QCheckBox("移動舊圖時刪除對應 npz（含完整圖檔名的 npz 會被刪除）")
        self.chk_delete_npz.setChecked(bool(self.cfg.get("mask_delete_npz_on_move", True)))
        mask_layout.addWidget(self.chk_delete_npz)

        hint = QLabel("OCR 依賴 imgutils.ocr.detect_text_with_ocr；未安裝時會自動略過。")
        hint.setStyleSheet("color:#666;")
        mask_layout.addWidget(hint)
        mask_layout.addStretch(1)
        self.tabs.addTab(tab_mask, "Mask")

        # ---- Tags Filter (Character Tags) ----
        tab_filter = QWidget()
        filter_layout = QVBoxLayout(tab_filter)
        filter_layout.addWidget(QLabel("<b>特徵標籤 (Character Tags) 過濾設定</b>"))
        filter_layout.addWidget(QLabel("符合黑名單且不符合白名單的標籤將被標記為紅框，且在 Batch to txt 時可選擇刪除。"))
        
        f_form = QFormLayout()
        self.ed_wl_words = QLineEdit(", ".join(self.cfg.get("char_tag_whitelist_words", [])))
        self.ed_wl_prefixes = QLineEdit(", ".join(self.cfg.get("char_tag_whitelist_prefixes", [])))
        self.ed_wl_suffixes = QLineEdit(", ".join(self.cfg.get("char_tag_whitelist_suffixes", [])))
        self.ed_bl_prefixes = QLineEdit(", ".join(self.cfg.get("char_tag_blacklist_prefixes", [])))
        self.ed_bl_suffixes = QPlainTextEdit(", ".join(self.cfg.get("char_tag_blacklist_suffixes", [])))
        self.ed_bl_suffixes.setMaximumHeight(100)

        f_form.addRow("白名單單字 (Whitelist Words):", self.ed_wl_words)
        f_form.addRow("白名單前綴 (Whitelist Prefixes):", self.ed_wl_prefixes)
        f_form.addRow("白名單後綴 (Whitelist Suffixes):", self.ed_wl_suffixes)
        f_form.addRow("黑名單前綴 (Blacklist Prefixes):", self.ed_bl_prefixes)
        filter_layout.addLayout(f_form)
        
        filter_layout.addWidget(QLabel("黑名單後綴 (Blacklist Suffixes):"))
        filter_layout.addWidget(self.ed_bl_suffixes)
        filter_layout.addStretch(1)
        
        self.tabs.addTab(tab_filter, "Tags Filter")

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QPushButton("Save")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

    def _parse_tags(self, s: str):
        raw = (s or "").strip()
        if not raw:
            return []
        if "\n" in raw:
            parts = [x.strip() for x in raw.splitlines() if x.strip()]
        else:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
        return [p.replace("_", " ").strip() for p in parts if p.strip()]

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
        cfg["char_tag_whitelist_words"] = [x.strip() for x in self.ed_wl_words.text().split(",") if x.strip()]
        cfg["char_tag_whitelist_prefixes"] = [x.strip() for x in self.ed_wl_prefixes.text().split(",") if x.strip()]
        cfg["char_tag_whitelist_suffixes"] = [x.strip() for x in self.ed_wl_suffixes.text().split(",") if x.strip()]
        cfg["char_tag_blacklist_prefixes"] = [x.strip() for x in self.ed_bl_prefixes.text().split(",") if x.strip()]
        
        raw_bl_suf = self.ed_bl_suffixes.toPlainText().replace("\n", ",")
        cfg["char_tag_blacklist_suffixes"] = [x.strip() for x in raw_bl_suf.split(",") if x.strip()]

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
        self.setup_shortcuts()
        self._hf_tokenizer = None

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

        self.btn_auto_tag = QPushButton("Auto Tag (WD14)")
        self.btn_auto_tag.clicked.connect(self.run_tagger)
        tags_toolbar.addWidget(self.btn_auto_tag)

        self.btn_batch_tagger = QPushButton("Batch Tagger")
        self.btn_batch_tagger.clicked.connect(self.run_batch_tagger)
        tags_toolbar.addWidget(self.btn_batch_tagger)

        self.btn_batch_tagger_to_txt = QPushButton("Batch Tagger to txt")
        self.btn_batch_tagger_to_txt.clicked.connect(self.run_batch_tagger_to_txt)
        tags_toolbar.addWidget(self.btn_batch_tagger_to_txt)

        self.btn_add_custom_tag = QPushButton("Add Tag")
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

        self.sec1_title = QLabel("<b>Folder / Meta</b>")
        self.tags_scroll_layout.addWidget(self.sec1_title)
        self.flow_top = TagFlowWidget(use_scroll=False)
        self.flow_top.set_translations_csv(self.translations_csv)
        self.flow_top.tag_clicked.connect(self.on_tag_button_toggled)
        self.tags_scroll_layout.addWidget(self.flow_top)

        self.tags_scroll_layout.addWidget(self.make_hline())

        self.sec2_title = QLabel("<b>Custom (per folder)</b>")
        self.tags_scroll_layout.addWidget(self.sec2_title)
        self.flow_custom = TagFlowWidget(use_scroll=False)
        self.flow_custom.set_translations_csv(self.translations_csv)
        self.flow_custom.tag_clicked.connect(self.on_tag_button_toggled)
        self.tags_scroll_layout.addWidget(self.flow_custom)

        self.tags_scroll_layout.addWidget(self.make_hline())

        self.sec3_title = QLabel("<b>Tagger (WD14)</b>")
        self.tags_scroll_layout.addWidget(self.sec3_title)
        self.flow_tagger = TagFlowWidget(use_scroll=False)
        self.flow_tagger.set_translations_csv(self.translations_csv)
        self.flow_tagger.tag_clicked.connect(self.on_tag_button_toggled)
        self.tags_scroll_layout.addWidget(self.flow_tagger)

        self.tags_scroll_layout.addStretch(1)
        self.tags_scroll.setWidget(tags_scroll_container)

        self.tabs.addTab(tags_tab, "TAGS")

        # ---- NL Tab ----
        nl_tab = QWidget()
        nl_layout = QVBoxLayout(nl_tab)
        nl_layout.setContentsMargins(5, 5, 5, 5)

        nl_toolbar = QHBoxLayout()
        nl_label = QLabel("<b>NL</b>")
        nl_toolbar.addWidget(nl_label)

        self.btn_run_llm = QPushButton("Run LLM")
        self.btn_run_llm.clicked.connect(self.run_llm_generation)
        nl_toolbar.addWidget(self.btn_run_llm)

        # ✅ Batch 按鍵保留在上方
        self.btn_batch_llm = QPushButton("Batch LLM")
        self.btn_batch_llm.clicked.connect(self.run_batch_llm)
        nl_toolbar.addWidget(self.btn_batch_llm)

        self.btn_batch_llm_to_txt = QPushButton("Batch LLM to txt")
        self.btn_batch_llm_to_txt.clicked.connect(self.run_batch_llm_to_txt)
        nl_toolbar.addWidget(self.btn_batch_llm_to_txt)

        self.btn_prev_nl = QPushButton("Prev")
        self.btn_prev_nl.clicked.connect(self.prev_nl_page)
        nl_toolbar.addWidget(self.btn_prev_nl)

        self.btn_next_nl = QPushButton("Next")
        self.btn_next_nl.clicked.connect(self.next_nl_page)
        nl_toolbar.addWidget(self.btn_next_nl)

        self.btn_default_prompt = QPushButton("Default Prompt")
        self.btn_default_prompt.clicked.connect(self.use_default_prompt)
        nl_toolbar.addWidget(self.btn_default_prompt)

        self.btn_custom_prompt = QPushButton("Custom Prompt")
        self.btn_custom_prompt.clicked.connect(self.use_custom_prompt)
        nl_toolbar.addWidget(self.btn_custom_prompt)
        self.nl_page_label = QLabel("Page 0/0")
        nl_toolbar.addWidget(self.nl_page_label)

        nl_toolbar.addStretch(1)
        nl_layout.addLayout(nl_toolbar)

        # ✅ RESULT 最上面
        self.nl_result_title = QLabel("<b>LLM Result</b>")
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

        self.tabs.addTab(nl_tab, "NL")

        # ---- Bottom: txt ----
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)
        bot_layout.setContentsMargins(5, 5, 5, 5)

        bot_toolbar = QHBoxLayout()
        bot_label = QLabel("<b>Actual Content (.txt)</b>")
        bot_toolbar.addWidget(bot_label)
        bot_toolbar.addSpacing(10)
        self.txt_token_label = QLabel("Tokens: 0")
        bot_toolbar.addWidget(self.txt_token_label)
        bot_toolbar.addStretch(1)

        self.btn_find_replace = QPushButton("Find/Replace")
        self.btn_find_replace.clicked.connect(self.open_find_replace)
        bot_toolbar.addWidget(self.btn_find_replace)

        self.btn_txt_undo = QPushButton("Undo Txt")
        self.btn_txt_redo = QPushButton("Redo Txt")
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
        self.progress_bar.setMaximumWidth(380)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Directory", self)
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        tools_menu = menubar.addMenu("Tools")
        unmask_action = QAction("Remove Background (Unmask)", self)
        unmask_action.triggered.connect(self.unmask_current_image)
        tools_menu.addAction(unmask_action)

        batch_unmask_action = QAction("Batch Remove Background (*background)", self)
        batch_unmask_action.triggered.connect(self.run_batch_unmask_background)
        tools_menu.addAction(batch_unmask_action)

        mask_text_action = QAction("Batch Mask Text (OCR)", self)
        mask_text_action.triggered.connect(self.run_batch_mask_text)
        tools_menu.addAction(mask_text_action)

        stroke_action = QAction("Stroke Eraser (Transparent)", self)
        stroke_action.triggered.connect(self.open_stroke_eraser)
        tools_menu.addAction(stroke_action)

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
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
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
                QMessageBox.information(self, "Info", "No images found.")

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
            self, "Delete", "Move to 'no_used'?",
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
                
                # === 修改重點開始 ===
                
                # 設定顏色：超過 225 才變紅，否則全黑
                text_color = "red" if count > 225 else "black"
                self.txt_token_label.setStyleSheet(f"color: {text_color}")
                
                # 設定文字：只顯示 "Tokens: 數字"
                self.txt_token_label.setText(f"Tokens: {count}")
                
                # === 修改重點結束 ===

            except Exception as e:
                print(f"Token count error: {e}")
                self.txt_token_label.setText("Tokens: Err")

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
        tag, ok = QInputDialog.getText(self, "Add Tag", "新增 tag：")
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
                self.nl_page_label.setText("Page 0/0")
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
        if not old_path or not new_path or old_path == new_path:
            return
        for i, p in enumerate(self.image_files):
            if os.path.abspath(p) == os.path.abspath(old_path):
                self.image_files[i] = new_path
                if i == self.current_index:
                    self.current_image_path = new_path
                break

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
        self.progress_bar.setFormat(f"{current}/{total}  {name}")

    def hide_progress(self):
        self.progress_bar.setVisible(False)
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

    def run_batch_mask_text(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        self.batch_mask_text_thread = BatchMaskTextWorker(
            self.image_files,
            self.settings,
            background_tag_checker=self._image_has_background_tag
        )
        self.batch_mask_text_thread.progress.connect(lambda i, t, name: self.show_progress(i, t, f"MaskText: {name}"))
        self.batch_mask_text_thread.per_image.connect(lambda oldp, newp: None)
        self.batch_mask_text_thread.done.connect(lambda: self.on_batch_done("Batch Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Batch Mask Text", e))
        self.batch_mask_text_thread.start()


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
            self.llm_base_url = str(new_cfg.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
            self.api_key = str(new_cfg.get("llm_api_key", ""))
            self.model_name = str(new_cfg.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
            self.llm_system_prompt = str(new_cfg.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
            self.default_user_prompt_template = str(new_cfg.get("llm_user_prompt_template", DEFAULT_APP_SETTINGS["llm_user_prompt_template"]))
            self.custom_prompt_template = str(new_cfg.get("llm_custom_prompt_template", DEFAULT_APP_SETTINGS.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
            self.default_custom_tags_global = list(new_cfg.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
            self.english_force_lowercase = bool(new_cfg.get("english_force_lowercase", True))

            # update current UI defaults
            if hasattr(self, "prompt_edit") and self.prompt_edit:
                self.prompt_edit.setPlainText(self.default_user_prompt_template)

            # refresh custom tags default file creation only affects new folders; existing folders keep their .custom_tags.json
            self.statusBar().showMessage("Settings 已儲存", 4000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())