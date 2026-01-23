# -*- coding: utf-8 -*-
"""
應用程式設定模組

提供設定的載入、儲存和輔助函數。
"""
import os
import json
import shutil
from pathlib import Path


# --------------------------
# Default Prompts
# --------------------------
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
APP_SETTINGS_FILE = os.path.join(os.getcwd(), "app_settings.json")

DEFAULT_APP_SETTINGS = {
    # LLM
    "llm_provider": "vlm_openrouter_api",  # Worker Name (Category=LLM)
    "llm_base_url": "https://openrouter.ai/api/v1",
    "llm_api_key": os.getenv("OPENROUTER_API_KEY", "<OPENROUTER_API_KEY>"),
    "llm_model": "mistralai/mistral-large-2512",
    "llm_system_prompt": DEFAULT_SYSTEM_PROMPT,
    "llm_user_prompt_template": DEFAULT_USER_PROMPT_TEMPLATE,
    "llm_custom_prompt_template": DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    "default_custom_tags": list(DEFAULT_CUSTOM_TAGS),
    "llm_skip_nsfw_on_batch": False,
    "llm_use_gray_mask": True,
    "last_open_dir": "",

    # Worker Selection
    "tagger_worker": "tagger_imgutils_generic",
    
    # Tagger (WD14)
    "tagger_model": "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full",
    "general_threshold": 0.25,
    "general_mcut_enabled": False,
    "character_threshold": 0.85,
    "character_mcut_enabled": True,
    "drop_overlap": True,

    # Text / normalization
    "english_force_lowercase": True,
    "text_auto_remove_empty_lines": True,
    "text_auto_format": True,
    "text_auto_save": True,
    "batch_to_txt_mode": "append",
    "batch_to_txt_folder_trigger": False,

    # LLM Resolution (Advanced)
    "llm_max_image_dimension": 1024,

    # Character Tags Filter (simple word matching)
    "char_tag_blacklist_words": [
        "hair", "eyes", "skin", "bun", "bangs", "sidelocks", "twintails", 
        "braid", "ponytail", "beard", "mustache", "ear", "horn", "tail", 
        "wing", "breast", "mole", "halo", "glasses", "fang", "heterochromia", 
        "headband", "freckles", "lip", "eyebrows", "eyelashes"
    ],
    "char_tag_whitelist_words": [
        "holding", "hand", "sitting", "covering", "playing", "background", "looking"
    ],

    # Mask / batch mask text
    "unmask_worker": "mask_transparent_background_local",
    "mask_text_worker": "mask_text_local",
    "detect_text_worker": "detect_imgutils_ocr_local",
    
    "mask_remover_mode": "base-nightly",
    "mask_default_alpha": 64,
    "mask_default_format": "webp",
    "mask_reverse": False,
    "mask_save_map_file": False,
    "mask_only_output_map": False,
    "mask_batch_only_if_has_background_tag": True,
    "mask_batch_detect_text_enabled": True,
    "mask_delete_npz_on_move": True,
    
    "mask_padding": 1,
    "mask_blur_radius": 3,
    
    # Advanced Mask Post-Processing
    # Background (Unmask) settings
    "mask_bg_shrink_size": 1,
    "mask_bg_blur_radius": 3,
    "mask_bg_min_alpha": 0,
    
    # Text (Mask Text) settings
    "mask_text_shrink_size": 1,
    "mask_text_blur_radius": 3,
    "mask_text_min_alpha": 0,
    
    # Batch Mask Logic
    "mask_batch_skip_once_processed": True,
    "mask_batch_min_foreground_ratio": 0.3,
    "mask_batch_max_foreground_ratio": 0.8,
    "mask_batch_skip_if_scenery_tag": True,

    # Advanced OCR Settings
    "mask_ocr_max_candidates": 300,
    "mask_ocr_heat_threshold": 0.2,
    "mask_ocr_box_threshold": 0.6,
    "mask_ocr_unclip_ratio": 2.3,
    "mask_text_alpha": 10,

    # UI / Theme
    "ui_language": "zh_tw",
    "ui_theme": "light",
}


def load_app_settings() -> dict:
    """載入應用程式設定"""
    cfg = dict(DEFAULT_APP_SETTINGS)
    try:
        if os.path.exists(APP_SETTINGS_FILE):
            try:
                with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    cfg.update(data)
            except json.JSONDecodeError as e:
                print(f"[Settings] JSON decode failed: {e}")
                # Backup corrupted file
                backup_path = APP_SETTINGS_FILE + ".bak"
                try:
                    shutil.copy2(APP_SETTINGS_FILE, backup_path)
                    print(f"[Settings] Corrupted settings backed up to {backup_path}")
                except Exception:
                    pass
                # Return defaults (which we already have in cfg)
            except Exception as e:
                print(f"[Settings] Unexpected error loading settings: {e}")
    except Exception as e:
        print(f"[Settings] load failed: {e}")
    return cfg


def save_app_settings(cfg: dict) -> bool:
    """儲存應用程式設定"""
    try:
        safe = dict(DEFAULT_APP_SETTINGS)
        safe.update(cfg or {})
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Settings] save failed: {e}")
        return False


# --------------------------
# Type Coercion Helpers
# --------------------------
def _coerce_bool(v, default=False):
    """將值轉換為布林值"""
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
    """將值轉換為浮點數"""
    try:
        return float(v)
    except Exception:
        return float(default)


def _coerce_int(v, default=0):
    """將值轉換為整數"""
    try:
        return int(v)
    except Exception:
        return int(default)
