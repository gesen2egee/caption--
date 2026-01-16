
import json
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict, Any

from .utils import load_image_sidecar, save_image_sidecar, image_sidecar_json_path
from .const import DEFAULT_APP_SETTINGS

APP_SETTINGS_FILE = os.path.join(str(Path.home()), ".ai_captioning_settings.json")

def load_app_settings() -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_APP_SETTINGS)
    if os.path.exists(APP_SETTINGS_FILE):
        try:
            with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    cfg.update(saved)
        except Exception:
            pass
    return cfg

def save_app_settings(cfg: Dict[str, Any]):
    try:
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ==========================================
#  Settings Wrapper
# ==========================================

class AppSettings:
    """
    應用程式設定的強型別封裝。
    提供方便的屬性存取與預設值管理。
    """
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    # --- LLM Shortcuts ---
    @property
    def llm_base_url(self) -> str:
        return self._data.get("llm_base_url", "")
    
    @property
    def llm_api_key(self) -> str:
        return self._data.get("llm_api_key", "")
    
    @property
    def llm_model(self) -> str:
        return self._data.get("llm_model", "")
    
    @property
    def system_prompt(self) -> str:
        return self._data.get("llm_system_prompt", "")
    
    @property
    def user_prompt_template(self) -> str:
        # 取得目前啟用的 Template Key
        key = self._data.get("llm_active_seed_template", "標準完整模式 (Standard)")
        # 從 Dict 中取得內容，若找不到則回傳預設（避免 Key Error）
        templates = self._data.get("llm_prompt_templates", {})
        return templates.get(key, "")

    @property
    def prompt_templates(self) -> Dict[str, str]:
        return self._data.get("llm_prompt_templates", {})
    
    @property
    def active_template_key(self) -> str:
        return self._data.get("llm_active_seed_template", "")

    def set_active_template(self, key: str):
        if key in self.prompt_templates:
            self._data["llm_active_seed_template"] = key

    def add_template(self, name: str, content: str):
        if "llm_prompt_templates" not in self._data:
            self._data["llm_prompt_templates"] = {}
        self._data["llm_prompt_templates"][name] = content

    def remove_template(self, name: str):
        if "llm_prompt_templates" in self._data:
            self._data["llm_prompt_templates"].pop(name, None)


    # --- Tagger Shortcuts ---
    @property
    def tagger_threshold(self) -> float:
        return float(self._data.get("general_threshold", 0.35))
    
    @property
    def char_threshold(self) -> float:
        return float(self._data.get("character_threshold", 0.85))

    # --- Text Shortcuts ---
    @property
    def auto_save(self) -> bool:
        return self._data.get("text_auto_save", True)

# ==========================================
#  Image Context (The star of the show)
# ==========================================

@dataclass
class ImageContext:
    """
    代表專案中單張圖片的所有上下文資訊。
    封裝了路徑、Sidecar 資料、圖片物件快取與備份邏輯。
    """
    path: str
    _sidecar: Optional[Dict[str, Any]] = None
    _image_cache: Optional[Image.Image] = None
    
    def __post_init__(self):
        # 確保路徑是絕對路徑
        self.path = os.path.abspath(self.path)

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)
    
    @property
    def dir_path(self) -> str:
        return os.path.dirname(self.path)

    @property
    def txt_path(self) -> str:
        return os.path.splitext(self.path)[0] + ".txt"
    
    @property
    def sidecar_path(self) -> str:
        return image_sidecar_json_path(self.path)

    @property
    def sidecar(self) -> Dict[str, Any]:
        """Lazy load sidecar data"""
        if self._sidecar is None:
            self._sidecar = load_image_sidecar(self.path)
        return self._sidecar
    
    def save_sidecar(self):
        """Save current sidecar data to disk"""
        if self._sidecar is not None:
            save_image_sidecar(self.path, self._sidecar)

    def reload_sidecar(self):
        """Force reload sidecar from disk"""
        self._sidecar = load_image_sidecar(self.path)

    # --- Image Handling ---

    def get_image(self, mode='RGB') -> Image.Image:
        """
        取得 PIL Image 物件。預設會快取以供後續使用。
        若記憶體吃緊，使用完後請呼叫 clear_image_cache()。
        """
        if self._image_cache:
            if mode and self._image_cache.mode != mode:
                # 若需要不同 mode，暫不快取這個轉換版，或者回傳副本
                return self._image_cache.convert(mode)
            return self._image_cache
        
        # Load independent copy
        img = Image.open(self.path)
        if mode and img.mode != mode:
            img = img.convert(mode)
        
        self._image_cache = img
        return img

    def clear_image_cache(self):
        """釋放圖片記憶體"""
        if self._image_cache:
            self._image_cache.close()
            self._image_cache = None

    # --- Tag / Text Helpers ---
    
    def get_caption_text(self) -> str:
        """讀取對應的 .txt 內容"""
        if os.path.exists(self.txt_path):
            try:
                with open(self.txt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return ""
    
    def save_caption_text(self, content: str):
        """寫入 .txt 內容"""
        try:
            with open(self.txt_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass

    # --- State / Backup (Placeholder for future) ---
    
    @property
    def has_tags(self) -> bool:
        """檢查 Sidecar 中是否有 WD14 標籤"""
        # 這裡假設 sidecar 結構，可依需求調整
        sc = self.sidecar
        return bool(sc.get("tagger_tags") or sc.get("rating"))

    @property
    def has_backup(self) -> bool:
        """檢查是否有原始備份"""
        backup_dir = os.path.join(self.dir_path, "raw_image")
        backup_path = os.path.join(backup_dir, self.filename)
        return os.path.exists(backup_path)

