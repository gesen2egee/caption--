"""
多語言本地化模組

負責：
- 自動掃描 locales 目錄下的語言配置文件
- 載入和管理翻譯字串
- 提供翻譯函數

使用方式：
    from lib.localization import get_text, get_available_languages, set_language
    
    # 取得翻譯字串
    text = get_text("ui.app_title")  # "Caption 神器" (zh_tw) / "Caption Tool" (en)
    
    # 取得可用語言列表
    languages = get_available_languages()  # [{"code": "zh_tw", "name": "繁體中文"}, ...]
    
    # 切換語言
    set_language("en")
"""

import os
import json
from typing import Dict, Any, List, Optional

# 模組級變數
_current_language: str = "zh_tw"
_locale_cache: Dict[str, Dict[str, Any]] = {}
_locales_dir: str = os.path.join(os.path.dirname(__file__), "locales")


def _load_locale(lang_code: str) -> Dict[str, Any]:
    """載入指定語言的配置文件"""
    if lang_code in _locale_cache:
        return _locale_cache[lang_code]
    
    locale_path = os.path.join(_locales_dir, f"{lang_code}.json")
    if not os.path.exists(locale_path):
        print(f"[Localization] Locale file not found: {locale_path}")
        return {}
    
    try:
        with open(locale_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _locale_cache[lang_code] = data
        return data
    except Exception as e:
        print(f"[Localization] Failed to load locale {lang_code}: {e}")
        return {}


def get_available_languages() -> List[Dict[str, str]]:
    """
    掃描 locales 目錄，返回所有可用的語言列表
    
    Returns:
        List[Dict]: [{"code": "zh_tw", "name": "繁體中文"}, ...]
    """
    languages = []
    
    if not os.path.exists(_locales_dir):
        print(f"[Localization] Locales directory not found: {_locales_dir}")
        return languages
    
    for filename in os.listdir(_locales_dir):
        if filename.endswith(".json"):
            lang_code = filename[:-5]  # 移除 .json
            locale_data = _load_locale(lang_code)
            meta = locale_data.get("meta", {})
            languages.append({
                "code": meta.get("code", lang_code),
                "name": meta.get("name", lang_code)
            })
    
    # 按名稱排序
    languages.sort(key=lambda x: x["name"])
    return languages


def set_language(lang_code: str) -> bool:
    """
    設定當前語言
    
    Args:
        lang_code: 語言代碼 (如 "zh_tw", "en")
    
    Returns:
        bool: 是否成功
    """
    global _current_language
    
    locale_path = os.path.join(_locales_dir, f"{lang_code}.json")
    if os.path.exists(locale_path):
        _current_language = lang_code
        _load_locale(lang_code)  # 預載入
        return True
    else:
        print(f"[Localization] Language not found: {lang_code}")
        return False


def get_current_language() -> str:
    """取得當前語言代碼"""
    return _current_language


def get_text(key: str, **kwargs) -> str:
    """
    取得翻譯字串
    
    Args:
        key: 以點分隔的鍵路徑，如 "ui.app_title" 或 "status.status_processing"
        **kwargs: 用於格式化的參數，如 current=1, total=10, filename="test.png"
    
    Returns:
        翻譯後的字串。如果找不到則返回原始 key。
    
    Example:
        get_text("ui.app_title")  # "Caption 神器"
        get_text("status.status_processing", current=1, total=10, filename="test.png")
        # "處理中 [1/10]: test.png"
    """
    locale_data = _load_locale(_current_language)
    
    # 解析點分隔的路徑
    parts = key.split(".")
    value = locale_data
    
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            # 找不到，嘗試回退到英文
            if _current_language != "en":
                en_data = _load_locale("en")
                value = en_data
                for p in parts:
                    if isinstance(value, dict) and p in value:
                        value = value[p]
                    else:
                        return key  # 真的找不到
                break
            return key
    
    if not isinstance(value, str):
        return key
    
    # 格式化參數
    if kwargs:
        try:
            # 使用 format 方法，支援 {name} 格式
            return value.format(**kwargs)
        except (KeyError, ValueError):
            return value
    
    return value


def tr(key: str, **kwargs) -> str:
    """get_text 的簡寫別名"""
    return get_text(key, **kwargs)


# 初始化：預載入預設語言
_load_locale(_current_language)
