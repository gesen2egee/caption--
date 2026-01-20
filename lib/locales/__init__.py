# -*- coding: utf-8 -*-
"""
語言包載入模組

每個語言一個 JSON 檔案，存放於 lib/locales/ 目錄。
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

_LOCALES_DIR = Path(__file__).parent
_cache: Dict[str, Dict[str, str]] = {}
_current_locale: Dict[str, str] = {}


def load_locale(code: str = "zh_tw") -> Dict[str, str]:
    """
    載入指定語言包。
    
    Args:
        code: 語言代碼，例如 "zh_tw" 或 "en"
        
    Returns:
        翻譯字典 {key: translated_string}
    """
    global _current_locale
    
    if code in _cache:
        _current_locale = _cache[code]
        return _current_locale
    
    path = _LOCALES_DIR / f"{code}.json"
    if not path.exists():
        # Fallback to English
        path = _LOCALES_DIR / "en.json"
        if not path.exists():
            print(f"[Locales] 警告: 找不到語言檔 {code}.json 或 en.json")
            _current_locale = {}
            return _current_locale
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            _cache[code] = json.load(f)
        _current_locale = _cache[code]
    except Exception as e:
        print(f"[Locales] 載入語言檔失敗 {path}: {e}")
        _current_locale = {}
    
    return _current_locale


def get_current_locale() -> Dict[str, str]:
    """取得當前載入的語言包"""
    return _current_locale


def tr(key: str, locale: Optional[Dict[str, str]] = None) -> str:
    """
    翻譯函數。
    
    Args:
        key: 翻譯鍵
        locale: 可選的語言包字典，若未提供則使用當前載入的語言包
        
    Returns:
        翻譯後的字串，若找不到則回傳原始 key
    """
    if locale is None:
        locale = _current_locale
    return locale.get(key, key)


def get_available_locales() -> list:
    """取得所有可用的語言代碼"""
    locales = []
    for f in _LOCALES_DIR.glob("*.json"):
        locales.append(f.stem)
    return sorted(locales)
