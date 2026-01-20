# -*- coding: utf-8 -*-
"""
Sidecar JSON 操作模組

處理圖片對應的 sidecar JSON 檔案（儲存標籤、NL 頁面、遮罩狀態等）。
"""
import os
import json


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
