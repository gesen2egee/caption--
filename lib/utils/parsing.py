# -*- coding: utf-8 -*-
"""
標籤解析和文字處理模組
"""
import re
from typing import List, Dict, Optional

# 嘗試匯入 tags_to_text（用於 try_tags_to_text_list）
try:
    from imgutils.tagging import tags_to_text
except ImportError:
    tags_to_text = None


def extract_bracket_content(text: str) -> List[str]:
    """提取大括號內的內容"""
    return re.findall(r'\{(.*?)\}', text)


def remove_underline(s: str) -> str:
    """去底線 helper"""
    return s.replace("_", " ") if s else ""


def smart_parse_tags(text: str) -> List[Dict[str, Optional[str]]]:
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
    
    t = text.strip().lower()
    
    bl_words = [w.strip().lower() for w in cfg.get("char_tag_blacklist_words", []) if w.strip()]
    wl_words = [w.strip().lower() for w in cfg.get("char_tag_whitelist_words", []) if w.strip()]
    
    if not bl_words:
        return False
    
    has_blacklist = any(bw in t for bw in bl_words)
    if not has_blacklist:
        return False
    
    has_whitelist = any(ww in t for ww in wl_words)
    if has_whitelist:
        return False
    
    return True


def normalize_for_match(s: str) -> str:
    """正規化字串以進行比對"""
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace(", ", "").replace(",", "")
    t = t.strip()
    t = t.rstrip(".")
    return t.strip()


def cleanup_csv_like_text(text: str, force_lower: bool = False) -> str:
    """清理 CSV 格式的文字"""
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    result = ", ".join(parts)
    if force_lower:
        result = result.lower()
    return result


def split_csv_like_text(text: str) -> List[str]:
    """分割 CSV 格式的文字"""
    return [p.strip() for p in text.split(",") if p.strip()]


def try_tags_to_text_list(tags_list) -> List[str]:
    """
    先 tags_to_text，再拆回 list；若失敗就 fallback
    """
    try:
        if tags_to_text is not None:
            s = tags_to_text(tags_list)
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts
        else:
            return [t.strip() for t in tags_list if str(t).strip()]
    except Exception:
        return [t.strip() for t in tags_list if str(t).strip()]
