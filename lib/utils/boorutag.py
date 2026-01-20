# -*- coding: utf-8 -*-
"""
Boorutag Meta 解析和標籤翻譯

包含：
- parse_boorutag_meta: 解析 boorutag meta 檔案
- ensure_tags_csv: 確保 Tags.csv 存在
- load_translations: 載入標籤翻譯
"""
import os
import csv
from typing import List, Tuple, Dict

try:
    from imgutils.tagging import remove_underline
except ImportError:
    def remove_underline(s):
        return s.replace("_", " ")


TAGS_CSV_LOCAL = "Tags.csv"
TAGS_CSV_URL_RAW = "https://raw.githubusercontent.com/waldolin/a1111-sd-webui-tagcomplete-TW/main/tags/Tags-tw-full-pack.csv"


def ensure_tags_csv(csv_path: str = TAGS_CSV_LOCAL) -> bool:
    """
    確保 Tags.csv 存在，若不存在則下載。
    """
    if os.path.exists(csv_path):
        return True
    
    try:
        import requests
        print(f"[Tags] 下載 {TAGS_CSV_URL_RAW}")
        resp = requests.get(TAGS_CSV_URL_RAW, timeout=30)
        resp.raise_for_status()
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        print(f"[Tags] 已儲存到 {csv_path}")
        return True
    except Exception as e:
        print(f"[Tags] 下載失敗: {e}")
        return False


def load_translations(csv_path: str = TAGS_CSV_LOCAL) -> Dict[str, str]:
    """
    載入標籤翻譯字典。
    """
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


def parse_boorutag_meta(meta_path: str) -> Tuple[List[str], List[str]]:
    """
    解析 boorutag meta 檔案。
    
    Returns:
        (tags_meta, hint_info)
    """
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
                    s_str = ', '.join(sources)
                    hint_info.append(f"the copyright of this image: {{{{{s_str}}}}}")

            if len(lines) >= 13 and lines[12]:
                characters = [c.strip() for c in lines[12].split(',') if c.strip()]
                if characters and len(characters) < 4:
                    c_str = ', '.join(characters)
                    hint_info.append(f"the characters of this image: {{{{{c_str}}}}}")

    except Exception as e:
        print(f"[boorutag] 解析出錯 {meta_path}: {e}")
    return tags_meta, hint_info
