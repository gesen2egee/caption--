# -*- coding: utf-8 -*-
import os
import re
from typing import Union, Dict, Any
from lib.utils.parsing import is_basic_character_tag, cleanup_csv_like_text

def write_batch_result(image_path: str, content: str, is_tagger: bool, cfg: Any, delete_chars: bool = False) -> str:
    """
    處理內容並寫入同名 .txt 檔案。
    
    Args:
        image_path: 圖片路徑
        content: 內容 (tagger tags string OR llm content string)
        is_tagger: 是否為 tagger 結果 (格式不同)
        cfg: 設定 (Dict 或 Settings 物件，需支援 .get 或屬性存取? 這裡假設其 behave like dict access for get)
             Wait, original code used `cfg.get`. If Settings object, it uses attributes usually, but main_window.py accessed it?
             Snippet 824: `cfg = self.settings`. `cfg.get(...)`.
             Wait, `Settings` dataclass (Step 774) does NOT have `.get()` method!
             It's a dataclass.
             Unless `MainWindow` wrapped it?
             Ah, `MainWindow` loads settings via `load_app_settings` which returns `Settings` dataclass.
             But dataclass doesn't have `.get()`.
             How does `cfg.get("batch_to_txt_mode")` work in `MainWindow`?
             Snippet 824 line 1980: `cfg = self.settings`.
             Line 1982: `mode = cfg.get("batch_to_txt_mode", "append")`.
             IF `self.settings` is `Settings` object, this crashes!
             UNLESS `self.settings` is a DICT!
             Let's check `MainWindow` init.
             Snippet 767 line 3050+.
             `self.settings = load_app_settings()`.
             Check `lib/core/settings.py` `load_app_settings`.
             Snippet 774 line 31 `create_settings_from_dict` returns `Settings`.
             Snippet 767 line 137 imported logic.
             If `load_app_settings` returns `dict` (old version) or `Settings` (new version)?
             I'll assume it might be a dict based on usage.
             BUT if it IS a dataclass, I need `getattr(cfg, "key", default)`.
             
             To be safe, I'll use a helper that handles both.
    
    Returns:
        寫入的最終字串。
    """
    
    def get_cfg(key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    mode = get_cfg("batch_to_txt_mode", "append")
    folder_trigger = get_cfg("batch_to_txt_folder_trigger", False)
    force_lower = get_cfg("english_force_lowercase", True)
    
    items = []
    if is_tagger:
        raw_list = [x.strip() for x in content.split(",") if x.strip()]
        if delete_chars:
            raw_list = [t for t in raw_list if not is_basic_character_tag(t, cfg)] # is_basic_character_tag implementation expects dict?
        items = raw_list
    else:
        # nl_content logic
        raw_lines = content.splitlines()
        sentences = []
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            if (line.startswith("(") and line.endswith(")")) or (line.startswith("（") and line.endswith("）")):
                continue
            line = re.sub(r"[\(（].*?[\)）]", "", line).strip()
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

    # Deduplication
    if mode == "append" and existing_content and items:
        search_text = existing_content.lower().replace("_", " ").replace("\n", " ")
        search_text = re.sub(r"\s+", " ", search_text)
        
        unique_items = []
        for item in items:
            t_norm = item.strip().lower().replace("_", " ")
            t_norm = re.sub(r"\s+", " ", t_norm)
            if not t_norm: 
                continue
            
            try:
                pattern = r"(?<!\w)" + re.escape(t_norm) + r"(?!\w)"
                if not re.search(pattern, search_text):
                    unique_items.append(item)
            except Exception:
                if t_norm not in search_text:
                    unique_items.append(item)
        
        items = unique_items
        if not items:
            return existing_content # No new content
    
    if is_tagger:
        new_part = ", ".join(items)
        if mode == "append" and existing_content:
            final = cleanup_csv_like_text(existing_content + ", " + new_part, force_lower)
        else:
            final = cleanup_csv_like_text(new_part, force_lower)
    else:
        new_part = ", ".join(items)
        if mode == "append" and existing_content:
            sep = ", "
            if existing_content.endswith(",") or existing_content.endswith("."):
                sep = " "
            final = existing_content + sep + new_part
        else:
            final = new_part
            
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final)
        return final
    except Exception as e:
        print(f"[BatchWriter] 寫入失敗 {txt_path}: {e}")
        return ""
