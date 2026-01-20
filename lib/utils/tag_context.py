# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Set

from lib.utils.boorutag import parse_boorutag_meta
from lib.utils.parsing import extract_bracket_content, try_tags_to_text_list
from lib.utils.sidecar import load_image_sidecar

def build_llm_tags_context_for_image(image_path: str) -> str:
    """
    為圖片構建 LLM 上下文標籤字串。
    來源包括：
    1. .boorutag 檔案
    2. 父資料夾名稱 (格式: ID_Tags)
    3. Sidecar JSON 中的 tagger_tags
    """
    top_tags = []
    try:
        hints = []
        tags_from_meta = []

        meta_path = str(image_path) + ".boorutag"
        if os.path.isfile(meta_path):
            tags_meta, hint_info = parse_boorutag_meta(meta_path)
            tags_from_meta.extend(tags_meta)
            hints.extend(hint_info)

        # Folder hint
        parent = Path(image_path).parent.name
        if "_" in parent:
            parts = parent.split("_", 1)
            if len(parts) > 1:
                folder_hint = parts[1]
                if "{" not in folder_hint:
                    folder_hint = f"{{{folder_hint}}}"
                hints.append(folder_hint)

        initial_keywords = []
        for h in hints:
            initial_keywords.extend(extract_bracket_content(h))

        combined = initial_keywords + tags_from_meta
        seen: Set[str] = set()
        top_tags = [x for x in combined if not (x in seen or seen.add(x))]
        top_tags = [str(t).replace("_", " ").strip() for t in top_tags if str(t).strip()]
    except Exception:
        top_tags = []

    # Tagger tags from sidecar
    tagger_parts = []
    try:
        sidecar = load_image_sidecar(image_path)
        raw = sidecar.get("tagger_tags", "")
        if raw:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
            parts = try_tags_to_text_list(parts)
            tagger_parts = [t.replace("_", " ").strip() for t in parts if t.strip()]
    except Exception:
        pass

    # Merge
    all_tags = []
    seen2: Set[str] = set()
    for t in (top_tags + tagger_parts):
        if t and t not in seen2:
            seen2.add(t)
            all_tags.append(t)

    return "\n".join(all_tags)
