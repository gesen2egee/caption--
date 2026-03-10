# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Any, Iterable


def format_find_replace_content(content: str) -> str:
    parts = [p.strip() for p in str(content or "").split(",") if p.strip()]
    return ", ".join(parts)


def run_find_replace_on_images(
    image_paths: Iterable[str],
    *,
    find_text: str,
    replace_text: str = "",
    case_sensitive: bool = False,
    regex: bool = False,
) -> dict[str, Any]:
    normalized_find = str(find_text or "")
    normalized_replace = str(replace_text or "")
    if not normalized_find:
        raise ValueError("find_text is required")

    replacement_count = 0
    changed_files: list[str] = []
    skipped_missing_txt: list[str] = []
    errors: list[dict[str, str]] = []

    for img_path in image_paths:
        normalized_path = str(img_path or "").strip()
        if not normalized_path:
            continue

        txt_path = os.path.splitext(normalized_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            skipped_missing_txt.append(normalized_path)
            continue

        try:
            with open(txt_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            new_content = content
            flags = 0 if case_sensitive else re.IGNORECASE
            replaced = 0

            if regex:
                new_content, replaced = re.subn(normalized_find, normalized_replace, content, flags=flags)
            elif case_sensitive:
                replaced = content.count(normalized_find)
                if replaced > 0:
                    new_content = content.replace(normalized_find, normalized_replace)
            else:
                pattern = re.compile(re.escape(normalized_find), re.IGNORECASE)
                new_content, replaced = pattern.subn(normalized_replace, content)

            replacement_count += replaced
            if new_content != content:
                formatted = format_find_replace_content(new_content)
                with open(txt_path, "w", encoding="utf-8") as handle:
                    handle.write(formatted)
                changed_files.append(normalized_path)
        except Exception as exc:
            errors.append(
                {
                    "image_path": normalized_path,
                    "error": str(exc),
                }
            )

    return {
        "find_text": normalized_find,
        "replace_text": normalized_replace,
        "case_sensitive": bool(case_sensitive),
        "regex": bool(regex),
        "target_count": len([str(path or "").strip() for path in image_paths if str(path or "").strip()]),
        "replacement_count": replacement_count,
        "changed_file_count": len(changed_files),
        "changed_files": changed_files,
        "skipped_missing_txt_count": len(skipped_missing_txt),
        "skipped_missing_txt_files": skipped_missing_txt,
        "error_count": len(errors),
        "errors": errors,
    }
