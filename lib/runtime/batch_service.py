# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Callable, Iterable

from lib.runtime.selection_service import load_text_content
from lib.utils.batch_writer import write_batch_result
from lib.utils.sidecar import load_image_sidecar


def restore_batch_tagger_to_txt(
    image_paths: Iterable[str],
    cfg: Any,
    *,
    delete_chars: bool = False,
    write_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    restored_files: list[str] = []
    files_to_process: list[str] = []

    for img_path in image_paths:
        normalized_path = str(img_path or "").strip()
        if not normalized_path:
            continue
        sidecar = load_image_sidecar(normalized_path)
        tags_str = str(sidecar.get("tagger_tags", "") or "")
        if tags_str:
            final = write_batch_result(normalized_path, tags_str, True, cfg, delete_chars)
            if callable(write_callback) and final:
                write_callback(normalized_path, final)
            restored_files.append(normalized_path)
        else:
            files_to_process.append(normalized_path)

    return {
        "restored_files": restored_files,
        "restored_count": len(restored_files),
        "files_to_process": files_to_process,
        "files_to_process_count": len(files_to_process),
    }


def restore_batch_llm_to_txt(
    image_paths: Iterable[str],
    cfg: Any,
    *,
    delete_chars: bool = False,
    write_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    restored_files: list[str] = []
    files_to_process: list[str] = []

    for img_path in image_paths:
        normalized_path = str(img_path or "").strip()
        if not normalized_path:
            continue
        sidecar = load_image_sidecar(normalized_path)
        nl_pages = sidecar.get("nl_pages", [])
        content = nl_pages[-1] if nl_pages and isinstance(nl_pages, list) else ""
        if content:
            final = write_batch_result(normalized_path, content, False, cfg, delete_chars)
            if callable(write_callback) and final:
                write_callback(normalized_path, final)
            restored_files.append(normalized_path)
        else:
            files_to_process.append(normalized_path)

    return {
        "restored_files": restored_files,
        "restored_count": len(restored_files),
        "files_to_process": files_to_process,
        "files_to_process_count": len(files_to_process),
    }


def filter_background_targets(
    image_paths: Iterable[str],
    *,
    sidecar_loader: Callable[[str], dict[str, Any]] = load_image_sidecar,
    text_loader: Callable[[str], str] = load_text_content,
) -> list[str]:
    targets: list[str] = []
    for img_path in image_paths:
        normalized_path = str(img_path or "").strip()
        if not normalized_path:
            continue
        try:
            sidecar = sidecar_loader(normalized_path)
            tags_all = (
                str(sidecar.get("tagger_tags", "") or "")
                + " "
                + str(sidecar.get("tags_context", "") or "")
            ).lower()
            txt_content = text_loader(normalized_path).lower()
            if "background" in tags_all or "background" in txt_content:
                targets.append(normalized_path)
        except Exception:
            continue
    return targets
