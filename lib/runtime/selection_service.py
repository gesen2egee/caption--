# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from natsort import natsorted

from lib.utils.file_ops import load_image_sidecar
from lib.utils.query_filter import DanbooruQueryFilter

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
IGNORED_DIR_NAMES = {"no_used", "unmask"}
IMAGE_BUNDLE_EXTENSIONS = (".txt", ".npz", ".boorutag", ".pool.json", ".json")


def discover_image_files(
    root_dir: str,
    valid_exts: Iterable[str] = IMAGE_EXTENSIONS,
    ignored_dirs: Iterable[str] = IGNORED_DIR_NAMES,
) -> list[str]:
    root_dir = str(root_dir or "").strip()
    if not root_dir or not os.path.isdir(root_dir):
        return []

    valid_exts = tuple(str(ext or "").lower() for ext in valid_exts)
    ignored_dir_names = {str(name or "").lower() for name in ignored_dirs}
    image_files: list[str] = []

    def _is_ignored(path: str) -> bool:
        return any(part.lower() in ignored_dir_names for part in Path(path).parts)

    try:
        for entry in os.scandir(root_dir):
            if entry.is_file() and entry.name.lower().endswith(valid_exts):
                if not _is_ignored(entry.path):
                    image_files.append(entry.path)
    except Exception:
        pass

    try:
        for entry in os.scandir(root_dir):
            if not entry.is_dir() or entry.name.lower() in ignored_dir_names:
                continue
            try:
                for sub in os.scandir(entry.path):
                    if sub.is_file() and sub.name.lower().endswith(valid_exts):
                        if not _is_ignored(sub.path):
                            image_files.append(sub.path)
            except Exception:
                pass
    except Exception:
        pass

    return list(natsorted(image_files))


def load_text_content(image_path: str) -> str:
    txt_path = os.path.splitext(str(image_path or ""))[0] + ".txt"
    if not os.path.exists(txt_path):
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return ""


def build_filter_content(
    image_path: str,
    *,
    include_tags: bool = True,
    include_text: bool = False,
) -> str:
    content_parts: list[str] = []
    if include_tags:
        sidecar = load_image_sidecar(image_path)
        content_parts.append(str(sidecar.get("tagger_tags", "") or ""))
    if include_text:
        content_parts.append(load_text_content(image_path))
    return " ".join(part for part in content_parts if part)


def filter_image_paths(
    image_paths: Iterable[str],
    query: str,
    *,
    include_tags: bool = True,
    include_text: bool = False,
) -> list[str]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return list(image_paths)

    qf = DanbooruQueryFilter(normalized_query)
    matched = [
        img_path
        for img_path in image_paths
        if qf.matches(
            build_filter_content(
                img_path,
                include_tags=include_tags,
                include_text=include_text,
            )
        )
    ]
    return list(qf.sort_images(matched))


def delete_image_bundle(
    image_path: str,
    *,
    destination_subdir: str = "no_used",
    extra_extensions: Iterable[str] = IMAGE_BUNDLE_EXTENSIONS,
) -> dict[str, Any]:
    source_image = str(image_path or "").strip()
    if not source_image:
        return {
            "deleted": False,
            "reason": "no_current_image",
        }

    src_dir = os.path.dirname(source_image)
    destination_dir = os.path.join(src_dir, str(destination_subdir or "no_used"))
    os.makedirs(destination_dir, exist_ok=True)

    files_to_move = [source_image]
    for ext in extra_extensions:
        candidate = os.path.splitext(source_image)[0] + str(ext or "")
        if os.path.exists(candidate):
            files_to_move.append(candidate)

    moved_files: list[str] = []
    move_errors: list[dict[str, str]] = []
    for file_path in files_to_move:
        try:
            target_path = os.path.join(destination_dir, os.path.basename(file_path))
            shutil.move(file_path, target_path)
            moved_files.append(target_path)
        except Exception as exc:
            move_errors.append(
                {
                    "source_path": file_path,
                    "error": str(exc),
                }
            )

    return {
        "deleted": True,
        "source_image": source_image,
        "destination_dir": destination_dir,
        "moved_file_count": len(moved_files),
        "moved_files": moved_files,
        "move_error_count": len(move_errors),
        "move_errors": move_errors,
    }
