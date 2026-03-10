# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import os
import shutil
from io import BytesIO
from typing import Any, Callable, Optional

from PIL import Image, ImageChops

from lib.utils.file_ops import backup_raw_image, has_raw_backup
from lib.utils.image_processing import process_mask_channel
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar
from lib.utils.tag_context import build_llm_tags_context_for_image


def build_processing_result(**kwargs) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "started": False,
        "task_started": False,
    }
    payload.update(kwargs)
    return payload


def plan_current_tagger_action(
    current_image_path: str,
    *,
    task_running: bool,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {"ready": False, "result": build_processing_result(reason="no_current_image")}
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(reason="task_running", image_path=normalized_path),
        }
    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            task_started=True,
            mode="tagger",
            image_path=normalized_path,
        ),
    }


def plan_current_llm_action(
    current_image_path: str,
    *,
    task_running: bool,
    user_prompt: Optional[str],
    runtime_prompt: str,
    confirm_missing_tags: bool,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {"ready": False, "result": build_processing_result(reason="no_current_image")}
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(reason="task_running", image_path=normalized_path),
        }

    resolved_user_prompt = str(user_prompt) if user_prompt is not None else str(runtime_prompt or "")
    tags_text = build_llm_tags_context_for_image(normalized_path)
    tags_context_available = bool(tags_text.strip())
    if "{tags}" in resolved_user_prompt and not tags_context_available and not confirm_missing_tags:
        return {
            "ready": False,
            "requires_confirmation": "confirm_missing_tags",
            "result": build_processing_result(
                reason="confirm_missing_tags_required",
                image_path=normalized_path,
                prompt_contains_tags_placeholder=True,
                tags_context_available=False,
            ),
        }

    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            task_started=True,
            mode="llm",
            image_path=normalized_path,
            user_prompt=resolved_user_prompt,
            confirm_missing_tags=bool(confirm_missing_tags),
            tags_context_available=tags_context_available,
        ),
    }


def plan_current_image_process_action(
    current_image_path: str,
    *,
    task_running: bool,
    explicit_prompt: Optional[str],
    runtime_prompt: str,
    default_prompt: str,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {
            "ready": False,
            "result": build_processing_result(reason="no_current_image"),
        }
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(reason="task_running", image_path=normalized_path),
        }

    prompt = resolve_image_process_prompt(explicit_prompt, runtime_prompt, default_prompt)
    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            task_started=True,
            mode="image_process",
            image_path=normalized_path,
            edit_prompt=prompt,
        ),
        "resolved_prompt": prompt,
    }


def plan_current_unmask_action(
    current_image_path: str,
    *,
    task_running: bool,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {
            "ready": False,
            "result": build_processing_result(reason="no_current_image"),
        }
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(reason="task_running", image_path=normalized_path),
        }
    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            task_started=True,
            mode="unmask",
            image_path=normalized_path,
        ),
    }


def plan_current_mask_text_action(
    current_image_path: str,
    *,
    task_running: bool,
    ocr_enabled: bool,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {
            "ready": False,
            "result": build_processing_result(reason="no_current_image"),
        }
    if not ocr_enabled:
        return {
            "ready": False,
            "result": build_processing_result(reason="ocr_disabled", image_path=normalized_path),
        }
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(reason="task_running", image_path=normalized_path),
        }
    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            task_started=True,
            mode="mask_text",
            image_path=normalized_path,
        ),
    }


def plan_current_restore_action(
    current_image_path: str,
    *,
    task_running: bool,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {
            "ready": False,
            "result": build_processing_result(reason="no_current_image"),
        }
    if not has_raw_backup(normalized_path):
        return {
            "ready": False,
            "result": build_processing_result(reason="no_backup", image_path=normalized_path),
        }
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(reason="task_running", image_path=normalized_path),
        }
    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            task_started=True,
            mode="restore",
            image_path=normalized_path,
        ),
    }


def plan_current_stroke_eraser_action(
    current_image_path: str,
    *,
    task_running: bool,
) -> dict[str, Any]:
    normalized_path = str(current_image_path or "").strip()
    if not normalized_path:
        return {
            "ready": False,
            "result": build_processing_result(reason="no_current_image", mode="stroke_eraser"),
        }
    if task_running:
        return {
            "ready": False,
            "result": build_processing_result(
                reason="task_running",
                mode="stroke_eraser",
                image_path=normalized_path,
            ),
        }
    return {
        "ready": True,
        "result": build_processing_result(
            started=True,
            mode="stroke_eraser",
            image_path=normalized_path,
        ),
    }


def resolve_image_process_prompt(
    explicit_prompt: Optional[str],
    runtime_prompt: str,
    default_prompt: str,
) -> str:
    prompt = str(explicit_prompt or "").strip()
    if prompt:
        return prompt

    prompt = str(runtime_prompt or "").strip()
    if prompt:
        return prompt

    return str(default_prompt or "").strip()


def load_stroke_mask(
    *,
    mask_png_base64: Optional[str] = None,
    mask_png_bytes: Optional[bytes] = None,
    mask_path: Optional[str] = None,
) -> tuple[Image.Image, str, str]:
    if mask_png_bytes:
        try:
            with Image.open(BytesIO(mask_png_bytes)) as image:
                return image.convert("L"), "bytes", ""
        except Exception as exc:
            raise ValueError(f"failed to decode stroke mask png bytes: {exc}") from exc

    raw_base64 = str(mask_png_base64 or "").strip()
    if raw_base64:
        if raw_base64.startswith("data:"):
            _, _, raw_base64 = raw_base64.partition(",")
        try:
            decoded_bytes = base64.b64decode(raw_base64)
        except Exception as exc:
            raise ValueError(f"invalid mask_png_base64: {exc}") from exc
        return load_stroke_mask(mask_png_bytes=decoded_bytes)

    resolved_mask_path = str(mask_path or "").strip()
    if resolved_mask_path:
        if not os.path.isfile(resolved_mask_path):
            raise ValueError(f"mask file does not exist: {resolved_mask_path}")
        try:
            with Image.open(resolved_mask_path) as image:
                return image.convert("L"), "path", resolved_mask_path
        except Exception as exc:
            raise ValueError(f"failed to load stroke mask: {resolved_mask_path}: {exc}") from exc

    raise ValueError("mask_png_base64, mask_png_bytes, or mask_path is required")


def apply_stroke_eraser(
    image_path: str,
    mask_image: Image.Image,
    *,
    shrink: int,
    blur: float,
    min_alpha: int,
    unique_path_resolver: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    normalized_path = str(image_path or "").strip()
    if not normalized_path:
        raise ValueError("image_path is required")
    if mask_image is None:
        raise ValueError("mask_image is required")

    resolver = unique_path_resolver or _default_unique_path
    backup_created = bool(backup_raw_image(normalized_path))

    src_dir = os.path.dirname(normalized_path)
    unmask_dir = os.path.join(src_dir, "unmask")
    os.makedirs(unmask_dir, exist_ok=True)

    ext = os.path.splitext(normalized_path)[1].lower()
    base_no_ext = os.path.splitext(normalized_path)[0]
    target_file = base_no_ext + ".webp"
    moved_original = ""

    if ext == ".webp":
        moved_original = resolver(os.path.join(unmask_dir, os.path.basename(normalized_path)))
        shutil.move(normalized_path, moved_original)
        src_for_processing = moved_original
        target_file = normalized_path
    else:
        src_for_processing = normalized_path
        if os.path.exists(target_file):
            existing_backup = resolver(os.path.join(unmask_dir, os.path.basename(target_file)))
            shutil.move(target_file, existing_backup)

    with Image.open(src_for_processing) as image:
        image_rgba = image.convert("RGBA")
        resized_mask = mask_image.convert("L").resize(image_rgba.size, Image.Resampling.NEAREST)
        keep_mask = Image.eval(resized_mask, lambda value: 0 if value > 0 else 255)
        keep_processed = process_mask_channel(
            keep_mask,
            shrink=int(shrink or 0),
            blur=float(blur or 0),
            min_alpha=int(min_alpha or 0),
        )

        alpha = image_rgba.getchannel("A")
        new_alpha = ImageChops.multiply(alpha, keep_processed)

        merged_pixels = []
        for original_pixel, new_alpha_value in zip(image_rgba.getdata(), new_alpha.getdata()):
            if original_pixel[3] == 0:
                merged_pixels.append((255, 255, 255, new_alpha_value))
            else:
                merged_pixels.append((original_pixel[0], original_pixel[1], original_pixel[2], new_alpha_value))

        image_rgba.putdata(merged_pixels)
        image_rgba.save(target_file, "WEBP")

    if ext != ".webp":
        moved_original = resolver(os.path.join(unmask_dir, os.path.basename(normalized_path)))
        shutil.move(normalized_path, moved_original)

        old_json = normalized_path + ".json"
        if os.path.exists(old_json):
            sidecar = load_image_sidecar(normalized_path)
            save_image_sidecar(target_file, sidecar)

            backup_json = moved_original + ".json"
            shutil.move(old_json, backup_json)

    return {
        "source_image_path": normalized_path,
        "output_image_path": target_file,
        "image_replaced": os.path.abspath(normalized_path) != os.path.abspath(target_file),
        "backup_created": backup_created,
        "moved_original_path": moved_original,
        "target_format": "webp",
        "mask_size": list(mask_image.size),
    }


def _default_unique_path(path: str) -> str:
    normalized_path = str(path or "").strip()
    if not os.path.exists(normalized_path):
        return normalized_path

    base, ext = os.path.splitext(normalized_path)
    for index in range(1, 9999):
        candidate = f"{base}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return normalized_path
