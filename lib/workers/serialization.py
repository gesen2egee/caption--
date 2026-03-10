# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from lib.core.dataclasses import BatchInstruction, FolderMeta, ImageData, Prompt, Settings
from lib.workers.base import WorkerInput, WorkerOutput


def _serialize_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


def _deserialize_image(data: Optional[Dict[str, Any]]) -> Optional[ImageData]:
    if not isinstance(data, dict):
        return None
    return ImageData(**data)


def _deserialize_settings(data: Optional[Dict[str, Any]]) -> Optional[Settings]:
    if not isinstance(data, dict):
        return None
    return Settings(**data)


def _deserialize_prompt(data: Optional[Dict[str, Any]]) -> Optional[Prompt]:
    if not isinstance(data, dict):
        return None
    batches = [
        BatchInstruction(**item)
        for item in list(data.get("batches", []) or [])
        if isinstance(item, dict)
    ]
    payload = dict(data)
    payload["batches"] = batches
    return Prompt(**payload)


def _deserialize_folder(data: Optional[Dict[str, Any]]) -> Optional[FolderMeta]:
    if not isinstance(data, dict):
        return None
    return FolderMeta(**data)


def serialize_worker_input(worker_input: WorkerInput) -> Dict[str, Any]:
    return {
        "image": _serialize_dataclass(worker_input.image),
        "images": [_serialize_dataclass(image) for image in list(worker_input.images or [])],
        "settings": _serialize_dataclass(worker_input.settings),
        "prompt": _serialize_dataclass(worker_input.prompt),
        "folder": _serialize_dataclass(worker_input.folder),
        "extra": dict(worker_input.extra or {}),
    }


def deserialize_worker_input(payload: Dict[str, Any]) -> WorkerInput:
    payload = dict(payload or {})
    return WorkerInput(
        image=_deserialize_image(payload.get("image")),
        images=[image for image in [_deserialize_image(item) for item in list(payload.get("images", []) or [])] if image is not None],
        settings=_deserialize_settings(payload.get("settings")),
        prompt=_deserialize_prompt(payload.get("prompt")),
        folder=_deserialize_folder(payload.get("folder")),
        extra=dict(payload.get("extra", {}) or {}),
    )


def serialize_worker_output(worker_output: WorkerOutput) -> Dict[str, Any]:
    return {
        "success": bool(worker_output.success),
        "image": _serialize_dataclass(worker_output.image),
        "images": [_serialize_dataclass(image) for image in list(worker_output.images or [])],
        "result_text": worker_output.result_text,
        "result_data": worker_output.result_data,
        "error": worker_output.error,
        "error_info": dict(worker_output.error_info or {}) if worker_output.error_info else None,
        "skipped": bool(worker_output.skipped),
        "skip_reason": worker_output.skip_reason,
        "metadata": dict(worker_output.metadata or {}),
    }


def deserialize_worker_output(payload: Dict[str, Any]) -> WorkerOutput:
    payload = dict(payload or {})
    return WorkerOutput(
        success=bool(payload.get("success", False)),
        image=_deserialize_image(payload.get("image")),
        images=[image for image in [_deserialize_image(item) for item in list(payload.get("images", []) or [])] if image is not None],
        result_text=payload.get("result_text"),
        result_data=payload.get("result_data"),
        error=payload.get("error"),
        error_info=dict(payload.get("error_info", {}) or {}) or None,
        skipped=bool(payload.get("skipped", False)),
        skip_reason=payload.get("skip_reason"),
        metadata=dict(payload.get("metadata", {}) or {}),
    )
