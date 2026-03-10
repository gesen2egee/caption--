# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional


def _safe_message(exc: BaseException | None, fallback: str) -> str:
    if exc is None:
        return fallback
    text = str(exc or "").strip()
    return text or fallback


def _error_code_from_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "runtime_error"
    explicit_code = getattr(exc, "code", None)
    if explicit_code:
        return str(explicit_code)

    error_type = exc.__class__.__name__
    if error_type == "CommandAccessError":
        return "command_access_denied"
    if error_type == "CommandNotFoundError":
        return "unknown_command"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return "invalid_input"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, KeyError):
        return "missing_key"
    return "runtime_error"


def build_runtime_error_info(
    exc: BaseException | None = None,
    *,
    code: Optional[str] = None,
    message: Optional[str] = None,
    source: str = "runtime",
    command_name: Optional[str] = None,
    access_mode: Optional[str] = None,
    status_code: Optional[int] = None,
    include_traceback: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_source = source or getattr(exc, "source", None) or "runtime"
    info: Dict[str, Any] = {
        "code": str(code or _error_code_from_exception(exc)),
        "type": exc.__class__.__name__ if exc is not None else "RuntimeError",
        "message": str(message or _safe_message(exc, "Runtime error")),
        "source": str(resolved_source),
    }

    resolved_command_name = command_name or getattr(exc, "command_name", None)
    resolved_access_mode = access_mode or getattr(exc, "access_mode", None)
    allowed_access_modes = getattr(exc, "allowed_access_modes", None)

    if resolved_command_name:
        info["command_name"] = str(resolved_command_name)
    if resolved_access_mode:
        info["access_mode"] = str(resolved_access_mode)
    if allowed_access_modes:
        info["allowed_access_modes"] = [str(value) for value in allowed_access_modes]
    if status_code is not None:
        info["status_code"] = int(status_code)
    merged_details: Dict[str, Any] = {}
    if getattr(exc, "details", None):
        merged_details.update(dict(getattr(exc, "details")))
    if details:
        merged_details.update(dict(details))
    if merged_details:
        info["details"] = merged_details
    if include_traceback and exc is not None:
        info["traceback"] = traceback.format_exc()
    return info
