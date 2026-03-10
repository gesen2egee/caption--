# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from lib.workers.base import WorkerOutput


@dataclass
class WorkerError(RuntimeError):
    message: str
    code: str = "worker_error"
    source: str = "worker"
    category: Optional[str] = None
    worker_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


class WorkerNotFoundError(WorkerError):
    def __init__(self, category: str, worker_name: str):
        super().__init__(
            f"Worker '{worker_name}' not found in category '{category}'",
            code="worker_not_found",
            source="worker.registry",
            category=category,
            worker_name=worker_name,
        )


class WorkerServiceUnavailableError(WorkerError):
    def __init__(self, category: str, worker_name: str, message: str = "worker service process is unavailable"):
        super().__init__(
            message,
            code="worker_service_unavailable",
            source="worker.service",
            category=category,
            worker_name=worker_name,
        )


class WorkerServiceTimeoutError(WorkerError):
    def __init__(self, category: str, worker_name: str, *, timeout: float, stderr_tail: Optional[list[str]] = None):
        details: Dict[str, Any] = {"timeout_seconds": float(timeout)}
        if stderr_tail:
            details["stderr_tail"] = list(stderr_tail)[-8:]
        super().__init__(
            f"worker service timeout for {category}/{worker_name}",
            code="worker_service_timeout",
            source="worker.service",
            category=category,
            worker_name=worker_name,
            details=details,
        )


class WorkerServiceRemoteError(WorkerError):
    def __init__(
        self,
        category: str,
        worker_name: str,
        message: str,
        *,
        error_info: Optional[Dict[str, Any]] = None,
        stderr_tail: Optional[list[str]] = None,
    ):
        details: Dict[str, Any] = {}
        if error_info:
            details["remote_error_info"] = dict(error_info)
        if stderr_tail:
            details["stderr_tail"] = list(stderr_tail)[-8:]
        super().__init__(
            message,
            code=str((error_info or {}).get("code") or "worker_service_remote_error"),
            source=str((error_info or {}).get("source") or "worker.service"),
            category=category,
            worker_name=worker_name,
            details=details or None,
        )


class WorkerUnsupportedActionError(WorkerError):
    def __init__(self, action: str):
        super().__init__(
            f"unsupported action: {action}",
            code="worker_unsupported_action",
            source="worker.service",
            details={"action": str(action)},
        )


def _guess_error_code_from_message(message: str) -> str:
    text = str(message or "").strip().lower()
    if not text:
        return "worker_error"
    if "timeout" in text or "逾時" in text:
        return "worker_service_timeout"
    if "not found" in text or "找不到" in text:
        return "worker_not_found"
    if "api key" in text:
        return "worker_auth_error"
    if "permission" in text or "權限" in text:
        return "worker_permission_denied"
    if "empty" in text or "缺少" in text or "missing" in text:
        return "worker_invalid_input"
    return "worker_error"


def build_worker_error_info(
    exc: BaseException | None = None,
    *,
    code: Optional[str] = None,
    message: Optional[str] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
    worker_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_message = str(message or getattr(exc, "message", None) or str(exc or "") or "Worker error").strip() or "Worker error"
    resolved_code = str(
        code
        or getattr(exc, "code", None)
        or _guess_error_code_from_message(resolved_message)
    )
    info: Dict[str, Any] = {
        "code": resolved_code,
        "type": exc.__class__.__name__ if exc is not None else "WorkerError",
        "message": resolved_message,
        "source": str(source or getattr(exc, "source", None) or "worker"),
    }
    resolved_category = category or getattr(exc, "category", None)
    resolved_worker_name = worker_name or getattr(exc, "worker_name", None)
    merged_details: Dict[str, Any] = {}
    if getattr(exc, "details", None):
        merged_details.update(dict(getattr(exc, "details")))
    if details:
        merged_details.update(dict(details))
    if resolved_category:
        merged_details.setdefault("category", str(resolved_category))
    if resolved_worker_name:
        merged_details.setdefault("worker_name", str(resolved_worker_name))
    if merged_details:
        info["details"] = merged_details
    return info


def build_worker_failure(
    message: str,
    *,
    code: str,
    source: str,
    category: Optional[str] = None,
    worker_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> WorkerOutput:
    return WorkerOutput(
        success=False,
        error=message,
        error_info=build_worker_error_info(
            None,
            code=code,
            message=message,
            source=source,
            category=category,
            worker_name=worker_name,
            details=details,
        ),
    )


def normalize_worker_output(
    worker_output: WorkerOutput,
    *,
    category: Optional[str] = None,
    worker_name: Optional[str] = None,
) -> WorkerOutput:
    if worker_output.success:
        return worker_output
    if worker_output.error_info:
        info = dict(worker_output.error_info)
        details = dict(info.get("details") or {})
        if category:
            details.setdefault("category", str(category))
        if worker_name:
            details.setdefault("worker_name", str(worker_name))
        if details:
            info["details"] = details
        worker_output.error_info = info
        return worker_output
    worker_output.error_info = build_worker_error_info(
        None,
        message=worker_output.error or "Worker error",
        category=category,
        worker_name=worker_name,
    )
    return worker_output
