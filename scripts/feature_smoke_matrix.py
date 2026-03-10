# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import cgi
import json
import shutil
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.runtime.backend_host import RuntimeBackendHost
from lib.runtime.runtime_regression import run_runtime_regression


class FeatureSkip(RuntimeError):
    pass


class _FakeSdServer:
    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[Thread] = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("fake sd-server is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "_FakeSdServer":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/v1/models":
                    owner.requests.append({"method": "GET", "path": self.path})
                    self._json({"data": [{"id": "fake-sd-server"}]})
                    return
                self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/v1/images/edits":
                    self._json({"error": "not found"}, status=404)
                    return

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                prompt = str(form.getvalue("prompt", "") or "")
                size = str(form.getvalue("size", "") or "")
                image_item = form["image"] if "image" in form else None
                if image_item is None or not getattr(image_item, "file", None):
                    self._json({"error": "missing image"}, status=400)
                    return

                from io import BytesIO
                from PIL import Image, ImageDraw

                image_bytes = image_item.file.read()

                with Image.open(BytesIO(image_bytes)) as uploaded:
                    edited = uploaded.convert("RGB")
                draw = ImageDraw.Draw(edited)
                width, height = edited.size
                draw.rectangle((0, 0, width - 1, height - 1), outline=(255, 0, 255), width=max(4, width // 64))
                draw.text((max(8, width // 20), max(8, height // 20)), "FAKE SD", fill=(255, 0, 255))
                output = BytesIO()
                edited.save(output, format="PNG")

                owner.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "prompt": prompt,
                        "size": size,
                        "image_bytes": len(image_bytes),
                    }
                )
                self._json({"data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}]})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None


def _wait_until(predicate: Callable[[], bool], timeout: float = 1800.0, interval: float = 0.5) -> None:
    deadline = time.time() + max(timeout, interval)
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("timed out while waiting for runtime task completion")


def _wait_task(host: RuntimeBackendHost, timeout: float = 1800.0) -> None:
    _wait_until(lambda: not host.is_task_running(), timeout=timeout)


def _latest_event(host: RuntimeBackendHost, event_name: str, *, task_name: Optional[str] = None, limit: int = 80) -> Optional[Dict[str, Any]]:
    events = list(host.get_runtime_events(limit=limit, profile="control") or [])
    for event in reversed(events):
        if str(event.get("name")) != event_name:
            continue
        payload = dict(event.get("payload", {}) or {})
        if task_name and str(payload.get("task_name") or "") != task_name:
            continue
        return {"name": event_name, "payload": payload, "timestamp": event.get("timestamp")}
    return None


def _make_text_fixture() -> Path:
    root = Path(tempfile.mkdtemp(prefix="caption_feature_smoke_text_"))
    (root / "a.png").write_bytes(b"png")
    (root / "a.txt").write_text("needle text", encoding="utf-8")
    (root / "b.png").write_bytes(b"png")
    (root / "b.txt").write_text("other", encoding="utf-8")
    return root


def _copy_image_fixture(image_path: Path, prefix: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=prefix))
    dst = root / image_path.name
    shutil.copy2(image_path, dst)
    return root


def _make_ocr_fixture() -> Path:
    from PIL import Image, ImageDraw

    root = Path(tempfile.mkdtemp(prefix="caption_feature_smoke_ocr_"))
    image_path = root / "ocr_sample.png"
    canvas = Image.new("RGBA", (768, 512), (255, 245, 230, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((40, 40, 728, 472), fill=(240, 220, 200, 255))
    draw.rectangle((80, 110, 688, 210), fill=(255, 255, 255, 255))
    draw.rectangle((80, 250, 688, 350), fill=(255, 255, 255, 255))
    draw.text((110, 135), "HELLO WORLD", fill=(0, 0, 0, 255))
    draw.text((110, 275), "MASK TEXT TEST", fill=(0, 0, 0, 255))
    canvas.save(image_path)
    return root


def _make_stroke_mask(target_image_path: Path) -> Path:
    from PIL import Image, ImageDraw

    with Image.open(target_image_path) as image:
        width, height = image.size
    mask_root = Path(tempfile.mkdtemp(prefix="caption_feature_smoke_mask_"))
    mask_path = mask_root / "stroke_mask.png"
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    left = max(0, width // 4)
    top = max(0, height // 4)
    right = min(width, left + max(32, width // 3))
    bottom = min(height, top + max(32, height // 6))
    draw.rectangle((left, top, right, bottom), fill=255)
    mask.save(mask_path)
    return mask_path


def _record_suite(results: Dict[str, Any], name: str, fn: Callable[[], Dict[str, Any]]) -> None:
    started_at = time.time()
    try:
        payload = fn()
        results["suites"].append(
            {
                "name": name,
                "status": "pass",
                "duration_seconds": round(time.time() - started_at, 3),
                "details": payload,
            }
        )
    except FeatureSkip as exc:
        results["suites"].append(
            {
                "name": name,
                "status": "skip",
                "duration_seconds": round(time.time() - started_at, 3),
                "reason": str(exc),
            }
        )
    except Exception as exc:
        results["suites"].append(
            {
                "name": name,
                "status": "fail",
                "duration_seconds": round(time.time() - started_at, 3),
                "error": repr(exc),
            }
        )


def _suite_runtime_regression(host: RuntimeBackendHost) -> Dict[str, Any]:
    return run_runtime_regression(host)


def _suite_worker_inventory(host: RuntimeBackendHost) -> Dict[str, Any]:
    scan_result = host.execute_command("workers.scan", access_mode="development")
    workers = host.get_runtime_workers()
    return {
        "scan_result": scan_result,
        "available_categories": list(workers.get("available_categories", []) or []),
        "categories": workers.get("categories", {}),
    }


def _suite_selection_editor(host: RuntimeBackendHost) -> Dict[str, Any]:
    root = _make_text_fixture()
    set_root = host.execute_command(
        "selection.set_root_dir",
        access_mode="development",
        dir_path=str(root),
    )
    filtered = host.execute_command(
        "selection.apply_filter",
        access_mode="development",
        query="needle",
        use_tags=False,
        use_text=True,
    )
    replaced = host.execute_command(
        "editor.find_replace",
        access_mode="development",
        find_text="needle",
        replace_text="replaced",
        scope_all=True,
        case_sensitive=False,
        regex=False,
    )
    cleared = host.execute_command("selection.clear_filter", access_mode="development")

    a_text = (root / "a.txt").read_text(encoding="utf-8")
    b_text = (root / "b.txt").read_text(encoding="utf-8")
    if "replaced" not in a_text:
        raise AssertionError("find/replace did not update a.txt")
    if "replaced" in b_text:
        raise AssertionError("find/replace should not modify unrelated file")

    return {
        "fixture_root": str(root),
        "set_root": set_root,
        "filtered": filtered,
        "replaced": replaced,
        "cleared": cleared,
        "a_txt": a_text,
        "b_txt": b_text,
    }


def _suite_tagger_llm(host: RuntimeBackendHost, image_path: Optional[Path]) -> Dict[str, Any]:
    if image_path is None:
        raise FeatureSkip("no --image supplied")
    if not image_path.exists():
        raise FeatureSkip(f"image not found: {image_path}")

    root = _copy_image_fixture(image_path, "caption_feature_smoke_tagger_llm_")
    image_copy = root / image_path.name

    host.update_runtime_settings(
        {
            "ui_language": "zh_tw",
            "worker_runtime_mode": "service",
            "llm_provider": "llm_llama_cpp_local",
            "llama_cpp_base_url": "http://127.0.0.1:8000/v1",
            "llama_cpp_n_ctx": 8192,
            "llama_cpp_n_threads": 0,
            "llama_cpp_server_workers": 1,
            "llama_cpp_server_autostart": True,
            "llama_cpp_server_exe": r"E:\caption--\tasks\runtime\llama-b8189\llama-server.exe",
            "startup_defer_worker_scan": True,
        }
    )
    host.execute_command("selection.set_root_dir", access_mode="development", dir_path=str(root))

    tagger_result = host.execute_command("action.run_tagger_current", access_mode="development")
    _wait_task(host, timeout=1800.0)

    llm_result = host.execute_command(
        "action.run_llm_current",
        access_mode="development",
        confirm_missing_tags=True,
    )
    _wait_task(host, timeout=1800.0)

    txt_path = image_copy.with_suffix(".txt")
    json_path = image_copy.with_suffix(".json")
    if not txt_path.exists():
        raise AssertionError("tagger/llm suite did not produce txt sidecar")
    if not json_path.exists():
        raise AssertionError("tagger/llm suite did not produce json sidecar")

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not str(json_payload.get("tagger_tags", "")).strip():
        raise AssertionError("missing tagger_tags in json sidecar")
    if not str(json_payload.get("llm_result", "")).strip():
        raise AssertionError("missing llm_result in json sidecar")

    last_events = host.get_runtime_events(limit=20, profile="control")
    return {
        "fixture_root": str(root),
        "image_copy": str(image_copy),
        "tagger_result": tagger_result,
        "llm_result": llm_result,
        "txt_preview": txt_path.read_text(encoding="utf-8")[:1200],
        "json_preview": {
            "tagger_tags": json_payload.get("tagger_tags"),
            "llm_result": json_payload.get("llm_result"),
            "nl_pages_count": len(list(json_payload.get("nl_pages", []) or [])),
        },
        "last_events": last_events,
    }


def _suite_delete_current(host: RuntimeBackendHost, image_path: Optional[Path]) -> Dict[str, Any]:
    if image_path is None:
        raise FeatureSkip("no --image supplied")
    if not image_path.exists():
        raise FeatureSkip(f"image not found: {image_path}")

    root = _copy_image_fixture(image_path, "caption_feature_smoke_delete_")
    image_copy = root / image_path.name
    txt_path = image_copy.with_suffix(".txt")
    json_path = image_copy.with_suffix(".json")
    txt_path.write_text("sample text", encoding="utf-8")
    json_path.write_text(json.dumps({"demo": True}, ensure_ascii=False, indent=2), encoding="utf-8")

    host.execute_command("selection.set_root_dir", access_mode="development", dir_path=str(root))
    delete_result = host.execute_command(
        "image.delete_current",
        access_mode="development",
        confirm=True,
    )

    no_used_dir = root / "no_used"
    moved_image = no_used_dir / image_copy.name
    moved_txt = no_used_dir / txt_path.name
    moved_json = no_used_dir / json_path.name
    if not moved_image.exists():
        raise AssertionError("image file was not moved into no_used")
    if not moved_txt.exists():
        raise AssertionError("txt sidecar was not moved into no_used")
    if not moved_json.exists():
        raise AssertionError("json sidecar was not moved into no_used")

    return {
        "fixture_root": str(root),
        "delete_result": delete_result,
        "moved_files": [str(moved_image), str(moved_txt), str(moved_json)],
    }


def _suite_heavy_feature_preflight(host: RuntimeBackendHost) -> Dict[str, Any]:
    capabilities = host.get_runtime_capabilities(mode="development")
    workers = capabilities.get("workers", {})
    categories = dict(workers.get("categories", {}) or {})

    checks: Dict[str, Dict[str, Any]] = {}
    for feature_name, category, note in (
        ("image_process", "IMAGE_PROCESS", "requires real image-edit fixture and sd-server path"),
        ("unmask", "UNMASK", "requires foreground/background fixture"),
        ("mask_text", "MASK_TEXT", "requires OCR text fixture"),
        ("restore", "RESTORE", "requires raw backup fixture"),
    ):
        available = bool(list(categories.get(category, []) or []))
        checks[feature_name] = {
            "worker_category": category,
            "worker_available": available,
            "status": "ready" if available else "blocked",
            "note": note,
        }
    return checks


def _suite_unmask(host: RuntimeBackendHost, image_path: Optional[Path]) -> Dict[str, Any]:
    if image_path is None:
        raise FeatureSkip("no --image supplied")
    if not image_path.exists():
        raise FeatureSkip(f"image not found: {image_path}")

    root = _copy_image_fixture(image_path, "caption_feature_smoke_unmask_")
    image_copy = root / image_path.name
    host.execute_command("selection.set_root_dir", access_mode="development", dir_path=str(root))
    start_result = host.execute_command("action.run_unmask_current", access_mode="development")
    _wait_task(host, timeout=1800.0)

    image_done = _latest_event(host, "task.image_done", task_name="unmask")
    if image_done is None:
        raise AssertionError("missing task.image_done event for unmask")

    payload = dict(image_done.get("payload", {}) or {})
    if not bool(payload.get("success")):
        error_info = dict(payload.get("error_info", {}) or {})
        blocker = error_info.get("message") or payload.get("error") or "unmask failed"
        raise AssertionError(str(blocker))

    json_path = image_copy.with_suffix(".json")
    if not json_path.exists():
        raise AssertionError("unmask suite did not produce json sidecar")
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    if not bool(sidecar.get("masked_background")):
        raise AssertionError("unmask suite did not mark masked_background")
    raw_backup_path = sidecar.get("raw_backup_path") or sidecar.get("raw_image_rel_path")
    if not str(raw_backup_path or "").strip():
        raise AssertionError("unmask suite missing raw backup path")

    return {
        "fixture_root": str(root),
        "image_copy": str(image_copy),
        "start_result": start_result,
        "image_done": image_done,
        "selection_after": host.get_runtime_state().get("selection", {}),
        "sidecar_preview": {
            "masked_background": sidecar.get("masked_background"),
            "foreground_ratio": sidecar.get("foreground_ratio"),
            "raw_backup_path": raw_backup_path,
        },
    }


def _suite_mask_text_restore(host: RuntimeBackendHost) -> Dict[str, Any]:
    root = _make_ocr_fixture()
    image_path = root / "ocr_sample.png"
    host.execute_command("selection.set_root_dir", access_mode="development", dir_path=str(root))

    mask_result = host.execute_command("action.run_mask_text_current", access_mode="development")
    _wait_task(host, timeout=1800.0)
    selection_after_mask = dict(host.get_runtime_state().get("selection", {}) or {})
    masked_path = Path(str(selection_after_mask.get("current_image_path") or ""))
    if masked_path.suffix.lower() != ".webp":
        raise AssertionError("mask_text did not switch current image to .webp result")

    masked_json = root / "ocr_sample.json"
    if not masked_json.exists():
        raise AssertionError("mask_text suite missing json sidecar")
    masked_sidecar = json.loads(masked_json.read_text(encoding="utf-8"))
    if not bool(masked_sidecar.get("masked_text")):
        raise AssertionError("mask_text suite did not set masked_text")

    restore_result = host.execute_command("action.run_restore_current", access_mode="development")
    _wait_task(host, timeout=1800.0)
    selection_after_restore = dict(host.get_runtime_state().get("selection", {}) or {})
    restored_path = Path(str(selection_after_restore.get("current_image_path") or ""))
    if restored_path.suffix.lower() != ".png":
        raise AssertionError("restore did not switch current image back to original .png")
    if (root / "ocr_sample.webp").exists():
        raise AssertionError("restore should remove processed .webp for OCR fixture")

    restored_sidecar = json.loads(masked_json.read_text(encoding="utf-8"))
    if bool(restored_sidecar.get("masked_text")) or bool(restored_sidecar.get("masked_background")):
        raise AssertionError("restore did not clear masked flags")

    return {
        "fixture_root": str(root),
        "mask_result": mask_result,
        "restore_result": restore_result,
        "selection_after_mask": selection_after_mask,
        "selection_after_restore": selection_after_restore,
        "sidecar_after_restore": {
            "masked_text": restored_sidecar.get("masked_text"),
            "masked_background": restored_sidecar.get("masked_background"),
            "raw_backup_path": restored_sidecar.get("raw_backup_path") or restored_sidecar.get("raw_image_rel_path"),
        },
        "files": sorted(path.name for path in root.iterdir()),
    }


def _suite_stroke_restore(host: RuntimeBackendHost, image_path: Optional[Path]) -> Dict[str, Any]:
    if image_path is None:
        raise FeatureSkip("no --image supplied")
    if not image_path.exists():
        raise FeatureSkip(f"image not found: {image_path}")

    root = _copy_image_fixture(image_path, "caption_feature_smoke_stroke_")
    image_copy = root / image_path.name
    mask_path = _make_stroke_mask(image_copy)
    host.execute_command("selection.set_root_dir", access_mode="development", dir_path=str(root))

    stroke_result = host.execute_command(
        "action.run_stroke_eraser_current",
        access_mode="development",
        mask_path=str(mask_path),
    )
    if not bool(stroke_result.get("completed")):
        raise AssertionError(f"stroke eraser did not complete: {stroke_result}")
    if not (root / "raw_image" / image_copy.name).exists():
        raise AssertionError("stroke eraser did not create raw_image backup")
    if not (root / "unmask").exists():
        raise AssertionError("stroke eraser did not create unmask backup folder")

    restore_result = host.execute_command("action.run_restore_current", access_mode="development")
    _wait_task(host, timeout=1800.0)
    selection_after_restore = dict(host.get_runtime_state().get("selection", {}) or {})
    restored_path = Path(str(selection_after_restore.get("current_image_path") or ""))
    if restored_path.name != image_copy.name:
        raise AssertionError("stroke restore did not keep/restore expected current image path")

    return {
        "fixture_root": str(root),
        "mask_path": str(mask_path),
        "stroke_result": stroke_result,
        "restore_result": restore_result,
        "selection_after_restore": selection_after_restore,
        "root_files": sorted(path.name for path in root.iterdir()),
        "unmask_files": sorted(path.name for path in (root / "unmask").iterdir()),
    }


def _suite_image_process(host: RuntimeBackendHost, image_path: Optional[Path]) -> Dict[str, Any]:
    if image_path is None:
        raise FeatureSkip("no --image supplied")
    if not image_path.exists():
        raise FeatureSkip(f"image not found: {image_path}")

    root = _copy_image_fixture(image_path, "caption_feature_smoke_image_process_")
    image_copy = root / image_path.name
    original_bytes = image_copy.read_bytes()
    with _FakeSdServer() as fake_server:
        host.update_runtime_settings(
            {
                "image_process_base_url": fake_server.base_url,
                "image_process_server_autostart": False,
                "image_process_server_exe": "",
            }
        )
        host.execute_command("selection.set_root_dir", access_mode="development", dir_path=str(root))
        start_result = host.execute_command("action.run_image_process_current", access_mode="development")
        _wait_task(host, timeout=600.0)

        image_done = _latest_event(host, "task.image_done", task_name="image_process", limit=120)
        if image_done is None:
            raise AssertionError("missing task.image_done event for image_process")

        payload = dict(image_done.get("payload", {}) or {})
        if not bool(payload.get("success")):
            error_info = dict(payload.get("error_info", {}) or {})
            raise AssertionError(error_info.get("message") or payload.get("error") or "image_process failed")

        new_bytes = image_copy.read_bytes()
        if new_bytes == original_bytes:
            raise AssertionError("image_process did not modify the image output")

        return {
            "fixture_root": str(root),
            "image_copy": str(image_copy),
            "start_result": start_result,
            "image_done": image_done,
            "selection_after": host.get_runtime_state().get("selection", {}),
            "fake_server_requests": fake_server.requests,
        }


def run_feature_smoke_matrix(image_path: Optional[Path]) -> Dict[str, Any]:
    host = RuntimeBackendHost()
    results: Dict[str, Any] = {
        "requested_image": str(image_path) if image_path else None,
        "suites": [],
    }

    _record_suite(results, "runtime_regression", lambda: _suite_runtime_regression(host))
    _record_suite(results, "worker_inventory", lambda: _suite_worker_inventory(host))
    _record_suite(results, "selection_editor", lambda: _suite_selection_editor(host))
    _record_suite(results, "tagger_llm_real_image", lambda: _suite_tagger_llm(host, image_path))
    _record_suite(results, "delete_current_bundle", lambda: _suite_delete_current(host, image_path))
    _record_suite(results, "heavy_feature_preflight", lambda: _suite_heavy_feature_preflight(host))
    _record_suite(results, "unmask_real_image", lambda: _suite_unmask(host, image_path))
    _record_suite(results, "mask_text_restore_fixture", lambda: _suite_mask_text_restore(host))
    _record_suite(results, "stroke_restore_real_image", lambda: _suite_stroke_restore(host, image_path))
    _record_suite(results, "image_process_real_image", lambda: _suite_image_process(host, image_path))

    summary = {"pass": 0, "fail": 0, "skip": 0}
    for suite in results["suites"]:
        status = str(suite.get("status"))
        summary[status] = summary.get(status, 0) + 1
    results["summary"] = summary
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run feature-by-feature runtime smoke suites.")
    parser.add_argument("--image", type=str, default="", help="Real image fixture for tagger/llm/delete suites.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser() if str(args.image or "").strip() else None
    results = run_feature_smoke_matrix(image_path)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for suite in results["suites"]:
            line = f"[{suite['status']}] {suite['name']}"
            if suite["status"] == "skip":
                line += f": {suite.get('reason', '')}"
            elif suite["status"] == "fail":
                line += f": {suite.get('error', '')}"
            print(line)
        print(json.dumps(results["summary"], ensure_ascii=False))

    return 1 if results["summary"].get("fail", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
