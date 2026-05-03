# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.core.dataclasses import ImageData, Settings
from lib.core.settings import load_app_settings
from lib.workers.base import WorkerInput
from lib.workers.llm_llama_cpp_local import LLMLlamaCppLocalWorker


def _make_temp_image() -> Path:
    root = Path(tempfile.mkdtemp(prefix="caption_llama_local_smoke_"))
    image_path = root / "llama_smoke.png"
    image = Image.new("RGB", (96, 96), (240, 240, 240))
    image.save(image_path)
    image.close()
    return image_path


def _build_worker(timeout_seconds: int) -> tuple[Settings, LLMLlamaCppLocalWorker]:
    cfg = dict(load_app_settings() or {})
    settings = Settings(**cfg)
    worker = LLMLlamaCppLocalWorker(
        {
            "base_url": settings.llama_cpp_base_url,
            "api_key": settings.llama_cpp_api_key,
            "model_name": settings.llama_cpp_model_alias,
            "model_path": settings.llama_cpp_model_path,
            "max_tokens": min(512, int(settings.llama_cpp_max_tokens or 512)),
            "n_ctx": int(settings.llama_cpp_n_ctx or 8192),
            "n_threads": int(settings.llama_cpp_n_threads or 0),
            "n_gpu_layers": int(settings.llama_cpp_n_gpu_layers or 99),
            "max_image_dim": 256,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": float(settings.llama_cpp_repeat_penalty or 1.0),
            "server_autostart": bool(settings.llama_cpp_server_autostart),
            "server_exe": settings.llama_cpp_server_exe,
            "server_workers": int(settings.llama_cpp_server_workers or 1),
            "server_start_timeout": int(timeout_seconds),
            "mmproj_path": settings.llama_cpp_mmproj_path,
            "enable_vision": bool(settings.llama_cpp_enable_vision),
        }
    )
    return settings, worker


def _run_optional_vision_probe(worker: LLMLlamaCppLocalWorker, settings: Settings, image_path: Path) -> dict:
    worker_input = WorkerInput(
        image=ImageData(path=str(image_path)),
        settings=settings,
        extra={
            "system_prompt": "You are a concise captioning engine.",
            "user_prompt": "Describe the image in one short English sentence.",
            "llm_input_repeat_count": 1,
            "thinking_mode": False,
        },
    )
    output = worker.process(worker_input)
    return {
        "success": bool(output.success),
        "error": output.error,
        "error_info": output.error_info,
        "result_text": output.result_text,
        "image_path": str(image_path),
    }


def run_llama_local_smoke(timeout_seconds: int = 20, image_path: Path | None = None) -> dict:
    settings, worker = _build_worker(timeout_seconds)
    endpoint = worker._resolve_chat_completions_url(worker.base_url)
    config_summary = worker._validate_runtime_config(endpoint)
    worker._ensure_server_ready(endpoint)
    worker._check_server_ready(endpoint, timeout_seconds=8.0)
    models_url = worker._resolve_models_url(endpoint)
    with urllib.request.urlopen(models_url, timeout=8.0) as resp:
        models_payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")

    result = {
        "success": True,
        "error": None,
        "error_info": None,
        "result_text": None,
        "image_path": str(image_path) if image_path else None,
        "model_path": settings.llama_cpp_model_path,
        "mmproj_path": settings.llama_cpp_mmproj_path,
        "server_exe": settings.llama_cpp_server_exe,
        "timeout_seconds": timeout_seconds,
        "models": models_payload,
        "config_summary": config_summary,
    }
    if image_path is not None:
        vision_result = _run_optional_vision_probe(worker, settings, image_path)
        result["vision_probe"] = vision_result
        result["success"] = bool(vision_result.get("success"))
        result["error"] = vision_result.get("error")
        result["error_info"] = vision_result.get("error_info")
        result["result_text"] = vision_result.get("result_text")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local llama.cpp smoke test.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    parser.add_argument("--timeout", type=int, default=20, help="Server startup timeout in seconds.")
    parser.add_argument("--image", type=str, default="", help="Optional image path for vision probe.")
    args = parser.parse_args()

    image_path = Path(str(args.image or "")).expanduser() if str(args.image or "").strip() else None
    if image_path is not None and not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")
    results = run_llama_local_smoke(
        timeout_seconds=max(5, int(args.timeout or 20)),
        image_path=image_path,
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results, ensure_ascii=False))
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
