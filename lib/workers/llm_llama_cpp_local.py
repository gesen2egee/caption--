# -*- coding: utf-8 -*-
"""
Local llama.cpp worker via llama-server OpenAI-compatible API.

Important:
- This worker uses llama-server HTTP API.
- It does NOT use llama-cpp-python's in-process Llama(model_path=...) loading,
  which currently fails for some qwen35 GGUF builds.
"""
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.workers.errors import WorkerError, build_worker_error_info


HF_BLOB_OR_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/"
    r"(?P<repo>[^/]+/[^/]+)/"
    r"(?:(?:blob)|(?:resolve))/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)$"
)
DEFAULT_MMPROJ_URL = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/mmproj-F32.gguf"
MIN_SERVER_BUILD_QWEN35_VL = 8189
MIN_SERVER_BUILD_GEMMA4 = 8828


class LLMLlamaCppLocalWorker(BaseWorker):
    category = "LLM"
    display_name = "LLaMA.cpp Local (GGUF)"
    description = "Run local GGUF models via llama-server OpenAI-compatible API"
    default_config = {
        "model_path": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/Qwen3.5-2B-BF16.gguf",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "",
        "model_name": "qwen35-vl-gguf",
        "max_tokens": 81920,
        "n_ctx": 8192,
        "n_threads": 0,
        "max_image_dim": 1024,
        "use_gray_mask": True,
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "server_autostart": True,
        "server_exe": "",
        "server_workers": 1,
        "server_start_timeout": 900,
        "n_gpu_layers": 99,
        "mmproj_path": DEFAULT_MMPROJ_URL,
        "enable_vision": True,
    }

    _server_proc = None
    _server_key = None
    _server_lock = Lock()
    _server_log_path = None
    _server_launch_summary = None

    def __init__(self, config: Dict = None):
        super().__init__(config)

        self.model_path = str(self.config.get("model_path", self.default_config["model_path"])).strip()
        self.base_url = str(self.config.get("base_url", self.default_config["base_url"])).strip()
        self.api_key = str(self.config.get("api_key", self.default_config["api_key"])).strip()
        self.model_name = str(self.config.get("model_name", self.default_config["model_name"])).strip()

        self.max_tokens = int(self.config.get("max_tokens", self.default_config["max_tokens"]))
        self.n_ctx = int(self.config.get("n_ctx", self.default_config["n_ctx"]))
        self.n_threads = int(self.config.get("n_threads", self.default_config["n_threads"]))
        self.max_image_dim = int(self.config.get("max_image_dim", self.default_config["max_image_dim"]))
        self.use_gray_mask = bool(self.config.get("use_gray_mask", self.default_config["use_gray_mask"]))

        self.temperature = float(self.config.get("temperature", self.default_config["temperature"]))
        self.top_p = float(self.config.get("top_p", self.default_config["top_p"]))
        self.top_k = int(self.config.get("top_k", self.default_config["top_k"]))
        self.min_p = float(self.config.get("min_p", self.default_config["min_p"]))
        self.presence_penalty = float(self.config.get("presence_penalty", self.default_config["presence_penalty"]))

        # Backward-compat: llm_task previously passed repeat_penalty.
        self.repetition_penalty = float(
            self.config.get(
                "repetition_penalty",
                self.config.get("repeat_penalty", self.default_config["repetition_penalty"]),
            )
        )

        self.server_autostart = bool(self.config.get("server_autostart", self.default_config["server_autostart"]))
        self.server_exe = str(self.config.get("server_exe", self.default_config["server_exe"])).strip()
        self.server_workers = int(self.config.get("server_workers", self.default_config["server_workers"]))
        self.server_start_timeout = int(
            self.config.get("server_start_timeout", self.default_config["server_start_timeout"])
        )
        self.n_gpu_layers = int(self.config.get("n_gpu_layers", self.default_config["n_gpu_layers"]))
        self.mmproj_path = str(self.config.get("mmproj_path", self.default_config["mmproj_path"])).strip()
        self.enable_vision = bool(self.config.get("enable_vision", self.default_config["enable_vision"]))
        if self.enable_vision and not self.mmproj_path:
            self.mmproj_path = DEFAULT_MMPROJ_URL

    @property
    def name(self) -> str:
        return "llm_llama_cpp_local"

    @classmethod
    def is_available(cls) -> bool:
        # Endpoint mode: always available in UI. Runtime gives actionable error
        # if llama-server is missing/unreachable.
        return True

    @staticmethod
    def _resolve_chat_completions_url(base_url: str) -> str:
        url = (base_url or "").strip().rstrip("/")
        if not url:
            raise RuntimeError("llama-server base URL is empty")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    @staticmethod
    def _resolve_models_url(chat_completions_url: str) -> str:
        if chat_completions_url.endswith("/chat/completions"):
            return chat_completions_url[: -len("/chat/completions")] + "/models"
        return chat_completions_url.rstrip("/") + "/models"

    @staticmethod
    def _extract_text_content(content_obj) -> str:
        if isinstance(content_obj, str):
            return content_obj
        if isinstance(content_obj, list):
            chunks = []
            for item in content_obj:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return " ".join(chunks).strip()
        return ""

    def _encode_image_to_data_url(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            if img.mode == "RGBA":
                if self.use_gray_mask:
                    bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
                    bg.paste(img, mask=img.split()[3])
                    image = bg.convert("RGB")
                else:
                    image = img.convert("RGB")
            else:
                image = img.convert("RGB")

        w, h = image.size
        if max(w, h) > self.max_image_dim > 0:
            scale = self.max_image_dim / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        image.close()
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _parse_hf_source(model_source: str) -> Tuple[Optional[str], Optional[str]]:
        src = (model_source or "").strip()
        if not src:
            return None, None

        parsed = HF_BLOB_OR_RESOLVE_RE.match(src)
        if parsed:
            return parsed.group("repo"), parsed.group("filename")

        parts = src.split("/")
        if len(parts) >= 3 and src.lower().endswith(".gguf"):
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            return repo_id, filename

        if len(parts) == 2:
            return src, None

        return None, None

    @staticmethod
    def _normalize_hf_download_url(model_source: str) -> str:
        src = (model_source or "").strip()
        if not src.lower().startswith(("http://", "https://")):
            return src
        parsed = HF_BLOB_OR_RESOLVE_RE.match(src)
        if not parsed:
            return src
        repo = parsed.group("repo")
        revision = parsed.group("revision")
        filename = parsed.group("filename")
        return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"

    @staticmethod
    def _parse_port_from_endpoint(endpoint: str) -> int:
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        if parsed.port:
            return int(parsed.port)
        if parsed.scheme == "https":
            return 443
        return 80

    def _resolve_llama_server_executable(self) -> str:
        if self.server_exe and os.path.exists(self.server_exe):
            return self.server_exe

        bundled_candidates = [
            os.path.join(os.getcwd(), "tasks", "runtime", "llama-b8848", "llama-server.exe"),
            os.path.join(os.getcwd(), "tasks", "runtime", "llama-b8189", "llama-server.exe"),
        ]
        for bundled in bundled_candidates:
            if os.path.exists(bundled):
                return bundled

        found = shutil.which("llama-server")
        if found:
            return found

        fallback = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "llama-server.exe",
        )
        if os.path.exists(fallback):
            return fallback

        raise RuntimeError(
            "找不到 llama-server。請先安裝支援目前模型的新版 llama-server，或在設定中指定 llama-server.exe 路徑。"
        )

    @staticmethod
    def _summarize_model_source(model_source: str) -> Dict[str, Optional[str]]:
        src = (model_source or "").strip()
        if not src:
            return {"raw": "", "source_type": "empty", "repo": None, "file": None, "size_label": None}
        if os.path.exists(src):
            text = Path(src).name
            source_type = "local_file"
            repo = None
            filename = Path(src).name
        else:
            repo, filename = LLMLlamaCppLocalWorker._parse_hf_source(src)
            text = src
            source_type = "hf" if repo else "unknown"
        # Only treat standalone size tokens as size labels.
        # This avoids false positives such as "Gemma-4-E4B" -> "4b".
        size_match = re.search(r"(?<![0-9a-z])(\d{1,3}(?:\.\d+)?b)(?![0-9a-z])", text.lower())
        return {
            "raw": src,
            "source_type": source_type,
            "repo": repo,
            "file": filename,
            "size_label": size_match.group(1) if size_match else None,
        }

    @staticmethod
    def _is_port_open(endpoint: str) -> bool:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((host, port)) == 0

    @classmethod
    def _tail_server_log(cls, max_lines: int = 40) -> list[str]:
        log_path = str(cls._server_log_path or "").strip()
        if not log_path or not os.path.exists(log_path):
            return []
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        return text.splitlines()[-max_lines:]

    def _build_config_summary(self, endpoint: str) -> Dict[str, object]:
        model_summary = self._summarize_model_source(self.model_path)
        mmproj_summary = self._summarize_model_source(self.mmproj_path)
        model_family = self._infer_model_family(model_summary)
        return {
            "endpoint": endpoint,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "model_source": model_summary,
            "mmproj_source": mmproj_summary,
            "model_family": model_family,
            "required_server_build": self._required_server_build(model_family),
            "enable_vision": bool(self.enable_vision),
            "n_ctx": int(self.n_ctx),
            "effective_server_ctx": int(self._effective_server_ctx_size()),
            "server_workers": int(self.server_workers),
            "server_autostart": bool(self.server_autostart),
            "server_exe": self.server_exe,
            "n_gpu_layers": int(self.n_gpu_layers),
            "n_threads": int(self.n_threads),
            "normalized_model_source": self._normalize_hf_download_url(self.model_path),
            "normalized_mmproj_source": self._normalize_hf_download_url(self.mmproj_path),
        }

    @staticmethod
    def _size_label_from_param_count(n_params: object) -> Optional[str]:
        try:
            value = int(n_params)
        except Exception:
            return None
        if value <= 0:
            return None
        billions = max(1, round(value / 1_000_000_000))
        return f"{billions}b"

    @staticmethod
    def _infer_model_family(model_summary: Dict[str, Optional[str]]) -> Optional[str]:
        raw = str(model_summary.get("raw") or "")
        repo = str(model_summary.get("repo") or "")
        file_name = str(model_summary.get("file") or "")
        haystack = " ".join([raw, repo, file_name]).lower()
        if "gemma-4" in haystack or "gemma4" in haystack:
            return "gemma4"
        if "qwen3.5" in haystack or "qwen3-5" in haystack or "qwen35" in haystack:
            return "qwen35-vl"
        return None

    @staticmethod
    def _required_server_build(model_family: Optional[str]) -> Optional[int]:
        if model_family == "gemma4":
            return MIN_SERVER_BUILD_GEMMA4
        if model_family == "qwen35-vl":
            return MIN_SERVER_BUILD_QWEN35_VL
        return None

    def _extract_server_state(self, models_payload: Dict[str, object]) -> Dict[str, object]:
        entries = list(models_payload.get("data", []) or [])
        aliases: list[str] = []
        capabilities: list[str] = []
        size_labels: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if item_id:
                aliases.append(item_id)
            for alias in list(item.get("aliases", []) or []):
                alias_text = str(alias or "").strip()
                if alias_text:
                    aliases.append(alias_text)
            meta = dict(item.get("meta", {}) or {})
            size_label = self._size_label_from_param_count(meta.get("n_params"))
            if size_label:
                size_labels.append(size_label)
        legacy_entries = list(models_payload.get("models", []) or [])
        for item in legacy_entries:
            if not isinstance(item, dict):
                continue
            capabilities.extend([str(cap or "").strip() for cap in list(item.get("capabilities", []) or []) if str(cap or "").strip()])
        dedup_aliases = sorted({value for value in aliases if value})
        dedup_capabilities = sorted({value for value in capabilities if value})
        dedup_sizes = sorted({value for value in size_labels if value})
        return {
            "aliases": dedup_aliases,
            "capabilities": dedup_capabilities,
            "size_labels": dedup_sizes,
        }

    @staticmethod
    def _find_listening_pid_for_port(port: int) -> Optional[int]:
        try:
            proc = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            return None
        output = (proc.stdout or "").splitlines()
        for line in output:
            text = " ".join(str(line or "").split())
            if not text or "LISTENING" not in text.upper():
                continue
            parts = text.split()
            if len(parts) < 5:
                continue
            local_address = parts[1]
            state = parts[3].upper()
            pid_text = parts[4]
            if state != "LISTENING":
                continue
            if local_address.endswith(f":{port}"):
                try:
                    return int(pid_text)
                except Exception:
                    return None
        return None

    @staticmethod
    def _get_process_details(pid: int) -> Dict[str, object]:
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        f"$p = Get-Process -Id {int(pid)} -ErrorAction Stop; "
                        "$obj = [PSCustomObject]@{"
                        "Id=$p.Id;ProcessName=$p.ProcessName;Path=$p.Path}; "
                        "$obj | ConvertTo-Json -Compress"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            raw = str(proc.stdout or "").strip()
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"Id": int(pid), "ProcessName": None, "Path": None}

    @classmethod
    def _stop_external_server_on_port(cls, endpoint: str, *, reason: str = "") -> Dict[str, object]:
        port = cls._parse_port_from_endpoint(endpoint)
        pid = cls._find_listening_pid_for_port(port)
        if pid is None:
            return {"stopped": False, "reason": "port_not_listening", "port": port}
        managed_pid = None if cls._server_proc is None else cls._server_proc.pid
        if managed_pid is not None and pid == managed_pid:
            cls._stop_managed_server()
            return {"stopped": True, "reason": "managed_server_restarted", "port": port, "pid": pid}

        details = cls._get_process_details(pid)
        process_name = str(details.get("ProcessName") or "").lower()
        process_path = str(details.get("Path") or "").lower()
        if "llama-server" not in process_name and not process_path.endswith("llama-server.exe"):
            return {
                "stopped": False,
                "reason": "port_busy_non_llama_server",
                "port": port,
                "pid": pid,
                "process": details,
                "trigger_reason": reason,
            }
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception:
            return {
                "stopped": False,
                "reason": "taskkill_failed",
                "port": port,
                "pid": pid,
                "process": details,
                "trigger_reason": reason,
            }
        time.sleep(1.0)
        return {
            "stopped": True,
            "reason": "external_llama_server_stopped",
            "port": port,
            "pid": pid,
            "process": details,
            "trigger_reason": reason,
        }

    def _validate_runtime_config(self, endpoint: str) -> Dict[str, object]:
        summary = self._build_config_summary(endpoint)
        model_summary = dict(summary["model_source"])
        mmproj_summary = dict(summary["mmproj_source"])

        if not self.base_url.strip():
            raise WorkerError(
                "缺少 llama-server base URL",
                code="llama_model_config_invalid",
                source="worker.llama_cpp_local",
                category=self.category,
                worker_name=self.name,
                details=summary,
            )

        if self.server_autostart:
            try:
                resolved_exe = self._resolve_llama_server_executable()
            except Exception as exc:
                raise WorkerError(
                    str(exc),
                    code="llama_server_missing",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=summary,
                ) from exc
            summary["resolved_server_exe"] = resolved_exe
            required_build = summary.get("required_server_build")
            build_no = self._get_llama_server_build(resolved_exe)
            if isinstance(required_build, int) and build_no is not None and build_no < required_build:
                model_family = str(summary.get("model_family") or "unknown model").strip()
                raise WorkerError(
                    (
                        f"目前的 {model_family} 模型需要 llama-server b{required_build}+，"
                        f"但你目前使用的是 b{build_no}。"
                        " 這版太舊，還沒支援這個模型架構。"
                        " 請改用更新的 llama-server 後再試。"
                    ),
                    code="llama_server_build_too_old",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=summary,
                )

        model_source_type = str(model_summary.get("source_type") or "")
        if model_source_type not in {"local_file", "hf"}:
            raise WorkerError(
                "llama_cpp_model_path 無法辨識，請提供本機 .gguf 檔案或 Hugging Face repo/file。",
                code="llama_model_config_invalid",
                source="worker.llama_cpp_local",
                category=self.category,
                worker_name=self.name,
                details=summary,
            )

        if self.enable_vision:
            if not self.mmproj_path.strip():
                raise WorkerError(
                    "目前啟用了 vision，但沒有提供 mmproj。",
                    code="llama_model_config_invalid",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=summary,
                )
            mmproj_source_type = str(mmproj_summary.get("source_type") or "")
            if mmproj_source_type not in {"local_file", "hf"}:
                raise WorkerError(
                    "llama_cpp_mmproj_path 無法辨識，請提供本機 mmproj 或 Hugging Face repo/file。",
                    code="llama_model_config_invalid",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=summary,
                )
            model_size = str(model_summary.get("size_label") or "").lower()
            mmproj_size = str(mmproj_summary.get("size_label") or "").lower()
            if model_size and mmproj_size and model_size != mmproj_size:
                raise WorkerError(
                    (
                        "目前的 llama 模型與 mmproj 尺寸不一致，"
                        f"model={model_size}, mmproj={mmproj_size}。"
                        " 請改用一致的配對。"
                    ),
                    code="llama_mmproj_mismatch",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=summary,
                )

        return summary

    @staticmethod
    def _get_llama_server_build(llama_server_exe: str) -> Optional[int]:
        try:
            proc = subprocess.run(
                [llama_server_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            return None
        version_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"version:\s*(\d+)", version_text)
        if not m:
            return None
        return int(m.group(1))

    def _check_server_ready(self, endpoint: str, timeout_seconds: float = 8.0):
        models_url = self._resolve_models_url(endpoint)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(models_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload: Dict[str, object] = {}
        try:
            payload = json.loads(body or "{}")
        except Exception:
            payload = {}
        server_state = self._extract_server_state(payload)
        model_ids = list(server_state.get("aliases", []) or [])
        if model_ids and self.model_name and self.model_name not in model_ids:
            raise RuntimeError(
                f"llama-server 已啟動，但 /models 中沒有別名 {self.model_name}；目前可用：{model_ids}"
            )
        if self.enable_vision:
            capabilities = [str(value or "").lower() for value in list(server_state.get("capabilities", []) or [])]
            if capabilities and "multimodal" not in capabilities:
                raise RuntimeError(
                    f"llama-server 已啟動，但目前模型不支援 vision/multimodal；capabilities={capabilities}"
                )
        expected_size = str(self._summarize_model_source(self.model_path).get("size_label") or "").lower()
        actual_sizes = [str(value or "").lower() for value in list(server_state.get("size_labels", []) or []) if str(value or "").strip()]
        if expected_size and actual_sizes and expected_size not in actual_sizes:
            raise RuntimeError(
                f"llama-server 模型尺寸與設定不一致；expected={expected_size}, actual={actual_sizes}"
            )

        completion_payload = {
            "model": self.model_name or (model_ids[0] if model_ids else ""),
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
        }
        completion_req = urllib.request.Request(
            endpoint,
            data=json.dumps(completion_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(completion_req, timeout=timeout_seconds) as resp:
            completion_body = resp.read().decode("utf-8", errors="replace")
        completion_json = json.loads(completion_body or "{}")
        if not list(completion_json.get("choices", []) or []):
            raise RuntimeError("llama-server chat/completions 已回應，但沒有 choices")
        return server_state

    def _effective_server_ctx_size(self) -> int:
        # llama-server splits KV cache across parallel slots. Treat configured
        # n_ctx as desired per-request context and size the server cache so
        # each slot still gets that amount.
        per_request_ctx = max(512, int(self.n_ctx or 0))
        slot_count = max(1, int(self.server_workers or 0))
        return per_request_ctx * slot_count

    @classmethod
    def _stop_managed_server(cls):
        proc = cls._server_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
        cls._server_proc = None
        cls._server_key = None
        cls._server_log_path = None
        cls._server_launch_summary = None

    def _build_server_command(self, endpoint: str, resolved_exe: str) -> Tuple[list[str], Dict[str, object]]:
        port = self._parse_port_from_endpoint(endpoint)
        cmd = [
            resolved_exe,
            "--alias",
            self.model_name or "qwen35-vl-gguf",
            "--port",
            str(port),
            "-c",
            str(self._effective_server_ctx_size()),
            "-ngl",
            str(self.n_gpu_layers),
        ]
        if self.n_threads > 0:
            cmd.extend(["-t", str(self.n_threads)])

        src = self.model_path
        if src and os.path.exists(src):
            cmd.extend(["-m", src])
        else:
            repo_id, hf_file = self._parse_hf_source(src)
            if repo_id:
                cmd.extend(["--hf-repo", repo_id])
                if hf_file:
                    cmd.extend(["--hf-file", hf_file])

        if self.server_workers > 0:
            cmd.extend(["-np", str(self.server_workers)])
        if self.enable_vision and self.mmproj_path:
            mmproj = self._normalize_hf_download_url(self.mmproj_path.strip())
            if mmproj.lower().startswith(("http://", "https://")):
                cmd.extend(["--mmproj-url", mmproj])
            else:
                cmd.extend(["--mmproj", mmproj])

        launch_summary = self._build_config_summary(endpoint)
        launch_summary.update(
            {
                "resolved_server_exe": resolved_exe,
                "port": port,
                "command": list(cmd),
            }
        )
        return cmd, launch_summary

    def _start_server_process(self, cmd: list[str], launch_summary: Dict[str, object]) -> subprocess.Popen:
        log_dir = Path(os.getcwd()) / "output" / "runtime-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"llama-server-{int(time.time())}.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(json.dumps({"launch_summary": launch_summary}, ensure_ascii=False, indent=2))
            handle.write("\n\n")
        log_handle = log_path.open("a", encoding="utf-8", errors="replace", buffering=1)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
        finally:
            log_handle.close()
        self.__class__._server_log_path = str(log_path)
        self.__class__._server_launch_summary = dict(launch_summary)
        return proc

    def _ensure_server_ready(self, endpoint: str):
        config_summary = self._validate_runtime_config(endpoint)
        key = (
            endpoint,
            self.model_path,
            self.model_name,
            self.n_ctx,
            self.n_threads,
            self.n_gpu_layers,
            self.server_workers,
            self.mmproj_path,
            self.enable_vision,
        )

        with self._server_lock:
            proc = self.__class__._server_proc
            if proc is not None:
                if proc.poll() is not None:
                    self.__class__._server_proc = None
                    self.__class__._server_key = None
                elif self.__class__._server_key == key:
                    self._wait_server_ready(endpoint, self.server_start_timeout)
                    return
                else:
                    # Managed llama-server exists but config changed; restart it
                    # so model_path / sampling-related server args actually apply.
                    self.__class__._stop_managed_server()

            # Reuse an already-running external llama-server if endpoint is reachable.
            # Note: for external processes we cannot reliably infer the loaded model path.
            try:
                self._check_server_ready(endpoint, timeout_seconds=6.0)
                return
            except Exception as exc:
                stop_result = self._stop_external_server_on_port(endpoint, reason=str(exc))
                config_summary["external_server_recovery"] = stop_result
                if not bool(stop_result.get("stopped")) and stop_result.get("reason") == "port_busy_non_llama_server":
                    raise WorkerError(
                        "llama-server 端口已被其他程式占用，無法自動重啟。",
                        code="llama_server_port_busy",
                        source="worker.llama_cpp_local",
                        category=self.category,
                        worker_name=self.name,
                        details={
                            **config_summary,
                            "external_server_recovery": stop_result,
                        },
                    ) from exc

            if not self.server_autostart:
                raise WorkerError(
                    "llama-server 未就緒，且 server_autostart 已關閉。",
                    code="llama_server_start_failed",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=config_summary,
                )

            exe = str(config_summary.get("resolved_server_exe") or self._resolve_llama_server_executable())
            build_no = self._get_llama_server_build(exe)
            if build_no is not None and build_no < 8189:
                raise WorkerError(
                    f"llama-server build 為 {build_no}，Qwen3.5 VL 需要 b8189+。",
                    code="llama_server_build_too_old",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details=config_summary,
                )
            cmd, launch_summary = self._build_server_command(endpoint, exe)
            proc = self._start_server_process(cmd, launch_summary)
            self.__class__._server_proc = proc
            self.__class__._server_key = key

            try:
                self._wait_server_ready(endpoint, self.server_start_timeout)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                self.__class__._server_proc = None
                self.__class__._server_key = None
                raise

    def _wait_server_ready(self, endpoint: str, timeout_seconds: int):
        deadline = time.time() + max(1, timeout_seconds)
        last_error = "unknown"
        last_exception: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                self._check_server_ready(endpoint, timeout_seconds=6.0)
                return
            except Exception as exc:
                last_error = str(exc)
                last_exception = exc
            proc = self.__class__._server_proc
            if proc is not None and proc.poll() is not None:
                stderr_tail = self._tail_server_log()
                stderr_text = "\n".join(stderr_tail).lower()
                if "unknown model architecture: 'gemma4'" in stderr_text:
                    raise WorkerError(
                        (
                            "llama-server 版本太舊，無法載入 Gemma 4（gemma4 architecture）。"
                            f" 目前 bundled build 是 {self._get_llama_server_build(self._resolve_llama_server_executable()) or 'unknown'}，"
                            f"Gemma 4 需要 b{MIN_SERVER_BUILD_GEMMA4}+。"
                        ),
                        code="llama_server_start_failed",
                        source="worker.llama_cpp_local",
                        category=self.category,
                        worker_name=self.name,
                        details={
                            "endpoint": endpoint,
                            "returncode": proc.poll(),
                            "port_open": self._is_port_open(endpoint),
                            "launch_summary": dict(self.__class__._server_launch_summary or {}),
                            "server_log_path": self.__class__._server_log_path,
                            "stderr_tail": stderr_tail,
                            "last_error": "unknown model architecture: gemma4",
                        },
                    )
                raise WorkerError(
                    "llama-server 在就緒前提前結束。",
                    code="llama_server_start_failed",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details={
                        "endpoint": endpoint,
                        "returncode": proc.poll(),
                        "port_open": self._is_port_open(endpoint),
                        "launch_summary": dict(self.__class__._server_launch_summary or {}),
                        "server_log_path": self.__class__._server_log_path,
                        "stderr_tail": self._tail_server_log(),
                        "last_error": last_error,
                    },
                )
            time.sleep(2.0)
        raise WorkerError(
            f"等待 llama-server 就緒逾時（{timeout_seconds}s）：{last_error}",
            code="llama_server_start_failed",
            source="worker.llama_cpp_local",
            category=self.category,
            worker_name=self.name,
            details={
                "endpoint": endpoint,
                "port_open": self._is_port_open(endpoint),
                "process_running": bool(self.__class__._server_proc and self.__class__._server_proc.poll() is None),
                "returncode": None if self.__class__._server_proc is None else self.__class__._server_proc.poll(),
                "launch_summary": dict(self.__class__._server_launch_summary or {}),
                "server_log_path": self.__class__._server_log_path,
                "stderr_tail": self._tail_server_log(),
                "last_error": last_error,
                "last_exception_type": last_exception.__class__.__name__ if last_exception is not None else None,
            },
        )

    def _call_caption(self, endpoint: str, image_data_url: str, user_prompt: str, system_prompt: str, thinking: bool):
        payload = {
            "model": self.model_name,
            "messages": [],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        })
        if thinking:
            payload["reasoning"] = {"enabled": True}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        resp_json = json.loads(text)
        if "error" in resp_json:
            raise RuntimeError(str(resp_json["error"]))
        choice = dict(resp_json.get("choices", [{}])[0] or {})
        message_obj = dict(choice.get("message", {}) or {})
        message = message_obj.get("content", "")
        result_text = " ".join(self._extract_text_content(message).strip().split())
        if not result_text:
            reasoning_text = " ".join(
                str(message_obj.get("reasoning_content", "") or "").strip().split()
            )
            finish_reason = str(choice.get("finish_reason") or "")
            if reasoning_text:
                raise WorkerError(
                    "模型只回了 reasoning_content，尚未輸出最終 content；請提高 max_tokens 後再試。",
                    code="worker_empty_content_response",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details={
                        "finish_reason": finish_reason,
                        "reasoning_preview": reasoning_text[:500],
                        "max_tokens": int(self.max_tokens),
                    },
                )
            raise WorkerError(
                "Model returned empty content",
                code="worker_empty_content_response",
                source="worker.llama_cpp_local",
                category=self.category,
                worker_name=self.name,
                details={"finish_reason": finish_reason, "max_tokens": int(self.max_tokens)},
            )
        return result_text

    def process(self, input_data: WorkerInput) -> WorkerOutput:
        try:
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            image_data = input_data.image
            if not image_data.path or not os.path.exists(image_data.path):
                return WorkerOutput(success=False, error=f"找不到圖片: {image_data.path}")

            settings = input_data.settings
            system_prompt = ""
            user_prompt = ""
            if settings:
                system_prompt = settings.llm_system_prompt or ""
                user_prompt = settings.llm_user_prompt_template or ""

            system_prompt = input_data.extra.get("system_prompt", system_prompt)
            user_prompt = input_data.extra.get("user_prompt", user_prompt)

            tags_context = image_data.tagger_tags or ""
            user_prompt = user_prompt.replace("{tags}", tags_context)

            repeat_count = 1
            if settings:
                repeat_count = int(getattr(settings, "llm_input_repeat_count", 1))
            repeat_count = int(input_data.extra.get("llm_input_repeat_count", repeat_count))
            repeat_count = max(1, repeat_count)
            if repeat_count > 1:
                user_prompt = "\n\n".join([user_prompt] * repeat_count)

            endpoint = self._resolve_chat_completions_url(self.base_url)
            self._ensure_server_ready(endpoint)

            image_data_url = self._encode_image_to_data_url(image_data.path)
            thinking = bool(input_data.extra.get("thinking_mode", False))

            result_text = self._call_caption(
                endpoint=endpoint,
                image_data_url=image_data_url,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                thinking=thinking,
            )

            image_data.llm_result = result_text
            if result_text not in image_data.nl_pages:
                image_data.nl_pages.append(result_text)

            return WorkerOutput(
                success=True,
                image=image_data,
                result_text=result_text,
            )

        except WorkerError as e:
            return WorkerOutput(
                success=False,
                error=str(e),
                error_info=build_worker_error_info(
                    e,
                    category=self.category,
                    worker_name=self.name,
                ),
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            lower = body.lower()
            if "image input is not supported" in lower:
                return WorkerOutput(
                    success=False,
                    error=(
                        "llama-server 回應：image input is not supported。"
                        "請確認 server 端使用 vision-capable 模型設定（qwen35-vl + mmproj），"
                        "並使用支援 vision/multimodal 的新版 llama-server。"
                    ),
                    error_info=build_worker_error_info(
                        None,
                        code="worker_image_input_unsupported",
                        message="llama-server does not support image input for the current model/server setup",
                        source="worker.llama_cpp_local",
                        worker_name=self.name,
                    ),
                )
            if "exceed_context_size_error" in lower or "available context size" in lower:
                prompt_tokens = None
                context_limit = None
                try:
                    payload = json.loads(body)
                    error_obj = dict(payload.get("error", {}) or {})
                    prompt_tokens = error_obj.get("n_prompt_tokens")
                    context_limit = error_obj.get("n_ctx")
                except Exception:
                    pass
                configured_ctx = int(self.n_ctx or 0)
                slot_count = max(1, int(self.server_workers or 0))
                slot_hint = ""
                if context_limit and configured_ctx and context_limit < configured_ctx and slot_count > 1:
                    slot_hint = (
                        f" 目前設定的 llama_cpp_server_workers 為 {slot_count}，"
                        "llama-server 會把總 context 分配到多個 slot；"
                        "若要單次請求拿到完整 context，請把 Server Slots 改成 1，"
                        "或提高總 context。"
                    )
                hint = (
                    f"llama-server 可用 context 只有 {context_limit or '未知'} tokens，"
                    f"但本次請求需要 {prompt_tokens or '更多'} tokens。"
                    f" 目前應用設定的 llama_cpp_n_ctx 為 {configured_ctx}。"
                    + slot_hint +
                    " 如果你是用本程式自動啟動 llama-server，重開這次任務後會改用設定值啟動。"
                    " 如果你是外部手動啟動 llama-server，請把 server 的 context size 調大後再試。"
                )
                return WorkerOutput(
                    success=False,
                    error=hint,
                    error_info=build_worker_error_info(
                        None,
                        code="worker_context_limit_exceeded",
                        message=hint,
                        source="worker.llama_cpp_local",
                        worker_name=self.name,
                        details={
                            "server_n_ctx": context_limit,
                            "prompt_tokens": prompt_tokens,
                            "configured_n_ctx": configured_ctx,
                            "server_workers": slot_count,
                            "spawned_ctx_size": self._effective_server_ctx_size(),
                        },
                    ),
                )
            return WorkerOutput(
                success=False,
                error=f"HTTP {e.code}: {body[:500]}",
                error_info=build_worker_error_info(
                    None,
                    code="worker_http_error",
                    message=f"HTTP {e.code}: {body[:500]}",
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                    details={"status_code": int(e.code)},
                ),
            )
        except Exception as e:
            return WorkerOutput(
                success=False,
                error=str(e),
                error_info=build_worker_error_info(
                    e,
                    source="worker.llama_cpp_local",
                    category=self.category,
                    worker_name=self.name,
                ),
            )

    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        if not self.base_url:
            return "缺少 llama-server base URL"
        if not self.model_name:
            return "缺少 llama-server model alias"
        return None
