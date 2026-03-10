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
from lib.workers.errors import build_worker_error_info


HF_BLOB_OR_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/"
    r"(?P<repo>[^/]+/[^/]+)/"
    r"(?:(?:blob)|(?:resolve))/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)$"
)
DEFAULT_MMPROJ_URL = "https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/resolve/main/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"


class LLMLlamaCppLocalWorker(BaseWorker):
    category = "LLM"
    display_name = "LLaMA.cpp Local (GGUF)"
    description = "Run local GGUF models via llama-server OpenAI-compatible API"
    default_config = {
        "model_path": "https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/blob/main/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
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

        bundled = os.path.join(os.getcwd(), "tasks", "runtime", "llama-b8189", "llama-server.exe")
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
            "找不到 llama-server。請先安裝 b8189+，或在設定中指定 llama-server.exe 路徑。"
        )

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
            _ = resp.read()

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

    def _ensure_server_ready(self, endpoint: str):
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
            except Exception:
                pass

            if not self.server_autostart:
                raise RuntimeError("llama-server 未就緒，且 server_autostart 已關閉。")

            exe = self._resolve_llama_server_executable()
            build_no = self._get_llama_server_build(exe)
            if build_no is not None and build_no < 8189:
                raise RuntimeError(
                    f"llama-server build 為 {build_no}，Qwen3.5 VL 需要 b8189+。"
                )
            port = self._parse_port_from_endpoint(endpoint)
            cmd = [
                exe,
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
                mmproj = self.mmproj_path.strip()
                if mmproj.lower().startswith(("http://", "https://")):
                    cmd.extend(["--mmproj-url", mmproj])
                else:
                    cmd.extend(["--mmproj", mmproj])

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
            )
            self.__class__._server_proc = proc
            self.__class__._server_key = key

            try:
                self._wait_server_ready(endpoint, self.server_start_timeout)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise

    def _wait_server_ready(self, endpoint: str, timeout_seconds: int):
        deadline = time.time() + max(1, timeout_seconds)
        last_error = "unknown"
        while time.time() < deadline:
            try:
                self._check_server_ready(endpoint, timeout_seconds=6.0)
                return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(2.0)
        raise RuntimeError(f"等待 llama-server 就緒逾時（{timeout_seconds}s）：{last_error}")

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
        message = (
            resp_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        result_text = " ".join(self._extract_text_content(message).strip().split())
        if not result_text:
            raise RuntimeError("Model returned empty content")
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

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            lower = body.lower()
            if "image input is not supported" in lower:
                return WorkerOutput(
                    success=False,
                    error=(
                        "llama-server 回應：image input is not supported。"
                        "請確認 server 端使用 vision-capable 模型設定（qwen35-vl + mmproj），"
                        "並使用 b8189+ 版本。"
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
            return WorkerOutput(success=False, error=f"HTTP {e.code}: {body[:500]}")
        except Exception as e:
            return WorkerOutput(success=False, error=str(e))

    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        if not self.base_url:
            return "缺少 llama-server base URL"
        if not self.model_name:
            return "缺少 llama-server model alias"
        return None
