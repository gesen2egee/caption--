# -*- coding: utf-8 -*-
"""
FLUX.2-klein image editing worker backed by stable-diffusion.cpp sd-server.

The worker keeps the existing worker id for backward compatibility, but no
longer loads GGUF through diffusers in-process. Instead it talks to the
OpenAI-compatible `/v1/images/edits` endpoint exposed by `sd-server`.
"""
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput


HF_BLOB_OR_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/"
    r"(?P<repo>[^/]+/[^/]+)/"
    r"(?:(?:blob)|(?:resolve))/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)$"
)


class ImageFlux2KleinGGUFLocalWorker(BaseWorker):
    category = "IMAGE_PROCESS"
    display_name = "FLUX.2-klein via stable-diffusion.cpp"
    description = "Instruction-based image editing through sd-server OpenAI Images API"
    default_config = {
        "base_url": "http://127.0.0.1:8001/v1",
        "model_name": "unsloth/FLUX.2-klein-4B-GGUF",
        "diffusion_model_path": "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-BF16.gguf",
        "vae_path": "https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors",
        "llm_path": "https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf",
        "num_inference_steps": 6,
        "guidance_scale": 3.5,
        "max_image_dimension": 1536,
        "seed": -1,
        "local_files_only": False,
        "server_autostart": True,
        "server_exe": "",
        "server_start_timeout": 900,
        "server_args_extra": "",
    }

    _server_proc = None
    _server_key = None
    _server_lock = Lock()

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.base_url = str(self.config.get("base_url", self.default_config["base_url"])).strip()
        self.model_name = str(self.config.get("model_name", self.default_config["model_name"])).strip()
        self.diffusion_model_path = str(
            self.config.get("diffusion_model_path", self.default_config["diffusion_model_path"])
        ).strip()
        self.vae_path = str(self.config.get("vae_path", self.default_config["vae_path"])).strip()
        self.llm_path = str(self.config.get("llm_path", self.default_config["llm_path"])).strip()
        self.num_inference_steps = int(
            self.config.get("num_inference_steps", self.default_config["num_inference_steps"])
        )
        self.guidance_scale = float(self.config.get("guidance_scale", self.default_config["guidance_scale"]))
        self.max_image_dimension = int(
            self.config.get("max_image_dimension", self.default_config["max_image_dimension"])
        )
        self.seed = int(self.config.get("seed", self.default_config["seed"]))
        self.local_files_only = bool(self.config.get("local_files_only", self.default_config["local_files_only"]))
        self.server_autostart = bool(
            self.config.get("server_autostart", self.default_config["server_autostart"])
        )
        self.server_exe = str(self.config.get("server_exe", self.default_config["server_exe"])).strip()
        self.server_start_timeout = int(
            self.config.get("server_start_timeout", self.default_config["server_start_timeout"])
        )
        self.server_args_extra = str(
            self.config.get("server_args_extra", self.default_config["server_args_extra"])
        ).strip()

    @property
    def name(self) -> str:
        return "image_flux2_klein_gguf_local"

    @classmethod
    def is_available(cls) -> bool:
        # Keep endpoint mode always selectable. Runtime reports actionable
        # errors when sd-server or assets are missing.
        return True

    @staticmethod
    def _resolve_models_url(base_url: str) -> str:
        url = (base_url or "").strip().rstrip("/")
        if not url:
            raise RuntimeError("sd-server base URL is empty")
        if url.endswith("/models"):
            return url
        if url.endswith("/v1"):
            return f"{url}/models"
        return f"{url}/v1/models"

    @staticmethod
    def _resolve_images_edits_url(base_url: str) -> str:
        url = (base_url or "").strip().rstrip("/")
        if not url:
            raise RuntimeError("sd-server base URL is empty")
        if url.endswith("/images/edits"):
            return url
        if url.endswith("/v1"):
            return f"{url}/images/edits"
        return f"{url}/v1/images/edits"

    @staticmethod
    def _parse_port_from_endpoint(endpoint: str) -> int:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        if parsed.port:
            return int(parsed.port)
        if parsed.scheme == "https":
            return 443
        return 80

    @staticmethod
    def _parse_hf_source(model_source: str) -> Tuple[Optional[str], Optional[str], str]:
        src = (model_source or "").strip()
        if not src:
            return None, None, "main"

        parsed = HF_BLOB_OR_RESOLVE_RE.match(src)
        if parsed:
            return parsed.group("repo"), parsed.group("filename"), parsed.group("revision")

        parts = src.split("/")
        if len(parts) >= 3 and any(src.lower().endswith(ext) for ext in (".gguf", ".safetensors", ".ckpt", ".pth")):
            return "/".join(parts[:2]), "/".join(parts[2:]), "main"

        if len(parts) == 2:
            return src, None, "main"

        return None, None, "main"

    @staticmethod
    def _pick_repo_file(repo_files: List[str], kind: str) -> Optional[str]:
        files = [f for f in (repo_files or []) if not f.endswith("/")]
        lower_kind = kind.lower()

        if lower_kind == "diffusion":
            candidates = [f for f in files if f.lower().endswith(".gguf")]
            priority_tokens = ["q4_k_m", "q5_k_m", "q6_k", "q8_0", "bf16", "f16"]
        elif lower_kind == "llm":
            candidates = [f for f in files if f.lower().endswith(".gguf")]
            priority_tokens = ["q4_k_m", "q4_0", "q5_k_m", "q8_0", "bf16", "f16"]
        else:
            exact = [f for f in files if os.path.basename(f).lower() in ("ae.safetensors", "flux2_ae.safetensors")]
            if exact:
                return exact[0]
            candidates = [f for f in files if f.lower().endswith(".safetensors")]
            priority_tokens = ["flux2_ae", "ae"]

        if not candidates:
            return None

        for token in priority_tokens:
            for filename in candidates:
                if token in filename.lower():
                    return filename
        return candidates[0]

    def _resolve_asset_to_local_path(self, source: str, kind: str) -> str:
        src = (source or "").strip()
        if not src:
            raise RuntimeError(f"缺少 {kind} 模型設定")
        if os.path.exists(src):
            return src

        from huggingface_hub import hf_hub_download, list_repo_files

        repo_id, filename, revision = self._parse_hf_source(src)
        if not repo_id:
            raise RuntimeError(f"找不到 {kind} 模型檔: {src}")

        if not filename:
            try:
                repo_files = list_repo_files(repo_id, repo_type="model")
            except Exception as exc:
                if self.local_files_only:
                    raise RuntimeError(f"{kind} local_files_only 已啟用，且快取中沒有可用檔案: {repo_id}") from exc
                raise RuntimeError(f"無法列出 {kind} repo 檔案: {repo_id}") from exc
            filename = self._pick_repo_file(repo_files, kind)
            if not filename:
                raise RuntimeError(f"在 repo 中找不到可用的 {kind} 檔案: {repo_id}")

        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                revision=revision or "main",
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            hint = "請先執行 setup.bat 預下載模型，或關閉 local_files_only。" if self.local_files_only else "請檢查 Hugging Face 網路連線或模型路徑。"
            raise RuntimeError(f"下載/解析 {kind} 模型失敗: {src}。{hint}") from exc

    def _resolve_sd_server_executable(self) -> str:
        if self.server_exe and os.path.exists(self.server_exe):
            return self.server_exe

        runtime_root = Path(os.getcwd()) / "tasks" / "runtime" / "stable-diffusion-cpp"
        if runtime_root.exists():
            candidates = sorted(runtime_root.glob("**/sd-server.exe"), reverse=True)
            if candidates:
                return str(candidates[0])

        found = shutil.which("sd-server")
        if found:
            return found

        raise RuntimeError("找不到 sd-server.exe。請先執行 setup.bat，或在設定中指定 stable-diffusion.cpp server 路徑。")

    def _check_server_ready(self, models_url: str, timeout_seconds: float = 8.0):
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            _ = resp.read()

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

    def _wait_server_ready(self, models_url: str, timeout_seconds: int):
        deadline = time.time() + max(1, timeout_seconds)
        last_error = "unknown"
        while time.time() < deadline:
            try:
                self._check_server_ready(models_url, timeout_seconds=6.0)
                return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(2.0)
        raise RuntimeError(f"等待 sd-server 就緒逾時（{timeout_seconds}s）：{last_error}")

    def _ensure_server_ready(self):
        models_url = self._resolve_models_url(self.base_url)
        key = (
            self.base_url,
            self.diffusion_model_path,
            self.vae_path,
            self.llm_path,
            self.num_inference_steps,
            self.guidance_scale,
            self.seed,
            self.server_args_extra,
            self.local_files_only,
        )

        with self._server_lock:
            proc = self.__class__._server_proc
            if proc is not None:
                if proc.poll() is not None:
                    self.__class__._server_proc = None
                    self.__class__._server_key = None
                elif self.__class__._server_key == key:
                    self._wait_server_ready(models_url, self.server_start_timeout)
                    return
                else:
                    self.__class__._stop_managed_server()

            try:
                self._check_server_ready(models_url, timeout_seconds=4.0)
                return
            except Exception:
                pass

            if not self.server_autostart:
                raise RuntimeError("sd-server 未就緒，且 image_process_server_autostart 已關閉。")

            exe = self._resolve_sd_server_executable()
            diffusion_path = self._resolve_asset_to_local_path(self.diffusion_model_path, "diffusion")
            vae_path = self._resolve_asset_to_local_path(self.vae_path, "vae")
            llm_path = self._resolve_asset_to_local_path(self.llm_path, "llm")

            port = self._parse_port_from_endpoint(self.base_url)
            cmd = [
                exe,
                "--listen-ip",
                "127.0.0.1",
                "--listen-port",
                str(port),
                "--diffusion-model",
                diffusion_path,
                "--vae",
                vae_path,
                "--llm",
                llm_path,
                "--steps",
                str(max(1, self.num_inference_steps)),
                "--guidance",
                str(self.guidance_scale),
                "--sampling-method",
                "euler",
                "--diffusion-fa",
            ]
            if self.seed >= 0:
                cmd.extend(["-s", str(self.seed)])
            if self.server_args_extra:
                cmd.extend(shlex.split(self.server_args_extra, posix=False))

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(exe).resolve().parent),
            )
            self.__class__._server_proc = proc
            self.__class__._server_key = key

            try:
                self._wait_server_ready(models_url, self.server_start_timeout)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise

    def _prepare_image(self, image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size

        w, h = img.size
        max_side = max(w, h)
        if max_side > self.max_image_dimension > 0:
            scale = self.max_image_dimension / float(max_side)
            w = max(64, int(w * scale))
            h = max(64, int(h * scale))
            img = img.resize((w, h), Image.Resampling.LANCZOS)

        return img, original_size

    @staticmethod
    def _encode_png(image: Image.Image) -> bytes:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _build_multipart(fields: List[Tuple[str, str]], files: List[Tuple[str, str, bytes, str]]):
        boundary = f"----CodexBoundary{uuid.uuid4().hex}"
        body = BytesIO()

        for name, value in fields:
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.write(str(value).encode("utf-8"))
            body.write(b"\r\n")

        for name, filename, content, content_type in files:
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            body.write(content)
            body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode("utf-8"))
        return body.getvalue(), boundary

    def _call_image_edit(self, image_bytes: bytes, prompt: str, size_text: str) -> Image.Image:
        endpoint = self._resolve_images_edits_url(self.base_url)
        fields = [
            ("prompt", prompt),
            ("size", size_text),
            ("n", "1"),
            ("output_format", "png"),
        ]
        files = [("image", "input.png", image_bytes, "image/png")]
        body, boundary = self._build_multipart(fields, files)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))

        data = payload.get("data") or []
        if not data:
            raise RuntimeError("sd-server 未回傳圖片資料")
        b64 = str(data[0].get("b64_json", "")).strip()
        if not b64:
            raise RuntimeError("sd-server 回傳空的 b64_json")

        decoded = base64.b64decode(b64)
        return Image.open(BytesIO(decoded)).convert("RGB")

    @staticmethod
    def _save_image(image: Image.Image, target_path: str):
        ext = os.path.splitext(target_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            image.convert("RGB").save(target_path, "JPEG", quality=95)
        elif ext == ".png":
            image.save(target_path, "PNG")
        elif ext == ".webp":
            image.save(target_path, "WEBP", quality=95)
        else:
            image.save(target_path)

    def process(self, input_data: WorkerInput) -> WorkerOutput:
        try:
            if not input_data.image:
                return WorkerOutput(success=False, error="Missing image data")

            image_data = input_data.image
            image_path = image_data.path
            if not image_path or not os.path.exists(image_path):
                return WorkerOutput(success=False, error=f"Image not found: {image_path}")

            prompt = str(input_data.extra.get("edit_prompt", "")).strip()
            if not prompt:
                return WorkerOutput(success=False, error="Image process prompt is empty")

            self._ensure_server_ready()

            input_image, original_size = self._prepare_image(image_path)
            image_bytes = self._encode_png(input_image)
            result_image = self._call_image_edit(image_bytes, prompt, f"{input_image.width}x{input_image.height}")

            if result_image.size != original_size:
                result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)

            self._save_image(result_image, image_path)
            return WorkerOutput(
                success=True,
                image=image_data,
                result_data={
                    "original_path": image_path,
                    "result_path": image_path,
                    "model_name": self.model_name,
                    "backend": "stable-diffusion.cpp",
                },
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return WorkerOutput(success=False, error=f"HTTP {exc.code}: {body[:500]}")
        except Exception as exc:
            return WorkerOutput(success=False, error=str(exc))

    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "Missing image data"
        if not self.base_url:
            return "Missing sd-server base URL"
        if not self.diffusion_model_path:
            return "Missing diffusion model path"
        if not self.vae_path:
            return "Missing VAE path"
        if not self.llm_path:
            return "Missing LLM path"
        return None
