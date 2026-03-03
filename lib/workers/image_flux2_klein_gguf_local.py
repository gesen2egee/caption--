# -*- coding: utf-8 -*-
"""
FLUX.2-klein GGUF local image processor worker.

This worker runs instruction-based image editing and overwrites the original
image file so sidecar/txt relationships stay consistent with existing app
behavior.
"""
import os
import traceback
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, List, Tuple

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput


class ImageFlux2KleinGGUFLocalWorker(BaseWorker):
    category = "IMAGE_PROCESS"
    display_name = "Local FLUX.2-klein-4B-GGUF"
    description = "Instruction-based image processing with FLUX.2-klein-4B GGUF"
    default_config = {
        "model_name": "unsloth/FLUX.2-klein-4B-GGUF",
        "base_model_name": "black-forest-labs/FLUX.2-klein-4B",
        "num_inference_steps": 6,
        "guidance_scale": 3.5,
        "strength": 0.85,
        "max_image_dimension": 1536,
        "seed": -1,
        # If True, only use local Hugging Face cache (no network).
        "local_files_only": False,
        # If True and GGUF load fails, fallback to full-precision base model.
        # Keep False by default to avoid unexpected huge downloads.
        "allow_full_model_fallback": False,
    }

    _pipeline = None
    _pipeline_key = None
    _pipeline_error = None
    _pipeline_lock = Lock()

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_name = str(self.config.get("model_name", self.default_config["model_name"]))
        self.base_model_name = str(self.config.get("base_model_name", self.default_config["base_model_name"]))
        self.num_inference_steps = int(self.config.get("num_inference_steps", self.default_config["num_inference_steps"]))
        self.guidance_scale = float(self.config.get("guidance_scale", self.default_config["guidance_scale"]))
        self.strength = float(self.config.get("strength", self.default_config["strength"]))
        self.max_image_dimension = int(self.config.get("max_image_dimension", self.default_config["max_image_dimension"]))
        self.seed = int(self.config.get("seed", self.default_config["seed"]))
        self.local_files_only = bool(self.config.get("local_files_only", self.default_config["local_files_only"]))
        self.allow_full_model_fallback = bool(
            self.config.get("allow_full_model_fallback", self.default_config["allow_full_model_fallback"])
        )

    @property
    def name(self) -> str:
        return "image_flux2_klein_gguf_local"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            import huggingface_hub  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _set_diffusers_xformers_safe():
        """
        Some Windows setups have broken xformers wheels that crash pipeline
        imports. If xformers import fails, force diffusers to treat it as unavailable.
        """
        try:
            import diffusers.utils.import_utils as import_utils
            if not import_utils.is_xformers_available():
                return
            try:
                import xformers.ops  # noqa: F401
            except Exception:
                import_utils._xformers_available = False
        except Exception:
            pass

    def _pick_gguf_filename(self, repo_files: List[str]) -> Optional[str]:
        ggufs = [f for f in repo_files if f.lower().endswith(".gguf")]
        if not ggufs:
            return None

        priority_tokens = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
        for token in priority_tokens:
            for filename in ggufs:
                if token.lower() in filename.lower():
                    return filename
        return ggufs[0]

    def _resolve_diffusers_classes(self) -> Tuple[type, type, type]:
        self._set_diffusers_xformers_safe()
        import diffusers

        pipeline_cls = getattr(diffusers, "Flux2KleinPipeline", None)
        if pipeline_cls is None:
            # Compatibility path for environments where Flux2KleinPipeline is
            # not exported yet but Flux2Pipeline exists.
            pipeline_cls = getattr(diffusers, "Flux2Pipeline", None)

        transformer_cls = getattr(diffusers, "Flux2Transformer2DModel", None)
        quant_cls = getattr(diffusers, "GGUFQuantizationConfig", None)

        missing = []
        if pipeline_cls is None:
            missing.append("Flux2KleinPipeline/Flux2Pipeline")
        if transformer_cls is None:
            missing.append("Flux2Transformer2DModel")
        if quant_cls is None:
            missing.append("GGUFQuantizationConfig")
        if missing:
            version = getattr(diffusers, "__version__", "unknown")
            raise RuntimeError(
                f"Current diffusers ({version}) is missing required FLUX.2 classes: {', '.join(missing)}. "
                "Please run: pip install -U diffusers transformers gguf"
            )

        return pipeline_cls, transformer_cls, quant_cls

    def _get_cached_gguf_files(self) -> List[str]:
        from huggingface_hub import snapshot_download

        try:
            snapshot_dir = snapshot_download(
                repo_id=self.model_name,
                repo_type="model",
                allow_patterns=["*.gguf"],
                local_files_only=True,
            )
        except Exception:
            return []
        return sorted([p.name for p in Path(snapshot_dir).glob("*.gguf")])

    def _resolve_gguf_filename(self) -> Tuple[str, bool]:
        from huggingface_hub import list_repo_files

        gguf_filename = str(self.config.get("gguf_filename", "")).strip()
        if gguf_filename:
            cached = gguf_filename in self._get_cached_gguf_files()
            return gguf_filename, cached

        cached = self._get_cached_gguf_files()
        if cached:
            picked = self._pick_gguf_filename(cached)
            if picked:
                return picked, True

        if self.local_files_only:
            raise RuntimeError(
                f"No cached GGUF file found for {self.model_name}. "
                "Disable local_files_only or pre-download the GGUF file."
            )

        repo_files = list_repo_files(self.model_name)
        picked = self._pick_gguf_filename(repo_files or [])
        if not picked:
            raise RuntimeError(f"No GGUF file found in repository: {self.model_name}")
        return picked, False

    def _resolve_base_model_source(self) -> str:
        from huggingface_hub import snapshot_download

        # Prefer an already cached snapshot to avoid unnecessary network HEAD
        # requests on environments with restricted proxy settings.
        if not self.local_files_only:
            try:
                return snapshot_download(
                    repo_id=self.base_model_name,
                    repo_type="model",
                    local_files_only=True,
                )
            except Exception:
                return self.base_model_name

        try:
            return snapshot_download(
                repo_id=self.base_model_name,
                repo_type="model",
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Base model is not available in local cache: {self.base_model_name}. "
                "Disable local_files_only for first-time download."
            ) from e

    @staticmethod
    def _format_load_error(prefix: str, err: Exception) -> str:
        message = str(err).replace("\n", " ").strip()
        if len(message) > 600:
            message = message[:600] + "..."

        lower = message.lower()
        hints = []
        if "meta tensor" in lower:
            hints.append("try: pip install -U diffusers accelerate")
        if "size mismatch" in lower or "in_layer.bias" in lower:
            hints.append("GGUF may be incompatible with current FLUX.2 config")
        if "proxyerror" in lower or "connection refused" in lower:
            hints.append("check network/proxy access to huggingface.co")
        if hints:
            message += " (" + "; ".join(hints) + ")"

        return f"{prefix}: {message}"

    def _place_pipeline(self, pipe):
        import torch

        if torch.cuda.is_available():
            try:
                pipe.to("cuda")
            except Exception:
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
        return pipe

    def _build_pipeline(self):
        import torch
        from huggingface_hub import hf_hub_download

        pipe_cls, transformer_cls, quant_cls = self._resolve_diffusers_classes()
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_errors = []
        base_model_source = self._resolve_base_model_source()
        from_pretrained_target = base_model_source

        gguf_filename, gguf_cached = self._resolve_gguf_filename()
        gguf_local_only = self.local_files_only or gguf_cached
        gguf_path = hf_hub_download(
            repo_id=self.model_name,
            filename=gguf_filename,
            repo_type="model",
            local_files_only=gguf_local_only,
        )
        quant_config = quant_cls(compute_dtype=compute_dtype)

        transformer = None
        try:
            transformer = transformer_cls.from_single_file(
                gguf_path,
                quantization_config=quant_config,
                config=base_model_source,
                subfolder="transformer",
                torch_dtype=compute_dtype,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            load_errors.append(self._format_load_error("GGUF transformer load failed", e))

        if transformer is not None:
            try:
                pipe = pipe_cls.from_pretrained(
                    from_pretrained_target,
                    transformer=transformer,
                    torch_dtype=compute_dtype,
                    local_files_only=self.local_files_only,
                )
                return self._place_pipeline(pipe)
            except Exception as e:
                load_errors.append(self._format_load_error(f"{pipe_cls.__name__} GGUF pipeline load failed", e))

        if self.allow_full_model_fallback:
            try:
                pipe = pipe_cls.from_pretrained(
                    from_pretrained_target,
                    torch_dtype=compute_dtype,
                    local_files_only=self.local_files_only,
                )
                return self._place_pipeline(pipe)
            except Exception as e:
                load_errors.append(self._format_load_error(f"{pipe_cls.__name__} full-model fallback failed", e))
        else:
            load_errors.append("Full-model fallback disabled (allow_full_model_fallback=False).")

        raise RuntimeError("Unable to load FLUX.2 [klein] GGUF pipeline. " + " | ".join(load_errors))

    def _get_pipeline(self):
        key = (
            self.model_name,
            self.base_model_name,
            str(self.config.get("gguf_filename", "")),
            self.local_files_only,
            self.allow_full_model_fallback,
        )
        with self._pipeline_lock:
            if self.__class__._pipeline_key != key:
                self.__class__._pipeline = None
                self.__class__._pipeline_error = None
                self.__class__._pipeline_key = key

            if self.__class__._pipeline is not None:
                return self.__class__._pipeline

            if self.__class__._pipeline_error:
                raise RuntimeError(self.__class__._pipeline_error)

            try:
                self.__class__._pipeline = self._build_pipeline()
                self.__class__._pipeline_error = None
            except Exception as e:
                self.__class__._pipeline_error = str(e)
                raise
        return self.__class__._pipeline

    def _prepare_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        original_size = img.size

        w, h = img.size
        max_side = max(w, h)
        if max_side > self.max_image_dimension > 0:
            scale = self.max_image_dimension / float(max_side)
            w = max(64, int(w * scale))
            h = max(64, int(h * scale))
            img = img.resize((w, h), Image.Resampling.LANCZOS)

        w = max(64, (img.width // 16) * 16)
        h = max(64, (img.height // 16) * 16)
        if (w, h) != img.size:
            img = img.resize((w, h), Image.Resampling.LANCZOS)

        return img, original_size

    @staticmethod
    def _extract_output_image(result):
        if hasattr(result, "images") and result.images:
            return result.images[0]
        if isinstance(result, list) and result:
            return result[0]
        return None

    def _run_pipeline(self, pipe, prompt: str, input_image: Image.Image):
        import torch

        base_kwargs = {
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
        }
        if self.seed >= 0:
            generator_device = "cuda" if torch.cuda.is_available() else "cpu"
            base_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(self.seed)

        # Different pipeline classes may accept different optional args.
        optional_sets = (
            {"guidance_scale": self.guidance_scale, "strength": self.strength},
            {"guidance_scale": self.guidance_scale},
            {"strength": self.strength},
            {},
        )
        image_arg_names = ("image", "input_image", "images", "reference_images")

        for arg_name in image_arg_names:
            for optional_kwargs in optional_sets:
                try:
                    result = pipe(**{**base_kwargs, **optional_kwargs, arg_name: input_image})
                    output_image = self._extract_output_image(result)
                    if output_image is not None:
                        return output_image
                except TypeError:
                    continue

        raise RuntimeError("Loaded pipeline does not accept image input for editing.")

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

            pipe = self._get_pipeline()
            input_image, original_size = self._prepare_image(image_path)
            output_image = self._run_pipeline(pipe, prompt, input_image)
            if output_image is None:
                return WorkerOutput(success=False, error="No image result returned by pipeline")

            if output_image.size != original_size:
                output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)

            self._save_image(output_image, image_path)

            return WorkerOutput(
                success=True,
                image=image_data,
                result_data={
                    "original_path": image_path,
                    "result_path": image_path,
                    "model_name": self.model_name,
                },
            )
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))

    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "Missing image data"
        return None
