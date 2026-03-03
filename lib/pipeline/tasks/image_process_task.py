# -*- coding: utf-8 -*-
"""
ImageProcessTask - image editing task.

Runs an IMAGE_PROCESS worker with an instruction prompt and writes the result
back to the image file while preserving the project's backup/sidecar flow.
"""
import traceback
from typing import Tuple

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.core.settings import DEFAULT_IMAGE_PROCESS_PROMPT_TEMPLATE
from lib.utils.file_ops import backup_raw_image
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar


class ImageProcessTask(BaseTask):
    @property
    def name(self) -> str:
        return "image_process"

    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        # Keep behavior aligned with TAGGER/LLM: no implicit skip by default.
        return False, ""

    def execute(self, context: TaskContext) -> TaskResult:
        try:
            should_skip, skip_reason = self.should_skip(context)
            if should_skip:
                return TaskResult(
                    success=True,
                    skipped=True,
                    skip_reason=skip_reason,
                    image=context.image,
                )

            image_path = context.image.path
            settings = context.settings

            prompt = str(context.extra.get("edit_prompt", "")).strip()
            if not prompt and settings:
                prompt = str(getattr(settings, "image_process_prompt_template", "")).strip()
            if not prompt:
                prompt = DEFAULT_IMAGE_PROCESS_PROMPT_TEMPLATE

            backup_raw_image(image_path)

            from lib.workers.registry import get_registry

            worker_name = (
                settings.image_process_worker
                if (settings and getattr(settings, "image_process_worker", ""))
                else "image_flux2_klein_gguf_local"
            )
            WorkerCls = get_registry().get_worker_class("IMAGE_PROCESS", worker_name)
            if not WorkerCls and worker_name == "image_flux2_klein_gguf_local":
                try:
                    from lib.workers.image_flux2_klein_gguf_local import ImageFlux2KleinGGUFLocalWorker
                    WorkerCls = ImageFlux2KleinGGUFLocalWorker
                except Exception:
                    WorkerCls = None
            if not WorkerCls:
                return TaskResult(
                    success=False,
                    error=f"Image Process Worker '{worker_name}' not found",
                    image=context.image,
                )

            config = {
                "model_name": getattr(settings, "image_process_model", "unsloth/FLUX.2-klein-4B-GGUF"),
                "num_inference_steps": getattr(settings, "image_process_steps", 6),
                "guidance_scale": getattr(settings, "image_process_guidance_scale", 3.5),
                "max_image_dimension": getattr(settings, "image_process_max_dimension", 1536),
                "seed": getattr(settings, "image_process_seed", -1),
                "local_files_only": getattr(settings, "image_process_local_files_only", False),
                "allow_full_model_fallback": getattr(settings, "image_process_allow_full_model_fallback", False),
                "gguf_filename": getattr(settings, "image_process_gguf_filename", ""),
            }

            worker_input = context.to_worker_input()
            worker_input.extra["edit_prompt"] = prompt

            worker = WorkerCls(config)
            worker_output = worker.process(worker_input)
            if not worker_output.success:
                return TaskResult(
                    success=False,
                    error=worker_output.error,
                    image=context.image,
                )

            if worker_output.skipped:
                return TaskResult(
                    success=True,
                    skipped=True,
                    skip_reason=worker_output.skip_reason,
                    image=context.image,
                )

            result_data = worker_output.result_data or {}
            new_path = result_data.get("result_path", image_path)

            sidecar = load_image_sidecar(new_path)
            sidecar["image_processed"] = True
            sidecar["image_process_model"] = config["model_name"]
            sidecar["image_process_prompt"] = prompt

            prompts = sidecar.get("image_process_prompts", [])
            if not isinstance(prompts, list):
                prompts = []
            prompts.append(prompt)
            sidecar["image_process_prompts"] = prompts[-20:]
            save_image_sidecar(new_path, sidecar)

            context.image.path = new_path

            return TaskResult(
                success=True,
                image=context.image,
                result_data={
                    "original_path": image_path,
                    "result_path": new_path,
                    "model_name": config["model_name"],
                    "prompt": prompt,
                },
            )

        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                image=context.image,
            )
