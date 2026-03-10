# -*- coding: utf-8 -*-
"""
LLMTask - LLM 自然語言生成任務

使用 VLM API 對圖片生成自然語言描述。
"""
import traceback
from typing import Tuple

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.runtime.errors import build_runtime_error_info
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar
from lib.utils.parsing import extract_llm_content_and_postprocess
from lib.workers import invoke_worker


class LLMTask(BaseTask):
    """
    LLM 自然語言生成任務
    
    使用 OpenRouter API 的 VLM 模型進行圖片描述生成。
    """
    
    @property
    def name(self) -> str:
        return "llm"
    
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """檢查 NSFW 跳過設定"""
        if context.extra.get("force_execution", False):
            return False, ""
            
        if context.settings and context.settings.llm_skip_nsfw_on_batch:
            tags = (context.image.tagger_tags or "").lower()
            if "explicit" in tags or "questionable" in tags:
                return True, "NSFW 內容"
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """執行 LLM 生成"""
        try:
            # 1. Skip 判斷
            should_skip, skip_reason = self.should_skip(context)
            if should_skip:
                return TaskResult(
                    success=True,
                    skipped=True,
                    skip_reason=skip_reason,
                    image=context.image,
                )
            
            # 2. 準備 Prompt
            user_prompt = context.extra.get("user_prompt", "")
            system_prompt = context.extra.get("system_prompt", "")
            
            if not user_prompt and context.settings:
                user_prompt = context.settings.llm_user_prompt_template
            if not system_prompt and context.settings:
                system_prompt = context.settings.llm_system_prompt
            
            # 3. 建立並呼叫 Worker
            # from lib.workers.vlm_openrouter_api import VLMOpenRouterAPIWorker
            worker_name = context.settings.llm_provider if (context.settings and context.settings.llm_provider) else "vlm_openrouter_api"
            config = {}
            if context.settings:
                if worker_name == "llm_llama_cpp_local":
                    config = {
                        "base_url": getattr(context.settings, "llama_cpp_base_url", "http://127.0.0.1:8000/v1"),
                        "api_key": getattr(context.settings, "llama_cpp_api_key", ""),
                        "model_name": getattr(context.settings, "llama_cpp_model_alias", "qwen35-vl-gguf"),
                        "max_image_dim": context.settings.llm_max_image_dimension,
                        "use_gray_mask": context.settings.llm_use_gray_mask,
                        "temperature": float(getattr(context.settings, "llama_cpp_temperature", 1.0)),
                        "top_p": float(getattr(context.settings, "llama_cpp_top_p", 0.8)),
                        "top_k": int(getattr(context.settings, "llama_cpp_top_k", 20)),
                        "min_p": float(getattr(context.settings, "llama_cpp_min_p", 0.0)),
                        "presence_penalty": float(getattr(context.settings, "llama_cpp_presence_penalty", 1.5)),
                        "repetition_penalty": float(getattr(context.settings, "llama_cpp_repeat_penalty", 1.0)),
                        "model_path": context.settings.llama_cpp_model_path,
                        "n_ctx": int(getattr(context.settings, "llama_cpp_n_ctx", 8192)),
                        "n_threads": int(getattr(context.settings, "llama_cpp_n_threads", 0)),
                        "n_gpu_layers": int(context.settings.llama_cpp_n_gpu_layers),
                        "max_tokens": int(context.settings.llama_cpp_max_tokens),
                        "mmproj_path": context.settings.llama_cpp_mmproj_path,
                        "enable_vision": bool(context.settings.llama_cpp_enable_vision),
                        "server_autostart": bool(getattr(context.settings, "llama_cpp_server_autostart", True)),
                        "server_exe": str(getattr(context.settings, "llama_cpp_server_exe", "")),
                        "server_workers": int(getattr(context.settings, "llama_cpp_server_workers", 1)),
                        "server_start_timeout": int(
                            getattr(context.settings, "llama_cpp_server_start_timeout", 900)
                        ),
                    }
                else:
                    config = {
                        "base_url": context.settings.llm_base_url,
                        "api_key": context.settings.llm_api_key,
                        "model_name": context.settings.llm_model,
                        "max_image_dim": context.settings.llm_max_image_dimension,
                        "use_gray_mask": context.settings.llm_use_gray_mask,
                        "temperature": context.settings.llm_temperature,
                        "top_p": context.settings.llm_top_p,
                        "reasoning_enabled": bool(context.settings.llm_thinking_mode),
                    }
            
            # 將 prompt 放入 extra
            worker_input = context.to_worker_input()
            worker_input.extra["user_prompt"] = user_prompt
            worker_input.extra["system_prompt"] = system_prompt
            if context.settings:
                worker_input.extra["llm_input_repeat_count"] = int(context.settings.llm_input_repeat_count)
                worker_input.extra["temperature"] = float(context.settings.llm_temperature)
                worker_input.extra["top_p"] = float(context.settings.llm_top_p)
                worker_input.extra["thinking_mode"] = bool(context.settings.llm_thinking_mode)
            
            worker_output = invoke_worker(
                "LLM",
                worker_name,
                config=config,
                worker_input=worker_input,
                settings=context.settings,
            )
            
            if not worker_output.success:
                return TaskResult(
                    success=False,
                    error=worker_output.error,
                    error_info=worker_output.error_info,
                    image=context.image,
                )
            
            # 4. 後處理：提取內容並儲存
            content = worker_output.result_text or ""
            english_force_lowercase = context.settings.english_force_lowercase if context.settings else True
            final_content = extract_llm_content_and_postprocess(content, english_force_lowercase)
            
            if final_content:
                sidecar = load_image_sidecar(context.image.path)
                nl_pages = sidecar.get("nl_pages", [])
                if final_content not in nl_pages:
                    nl_pages.append(final_content)
                sidecar["nl_pages"] = nl_pages
                sidecar["llm_result"] = final_content
                save_image_sidecar(context.image.path, sidecar)
                
                # 更新 ImageData
                context.image.llm_result = final_content
                if final_content not in context.image.nl_pages:
                    context.image.nl_pages.append(final_content)
            
            return TaskResult(
                success=True,
                image=context.image,
                result_text=final_content,
            )
            
        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                error_info=build_runtime_error_info(e, source="pipeline.task.llm"),
                image=context.image,
            )
