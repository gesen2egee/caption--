# -*- coding: utf-8 -*-
"""
LLM 任務

單圖和批量 LLM 處理任務。
"""
from typing import List, Callable, Optional

from lib.tasks.base import SingleImageTask, BatchTask
from lib.workers.base import BaseWorker, WorkerInput
from lib.workers.vlm_openrouter_api import VLMOpenRouterAPIWorker
from lib.core.dataclasses import ImageData, Settings


class LLMTask(SingleImageTask):
    """
    單圖 LLM 處理任務
    """
    
    def __init__(self, image: ImageData, settings: Settings = None, 
                 prompt_mode: str = "default"):
        super().__init__(image, settings)
        self.prompt_mode = prompt_mode  # "default" or "custom"
    
    @property
    def name(self) -> str:
        return "llm_single"
    
    def get_worker(self) -> BaseWorker:
        config = {}
        if self.settings:
            config = {
                "base_url": self.settings.llm_base_url,
                "api_key": self.settings.llm_api_key,
                "model_name": self.settings.llm_model,
                "max_image_dim": self.settings.llm_max_image_dimension,
                "use_gray_mask": self.settings.llm_use_gray_mask,
            }
        return VLMOpenRouterAPIWorker(config)
    
    def prepare_input(self) -> WorkerInput:
        # 根據 prompt_mode 選擇提示詞
        user_prompt = ""
        if self.settings:
            if self.prompt_mode == "custom":
                user_prompt = self.settings.llm_custom_prompt_template
            else:
                user_prompt = self.settings.llm_user_prompt_template
        
        return WorkerInput(
            image=self.image,
            settings=self.settings,
            extra={
                "system_prompt": self.settings.llm_system_prompt if self.settings else "",
                "user_prompt": user_prompt,
            }
        )


class BatchLLMTask(BatchTask):
    """
    批量 LLM 處理任務
    """
    
    def __init__(self, images: List[ImageData], settings: Settings = None,
                 prompt_mode: str = "default",
                 tags_context_getter: Callable[[str], str] = None):
        super().__init__(images, settings)
        self.prompt_mode = prompt_mode
        self.tags_context_getter = tags_context_getter
    
    @property
    def name(self) -> str:
        return "llm_batch"
    
    def get_worker(self) -> BaseWorker:
        config = {}
        if self.settings:
            config = {
                "base_url": self.settings.llm_base_url,
                "api_key": self.settings.llm_api_key,
                "model_name": self.settings.llm_model,
                "max_image_dim": self.settings.llm_max_image_dimension,
                "use_gray_mask": self.settings.llm_use_gray_mask,
            }
        return VLMOpenRouterAPIWorker(config)
    
    def should_process_image(self, image: ImageData) -> bool:
        # 若設定跳過 NSFW，檢查標籤
        if self.settings and self.settings.llm_skip_nsfw_on_batch:
            tags = image.tagger_tags or ""
            if "rating:explicit" in tags or "rating:questionable" in tags:
                return False
        return True
    
    def prepare_input(self, image: ImageData) -> WorkerInput:
        # 更新 tags context
        if self.tags_context_getter:
            image.tagger_tags = self.tags_context_getter(image.path)
        
        # 根據 prompt_mode 選擇提示詞
        user_prompt = ""
        if self.settings:
            if self.prompt_mode == "custom":
                user_prompt = self.settings.llm_custom_prompt_template
            else:
                user_prompt = self.settings.llm_user_prompt_template
        
        return WorkerInput(
            image=image,
            settings=self.settings,
            extra={
                "system_prompt": self.settings.llm_system_prompt if self.settings else "",
                "user_prompt": user_prompt,
            }
        )
