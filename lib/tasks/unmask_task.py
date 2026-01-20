# -*- coding: utf-8 -*-
"""
去背任務

單圖和批量去背任務。
"""
from typing import List, Callable, Optional

from lib.tasks.base import SingleImageTask, BatchTask
from lib.workers.base import BaseWorker, WorkerInput
from lib.workers.mask_transparent_background_local import MaskTransparentBackgroundLocalWorker
from lib.core.dataclasses import ImageData, Settings
from lib.utils.sidecar import load_image_sidecar


class UnmaskTask(SingleImageTask):
    """
    單圖去背任務
    """
    
    @property
    def name(self) -> str:
        return "unmask_single"
    
    def get_worker(self) -> BaseWorker:
        config = {}
        if self.settings:
            config = {
                "mode": self.settings.mask_remover_mode,
                "default_alpha": self.settings.mask_default_alpha,
                "default_format": self.settings.mask_default_format,
                "padding": self.settings.mask_padding,
                "blur_radius": self.settings.mask_blur_radius,
                "min_foreground_ratio": self.settings.mask_batch_min_foreground_ratio,
                "max_foreground_ratio": self.settings.mask_batch_max_foreground_ratio,
            }
        return MaskTransparentBackgroundLocalWorker(config)


class BatchUnmaskTask(BatchTask):
    """
    批量去背任務
    """
    
    def __init__(self, images: List[ImageData], settings: Settings = None,
                 background_tag_checker: Callable[[str], bool] = None):
        super().__init__(images, settings)
        self.background_tag_checker = background_tag_checker
    
    @property
    def name(self) -> str:
        return "unmask_batch"
    
    def get_worker(self) -> BaseWorker:
        config = {}
        if self.settings:
            config = {
                "mode": self.settings.mask_remover_mode,
                "default_alpha": self.settings.mask_default_alpha,
                "default_format": self.settings.mask_default_format,
                "padding": self.settings.mask_padding,
                "blur_radius": self.settings.mask_blur_radius,
                "min_foreground_ratio": self.settings.mask_batch_min_foreground_ratio,
                "max_foreground_ratio": self.settings.mask_batch_max_foreground_ratio,
            }
        return MaskTransparentBackgroundLocalWorker(config)
    
    def should_process_image(self, image: ImageData) -> bool:
        # 檢查是否已處理過
        if self.settings and self.settings.mask_batch_skip_once_processed:
            sidecar = load_image_sidecar(image.path)
            if sidecar.get("masked_background"):
                return False
        
        # 檢查是否需要 background 標籤
        if self.settings and self.settings.mask_batch_only_if_has_background_tag:
            if self.background_tag_checker:
                if not self.background_tag_checker(image.path):
                    return False
            else:
                tags = image.tagger_tags or ""
                if "background" not in tags.lower():
                    return False
        
        # 檢查是否有 scenery 標籤
        if self.settings and self.settings.mask_batch_skip_if_scenery_tag:
            tags = image.tagger_tags or ""
            if "indoors" in tags.lower() or "outdoors" in tags.lower():
                return False
        
        return True
