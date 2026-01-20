# -*- coding: utf-8 -*-
"""
Batch 類別

Batch 包含圖片列表和任務類型，對每張圖執行對應的 Worker。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

from lib.core.dataclasses import ImageData, Settings
from lib.workers.base import WorkerInput, WorkerOutput


class TaskType(Enum):
    """任務類型枚舉"""
    TAGGER = "tagger"
    LLM = "llm"
    UNMASK = "unmask"
    MASK_TEXT = "mask_text"
    RESTORE = "restore"
    OCR = "ocr"


class Batch(QObject):
    """
    Batch 類別
    
    包含圖片列表，對每張圖執行指定類型的 Worker。
    單圖和批量的唯一差異是 images 列表的長度。
    """
    
    # 信號
    progress = pyqtSignal(int, int, str)      # (current, total, filename)
    image_done = pyqtSignal(str, object)      # (image_path, WorkerOutput)
    done = pyqtSignal()                        # Batch 完成
    error = pyqtSignal(str)                    # 錯誤
    
    def __init__(self, 
                 task_type: TaskType,
                 images: List[ImageData],
                 settings: Settings = None,
                 options: Dict[str, Any] = None,
                 parent=None):
        super().__init__(parent)
        self.task_type = task_type
        self.images = images
        self.settings = settings
        self.options = options or {}
        self._stop = False
        self._results: List[WorkerOutput] = []
    
    def stop(self):
        """請求中止"""
        self._stop = True
    
    def is_stopped(self) -> bool:
        """是否已中止"""
        return self._stop
    
    @property
    def results(self) -> List[WorkerOutput]:
        """取得所有結果"""
        return self._results
    
    def _create_worker(self):
        """根據 task_type 建立對應的 Worker"""
        if self.task_type == TaskType.TAGGER:
            from lib.workers.tagger_imgutils_tagging_local import TaggerImgutilsTaggingLocalWorker
            config = {}
            if self.settings:
                config = {
                    "model_name": self.settings.tagger_model,
                    "general_threshold": self.settings.general_threshold,
                    "general_mcut_enabled": self.settings.general_mcut_enabled,
                    "character_threshold": self.settings.character_threshold,
                    "character_mcut_enabled": self.settings.character_mcut_enabled,
                    "drop_overlap": self.settings.drop_overlap,
                }
            return TaggerImgutilsTaggingLocalWorker(config)
        
        elif self.task_type == TaskType.LLM:
            from lib.workers.vlm_openrouter_api import VLMOpenRouterAPIWorker
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
        
        elif self.task_type == TaskType.UNMASK:
            from lib.workers.mask_transparent_background_local import MaskTransparentBackgroundLocalWorker
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
        
        elif self.task_type == TaskType.RESTORE:
            from lib.workers.image_restore_raw import ImageRestoreRawWorker
            return ImageRestoreRawWorker()
        
        elif self.task_type == TaskType.OCR:
            from lib.workers.detect_imgutils_ocr_local import DetectImgutilsOCRLocalWorker
            config = {}
            if self.settings:
                config = {
                    "max_candidates": self.settings.mask_ocr_max_candidates,
                    "heat_threshold": self.settings.mask_ocr_heat_threshold,
                    "box_threshold": self.settings.mask_ocr_box_threshold,
                    "unclip_ratio": self.settings.mask_ocr_unclip_ratio,
                }
            return DetectImgutilsOCRLocalWorker(config)
        
        else:
            raise ValueError(f"未知的任務類型: {self.task_type}")
    
    def _should_process(self, image: ImageData) -> tuple[bool, str]:
        """
        判斷是否應該處理此圖片
        
        Returns:
            (should_process, skip_reason)
        """
        # 根據任務類型和設定判斷
        if self.task_type == TaskType.UNMASK and self.settings:
            # 檢查是否已處理過
            if self.settings.mask_batch_skip_once_processed and image.masked_background:
                return False, "已去背"
            
            # 檢查是否需要 background 標籤
            if self.settings.mask_batch_only_if_has_background_tag:
                tags = (image.tagger_tags or "").lower()
                if "background" not in tags:
                    return False, "無 background 標籤"
            
            # 檢查 scenery 標籤
            if self.settings.mask_batch_skip_if_scenery_tag:
                tags = (image.tagger_tags or "").lower()
                if "indoors" in tags or "outdoors" in tags:
                    return False, "包含 scenery 標籤"
        
        elif self.task_type == TaskType.LLM and self.settings:
            # 檢查 NSFW 跳過
            if self.settings.llm_skip_nsfw_on_batch:
                tags = (image.tagger_tags or "").lower()
                if "explicit" in tags or "questionable" in tags:
                    return False, "NSFW 內容"
        
        return True, ""
    
    def run(self):
        """執行 Batch"""
        try:
            worker = self._create_worker()
            total = len(self.images)
            
            for i, image in enumerate(self.images):
                if self._stop:
                    break
                
                self.progress.emit(i + 1, total, image.filename)
                
                # 檢查是否應該處理
                should_process, skip_reason = self._should_process(image)
                if not should_process:
                    output = WorkerOutput(
                        success=True,
                        skipped=True,
                        skip_reason=skip_reason,
                        image=image,
                    )
                    self._results.append(output)
                    self.image_done.emit(image.path, output)
                    continue
                
                # 執行 Worker
                try:
                    input_data = WorkerInput(
                        image=image,
                        settings=self.settings,
                        extra=self.options,
                    )
                    output = worker.process(input_data)
                    self._results.append(output)
                    self.image_done.emit(image.path, output)
                    
                except Exception as e:
                    output = WorkerOutput(success=False, error=str(e), image=image)
                    self._results.append(output)
                    self.image_done.emit(image.path, output)
            
            self.done.emit()
            
        except Exception as e:
            self.error.emit(str(e))
