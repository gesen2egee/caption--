# -*- coding: utf-8 -*-
"""
MaskTextTask - 去文字任務

使用 OCR 偵測文字並將其區域設為透明。
"""
import os
import traceback
from typing import Tuple

from PIL import Image, ImageDraw

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar
from lib.utils.file_ops import backup_raw_image


class MaskTextTask(BaseTask):
    """
    去文字任務
    
    使用 OCR 偵測圖片中的文字區域，並將這些區域的 Alpha 通道設為指定值。
    """
    
    @property
    def name(self) -> str:
        return "mask_text"
    
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """檢查是否應跳過"""
        settings = context.settings
        image = context.image
        
        if not settings:
            return False, ""
        
        # 已處理過
        if settings.mask_batch_skip_once_processed and image.masked_text:
            return True, "已去字"
        
        # 需要 background 標籤
        if settings.mask_batch_only_if_has_background_tag:
            tags = (image.tagger_tags or "").lower()
            if "background" not in tags:
                return True, "無 background 標籤"
        
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """執行去文字"""
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
            
            image_path = context.image.path
            settings = context.settings
            
            # 2. 備份原圖
            backup_raw_image(image_path)
            
            # 3. 呼叫 Worker
            from lib.workers.registry import get_registry
            
            worker_name = settings.mask_text_worker if (settings and settings.mask_text_worker) else "mask_text_local"
            WorkerCls = get_registry().get_worker_class("MASK_TEXT", worker_name)
            
            if not WorkerCls:
                 return TaskResult(success=False, error=f"Mask Text Worker '{worker_name}' not found", image=context.image)

            config = {}
            if settings:
                config = {
                    "mask_ocr_max_candidates": int(settings.mask_ocr_max_candidates),
                    "mask_ocr_heat_threshold": float(settings.mask_ocr_heat_threshold),
                    "mask_ocr_box_threshold": float(settings.mask_ocr_box_threshold),
                    "mask_ocr_unclip_ratio": float(settings.mask_ocr_unclip_ratio),
                    "default_alpha": settings.mask_text_alpha, # Corrected key usage
                    "mask_default_format": settings.mask_default_format
                }
            
            worker = WorkerCls(config)
            worker_output = worker.process(context.to_worker_input())
            
            if not worker_output.success:
                return TaskResult(success=False, error=worker_output.error, image=context.image)
                
            # Worker returns updated image status, but maybe we need to reload sidecar or handle result data
            # NOTE: The original local worker implementation handles file saving.
            # We just need to sync back the result.
            
            boxes = worker_output.result_data.get("box_count", 0) if worker_output.result_data else 0
            new_path = worker_output.result_data.get("result_path", image_path) if worker_output.result_data else image_path
            
            # Since worker handles saving, we just update context
            context.image.masked_text = True
            context.image.path = new_path
             
            return TaskResult(
                success=True,
                image=context.image,
                result_data={
                    "original_path": image_path,
                    "result_path": new_path,
                    "box_count": boxes,
                },
            )
            
        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                image=context.image,
            )
