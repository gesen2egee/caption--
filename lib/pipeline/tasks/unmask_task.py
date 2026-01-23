# -*- coding: utf-8 -*-
"""
UnmaskTask - 去背任務

使用 transparent-background 移除圖片背景。
"""
import os
import traceback
from typing import Tuple

from PIL import Image
import numpy as np

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar
from lib.utils.file_ops import backup_raw_image


class UnmaskTask(BaseTask):
    """
    去背任務
    
    使用 transparent-background 模型移除圖片背景，
    並根據設定進行前景比例檢查和 Alpha 調整。
    """
    
    @property
    def name(self) -> str:
        return "unmask"
    
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """檢查是否應跳過"""
        settings = context.settings
        image = context.image
        
        if not settings:
            return False, ""
            
        if context.extra.get("force_execution", False):
            return False, ""
        
        # 已處理過
        if settings.mask_batch_skip_once_processed and image.masked_background:
            return True, "已去背"
        
        # 需要 background 標籤
        if settings.mask_batch_only_if_has_background_tag:
            tags = (image.tagger_tags or "").lower()
            if "background" not in tags:
                return True, "無 background 標籤"
        
        # 跳過 scenery 標籤
        if settings.mask_batch_skip_if_scenery_tag:
            tags = (image.tagger_tags or "").lower()
            if "indoors" in tags or "outdoors" in tags:
                return True, "包含 scenery 標籤"
        
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """執行去背"""
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
            
            # 3. 建立並呼叫 Worker
            # from lib.workers.mask_transparent_background_local import MaskTransparentBackgroundLocalWorker
            from lib.workers.registry import get_registry
            
            worker_name = settings.unmask_worker if (settings and settings.unmask_worker) else "mask_transparent_background_local"
            WorkerCls = get_registry().get_worker_class("UNMASK", worker_name)
            
            if not WorkerCls:
                 return TaskResult(success=False, error=f"Unmask Worker '{worker_name}' not found", image=context.image)

            config = {}
            if settings:
                config = {
                    "mode": settings.mask_remover_mode,
                    "default_alpha": settings.mask_default_alpha,
                    "default_format": settings.mask_default_format,
                    "padding": settings.mask_padding,
                    "blur_radius": settings.mask_blur_radius,
                    "min_foreground_ratio": settings.mask_batch_min_foreground_ratio,
                    "max_foreground_ratio": settings.mask_batch_max_foreground_ratio,
                }
            
            worker = WorkerCls(config)
            worker_output = worker.process(context.to_worker_input())
            
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
            
            # 4. 更新 sidecar
            new_path = worker_output.result_data.get("result_path", image_path) if worker_output.result_data else image_path
            sidecar = load_image_sidecar(new_path)
            sidecar["masked_background"] = True
            save_image_sidecar(new_path, sidecar)
            
            # 更新 ImageData
            context.image.masked_background = True
            context.image.path = new_path
            
            return TaskResult(
                success=True,
                image=context.image,
                result_data={
                    "original_path": image_path,
                    "result_path": new_path,
                    "foreground_ratio": worker_output.result_data.get("foreground_ratio") if worker_output.result_data else None,
                },
            )
            
        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                image=context.image,
            )
