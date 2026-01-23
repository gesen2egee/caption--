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
            
        if context.extra.get("force_execution", False):
            return False, ""
        
        # 已處理過
        if settings.mask_batch_skip_once_processed and image.masked_text:
            return True, "已去字"
        
        # 需要 background 標籤
        if settings.mask_batch_skip_once_processed and image.masked_text:
            return True, "已去字"
        
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
            
            # Worker returns updated image status
            boxes = worker_output.result_data.get("box_count", 0) if worker_output.result_data else 0
            new_path = worker_output.result_data.get("result_path", image_path) if worker_output.result_data else image_path
            
            # --- Advanced Post-Processing ---
            if settings and os.path.exists(new_path):
                from lib.utils.image_processing import apply_advanced_mask_processing
                from PIL import Image
                
                # Check if we need to process
                need_proc = any([
                    settings.mask_text_shrink_size > 0,
                    settings.mask_text_blur_radius > 0,
                    settings.mask_text_min_alpha > 0,
                    # We always check fill white? User said "原始alpha alpha=0..." 
                    # Let's assume we enforce it if any mask op is done or just always?
                    # User request: "原始alpha alpha=0像素填補白色...".
                    # I'll do it as part of processing.
                    True 
                ])
                
                if need_proc:
                    try:
                        with Image.open(new_path) as img:
                            img = img.convert("RGBA")
                            
                            # Apply advanced processing to Alpha
                            # Note: This processes the WHOLE alpha channel of the result.
                            processed_alpha = apply_advanced_mask_processing(
                                img, 
                                img.getchannel("A"), # Use current alpha as mask
                                shrink=settings.mask_text_shrink_size,
                                blur=settings.mask_text_blur_radius,
                                min_alpha=settings.mask_text_min_alpha
                            )
                            
                            # Put alpha back
                            img.putalpha(processed_alpha)
                            
                            # "alpha=0 像素填補白色" logic
                            # Create white background
                            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                            # Composite: If alpha is 0, show white.
                            # Actually, we want to CHANGE the RGB of the pixel to White if Alpha is 0.
                            # But keep the Alpha as 0.
                            # So we want (255,255,255,0) where it was (r,g,b,0).
                            # This helps some Inpainting models.
                            
                            datas = img.getdata()
                            new_data = []
                            for item in datas:
                                # item is (r,g,b,a)
                                if item[3] == 0:
                                    new_data.append((255, 255, 255, 0))
                                else:
                                    new_data.append(item)
                            img.putdata(new_data)
                            
                            img.save(new_path)
                            
                    except Exception as e:
                        print(f"Post-processing failed: {e}")
                        traceback.print_exc()

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
