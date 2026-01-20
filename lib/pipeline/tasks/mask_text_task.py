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
            
            # 3. OCR 偵測
            try:
                from imgutils.ocr import detect_text_with_ocr
            except ImportError:
                return TaskResult(
                    success=False,
                    error="imgutils.ocr 未安裝",
                    image=context.image,
                )
            
            img = Image.open(image_path).convert("RGBA")
            rgb_img = img.convert("RGB")
            
            ocr_config = {}
            if settings:
                ocr_config = {
                    "max_candidates": int(settings.mask_ocr_max_candidates),
                    "heat_threshold": float(settings.mask_ocr_heat_threshold),
                    "box_threshold": float(settings.mask_ocr_box_threshold),
                    "unclip_ratio": float(settings.mask_ocr_unclip_ratio),
                }
            
            try:
                boxes = detect_text_with_ocr(rgb_img, **ocr_config)
            except TypeError:
                boxes = detect_text_with_ocr(rgb_img)
            
            if not boxes:
                return TaskResult(
                    success=True,
                    image=context.image,
                    result_data={"original_path": image_path, "box_count": 0},
                )
            
            # 4. 建立遮罩並套用 Alpha
            alpha_orig = img.split()[3]
            text_mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(text_mask)
            
            target_alpha_val = settings.mask_text_alpha if settings else 10
            
            for box in boxes:
                # 處理嵌套結構
                if len(box) > 0 and isinstance(box[0], (list, tuple)):
                    actual_box = box[0]
                else:
                    actual_box = box
                box_int = tuple(int(round(float(c))) for c in actual_box[:4])
                draw.rectangle(box_int, fill=255)
            
            # 合成新 Alpha
            target_alpha_layer = Image.new("L", img.size, target_alpha_val)
            new_alpha = Image.composite(target_alpha_layer, alpha_orig, text_mask)
            
            # 重組 RGBA
            r, g, b, _ = img.split()
            output_img = Image.merge("RGBA", (r, g, b, new_alpha))
            
            # 5. 儲存
            default_format = settings.mask_default_format if settings else "webp"
            base, ext = os.path.splitext(image_path)
            new_path = f"{base}.{default_format}"
            
            if default_format == "webp":
                output_img.save(new_path, "WEBP", quality=95, lossless=True)
            else:
                output_img.save(new_path, "PNG")
            
            # 6. 更新 sidecar
            sidecar = load_image_sidecar(new_path)
            sidecar["masked_text"] = True
            save_image_sidecar(new_path, sidecar)
            
            # 更新 ImageData
            context.image.masked_text = True
            context.image.path = new_path
            
            return TaskResult(
                success=True,
                image=context.image,
                result_data={
                    "original_path": image_path,
                    "result_path": new_path,
                    "box_count": len(boxes),
                },
            )
            
        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                image=context.image,
            )
