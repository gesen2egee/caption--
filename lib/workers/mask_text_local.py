# -*- coding: utf-8 -*-
import os
import traceback
from typing import Optional, Dict
from PIL import Image, ImageDraw, ImageFilter

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.utils.file_ops import backup_raw_image
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar

class MaskTextLocalWorker(BaseWorker):
    """
    Mask Text Worker
    
    使用 OCR 偵測文字並將其模糊化/移除。
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.max_candidates = self.config.get("mask_ocr_max_candidates", 300)
        self.heat_threshold = self.config.get("mask_ocr_heat_threshold", 0.2)
        self.box_threshold = self.config.get("mask_ocr_box_threshold", 0.6)
        self.unclip_ratio = self.config.get("mask_ocr_unclip_ratio", 2.3)
        self.default_format = self.config.get("mask_default_format", "webp")
        self.blur_radius = 15

    @property
    def name(self) -> str:
        return "mask_text_local"

    def process(self, input_data: WorkerInput) -> WorkerOutput:
        try:
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")

            image_data = input_data.image
            image_path = image_data.path
            
            # Backup
            backup_raw_image(image_path)
            
            # OCR
            try:
                from imgutils.ocr import detect_text_with_ocr
            except ImportError:
                return WorkerOutput(success=False, error="imgutils.ocr 未安裝")
                
            img = Image.open(image_path).convert("RGBA")
            rgb_img = img.convert("RGB")
            
            try:
                boxes = detect_text_with_ocr(
                    rgb_img,
                    max_candidates=int(self.max_candidates),
                    heat_threshold=float(self.heat_threshold),
                    box_threshold=float(self.box_threshold),
                    unclip_ratio=float(self.unclip_ratio)
                )
            except TypeError:
                # Fallback for older versions if arguments differ
                boxes = detect_text_with_ocr(rgb_img)

            if not boxes:
                 return WorkerOutput(
                     success=True, 
                     result_data={"original_path": image_path, "box_count": 0}, 
                     image=image_data
                 )
                 
            # Prepare Alpha Mask
            # 0 = Transparent, 255 = Opaque
            # PIL "L" mode: 0 is black, 255 is white.
            # We want detected text areas to have a specific alpha value.
            
            # Initial alpha from image
            alpha_orig = img.split()[3]
            
            # Create a "burn" mask for text areas
            # We start with a copy of original alpha
            text_mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(text_mask)
            
            target_alpha_val = int(self.config.get("default_alpha", 64))
            
            for box in boxes:
                if len(box) > 0 and isinstance(box[0], (list, tuple)):
                    actual_box = box[0]
                else:
                    actual_box = box
                box_int = tuple(int(round(float(c))) for c in actual_box[:4])
                # Fill text area in text_mask with 255 (meaning "this is text")
                draw.rectangle(box_int, fill=255)
            
            # Now we create the new alpha channel
            # Where text_mask is 255, we want target_alpha_val
            # Where text_mask is 0, we want original alpha
            
            target_alpha_layer = Image.new("L", img.size, target_alpha_val)
            new_alpha = Image.composite(target_alpha_layer, alpha_orig, text_mask)
            
            # Reconstruct RGBA image with original RGB and new Alpha
            r, g, b, _ = img.split()
            output_img = Image.merge("RGBA", (r, g, b, new_alpha))
            
            # Save
            base, ext = os.path.splitext(image_path)
            new_path = f"{base}.{self.default_format}"
            
            if self.default_format == "webp":
                output_img.save(new_path, "WEBP", quality=95, lossless=True)
            else:
                output_img.save(new_path, "PNG")
                
            # Update Sidecar
            sidecar = load_image_sidecar(image_path)
            sidecar["masked_text"] = True
            save_image_sidecar(new_path, sidecar)
            
            image_data.masked_text = True
            
            return WorkerOutput(
                success=True,
                image=image_data,
                result_data={
                    "original_path": image_path,
                    "result_path": new_path,
                    "box_count": len(boxes)
                }
            )

        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))

    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        return None
