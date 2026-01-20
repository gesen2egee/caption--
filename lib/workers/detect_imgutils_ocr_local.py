# -*- coding: utf-8 -*-
"""
OCR 文字偵測 Worker (使用 imgutils)

使用 imgutils.ocr 進行文字區域偵測。
"""
import traceback
from typing import Optional, Dict, List, Tuple

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput


class DetectImgutilsOCRLocalWorker(BaseWorker):
    """
    OCR 文字偵測 Worker
    
    使用 imgutils.ocr 偵測圖片中的文字區域。
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取設定
        self.max_candidates = self.config.get("max_candidates", 300)
        self.heat_threshold = self.config.get("heat_threshold", 0.2)
        self.box_threshold = self.config.get("box_threshold", 0.6)
        self.unclip_ratio = self.config.get("unclip_ratio", 2.3)
    
    @property
    def name(self) -> str:
        return "detect_imgutils_ocr_local"
    
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行文字偵測"""
        try:
            # 驗證輸入
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            
            image_data = input_data.image
            
            # 嘗試匯入 imgutils.ocr
            try:
                from imgutils.ocr import detect_text_with_ocr
            except ImportError:
                return WorkerOutput(
                    success=True,
                    skipped=True,
                    skip_reason="imgutils.ocr 未安裝",
                    image=image_data,
                )
            
            # 執行偵測
            boxes = detect_text_with_ocr(
                image_data.path,
                max_candidates=self.max_candidates,
                heat_threshold=self.heat_threshold,
                box_threshold=self.box_threshold,
                unclip_ratio=self.unclip_ratio,
            )
            
            # boxes 格式: [(x1, y1, x2, y2), ...]
            return WorkerOutput(
                success=True,
                image=image_data,
                result_data={"text_boxes": boxes},
                metadata={"box_count": len(boxes)}
            )
            
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        return None
