# -*- coding: utf-8 -*-
"""
透明背景去除 Worker (使用 transparent_background)

使用 transparent_background 進行背景移除。
支援的模式: base-nightly 等
"""
import os
import traceback
from typing import Optional, Dict

from PIL import Image
import numpy as np

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.core.dataclasses import ImageData


# 模式預設設定
MODE_PRESETS = {
    "base-nightly": {
        "mode": "base-nightly",
    },
    "base": {
        "mode": "base",
    },
}


class MaskTransparentBackgroundLocalWorker(BaseWorker):
    """
    透明背景去除 Worker
    
    使用 transparent_background 進行背景移除。
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取設定
        self.mode = self.config.get("mode", "base-nightly")
        self.default_alpha = self.config.get("default_alpha", 64)
        self.default_format = self.config.get("default_format", "webp")
        self.padding = self.config.get("padding", 1)
        self.blur_radius = self.config.get("blur_radius", 3)
        self.min_foreground_ratio = self.config.get("min_foreground_ratio", 0.3)
        self.max_foreground_ratio = self.config.get("max_foreground_ratio", 0.8)
        
        self._remover = None
    
    @property
    def name(self) -> str:
        return "mask_transparent_background_local"
    
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'MaskTransparentBackgroundLocalWorker':
        """從預設設定建立 Worker"""
        preset = MODE_PRESETS.get(preset_name, MODE_PRESETS["base-nightly"])
        config = {**preset, **kwargs}
        return cls(config)
    
    def _get_remover(self):
        """懶載入 Remover"""
        if self._remover is None:
            try:
                from transparent_background import Remover
                self._remover = Remover(mode=self.mode)
            except Exception as e:
                raise RuntimeError(f"無法載入 transparent_background: {e}")
        return self._remover
    
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行背景移除"""
        try:
            # 驗證輸入
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            
            image_data = input_data.image
            image_path = image_data.path
            
            # 載入圖片
            img = Image.open(image_path).convert("RGBA")
            
            # 取得 Remover
            remover = self._get_remover()
            
            # 執行去背
            result = remover.process(img, type="rgba")
            
            # 檢查前景比例
            alpha = np.array(result.split()[3])
            fg_ratio = np.sum(alpha > 128) / alpha.size
            
            if fg_ratio < self.min_foreground_ratio:
                return WorkerOutput(
                    success=True,
                    skipped=True,
                    skip_reason=f"前景比例過低: {fg_ratio:.2%}",
                    image=image_data,
                )
            
            if fg_ratio > self.max_foreground_ratio:
                return WorkerOutput(
                    success=True,
                    skipped=True,
                    skip_reason=f"前景比例過高: {fg_ratio:.2%}",
                    image=image_data,
                )
            
            # 應用透明度
            if self.default_alpha < 255:
                alpha_channel = result.split()[3]
                # 將不透明區域改為半透明
                alpha_array = np.array(alpha_channel)
                alpha_array = np.where(alpha_array > 128, 255, self.default_alpha)
                result.putalpha(Image.fromarray(alpha_array.astype(np.uint8)))
            
            # 儲存結果
            base, ext = os.path.splitext(image_path)
            new_path = f"{base}.{self.default_format}"
            
            if self.default_format == "webp":
                result.save(new_path, "WEBP", quality=95, lossless=True)
            else:
                result.save(new_path, "PNG")
            
            # 更新 ImageData
            image_data.masked_background = True
            if new_path != image_path:
                image_data.path = new_path
            
            return WorkerOutput(
                success=True,
                image=image_data,
                result_path=new_path,
                metadata={"foreground_ratio": fg_ratio}
            )
            
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        return None
