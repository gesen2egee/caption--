# -*- coding: utf-8 -*-
"""
WD14 Tagger Worker (使用 imgutils)

使用 imgutils.tagging 的 WD14 標籤模型。
支援的模型: EVA02_Large, SwinV2_v3 等
"""
import traceback
from typing import Optional, Dict

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.core.dataclasses import ImageData


# 模型預設設定
MODEL_PRESETS = {
    "EVA02_Large": {
        "model_name": "EVA02_Large",
        "general_threshold": 0.2,
        "character_threshold": 0.85,
    },
    "SwinV2_v3": {
        "model_name": "SwinV2_v3",
        "general_threshold": 0.35,
        "character_threshold": 0.85,
    },
}


class TaggerImgutilsTaggingLocalWorker(BaseWorker):
    """
    WD14 Tagger Worker
    
    使用 imgutils.tagging 進行圖片標籤識別。
    """
    
    category = "TAGGER"
    display_name = "Local (imgutils)"
    description = "Run WD14 models locally using imgutils"
    default_config = {
        "model_name": "EVA02_Large",
        "general_threshold": 0.2,
        "character_threshold": 0.85
    }
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取設定
        self.model_name = self.config.get("model_name", "EVA02_Large")
        self.general_threshold = self.config.get("general_threshold", 0.2)
        self.general_mcut_enabled = self.config.get("general_mcut_enabled", False)
        self.character_threshold = self.config.get("character_threshold", 0.85)
        self.character_mcut_enabled = self.config.get("character_mcut_enabled", True)
        self.drop_overlap = self.config.get("drop_overlap", True)
    
    @property
    def name(self) -> str:
        return "tagger_imgutils_tagging_local"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import imgutils.tagging
            return True
        except ImportError:
            return False
    
    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'TaggerImgutilsTaggingLocalWorker':
        """從預設設定建立 Worker"""
        preset = MODEL_PRESETS.get(preset_name, MODEL_PRESETS["EVA02_Large"])
        config = {**preset, **kwargs}
        return cls(config)
    
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行標籤識別"""
        try:
            # 驗證輸入
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            
            image_data = input_data.image
            
            # 載入圖片
            img = Image.open(image_data.path)
            
            # 呼叫 imgutils
            from imgutils.tagging import get_wd14_tags, tags_to_text, remove_underline
            import inspect
            
            # 準備參數
            kwargs = {
                "model_name": self.model_name,
                "general_threshold": self.general_threshold,
                "general_mcut_enabled": self.general_mcut_enabled,
                "character_threshold": self.character_threshold,
                "character_mcut_enabled": self.character_mcut_enabled,
                "drop_overlap": self.drop_overlap,
            }
            
            # 只使用支援的參數
            try:
                sig = inspect.signature(get_wd14_tags)
                allowed = set(sig.parameters.keys())
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            except Exception:
                pass
            
            # 執行標籤識別
            rating, features, chars = get_wd14_tags(img, **kwargs)
            
            # 正規化標籤 (移除底線)
            features = {remove_underline(k): v for k, v in features.items()}
            chars = {remove_underline(k): v for k, v in chars.items()}
            
            # 轉換為文字
            all_tags = {**features, **chars}
            tags_text = tags_to_text(all_tags)
            
            # 更新 ImageData
            image_data.tagger_tags = tags_text
            image_data.tagger_rating = rating
            image_data.tagger_features = features
            image_data.tagger_chars = chars
            
            return WorkerOutput(
                success=True,
                image=image_data,
                result_text=tags_text,
                result_data={
                    "rating": rating,
                    "features": features,
                    "chars": chars,
                }
            )
            
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        return None
