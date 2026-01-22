# -*- coding: utf-8 -*-
"""
Imgutils Generic Tagger Worker (Timm Models)

使用 imgutils.generic.multilabel_timm_predict 進行標籤識別。
支援自定義 Repo ID 的模型 (如 Makki2104/animetimm 系列)。
"""
import traceback
from typing import Optional, Dict

from PIL import Image

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.core.dataclasses import ImageData


# 模型預設設定
MODEL_PRESETS = {
    "EVA02_Large_Animetimm": {
        "repo_id": "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8,
    },
    "ConvNextV2_Huge_Animetimm": {
        "repo_id": "Makki2104/animetimm/convnextv2_huge.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8,
    },
    "SwinV2_Base_Animetimm": {
        "repo_id": "Makki2104/animetimm/swinv2_base_window8_256.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8,
    },
}


class TaggerImgutilsGenericWorker(BaseWorker):
    """
    Imgutils Generic Tagger Worker
    
    使用 imgutils.generic.multilabel_timm_predict 支援更多模型。
    """
    
    category = "TAGGER"
    display_name = "Local (imgutils-timm)"
    description = "Run generic Timm models via imgutils (e.g. Makki2104/animetimm)"
    default_config = {
        "repo_id": "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full",
        "general_threshold": 0.25,
        "character_threshold": 0.8
    }
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取設定
        # 這裡的 tagger_model 會對應到 UI 的設定欄位，我們將其作為 repo_id 使用
        self.repo_id = self.config.get("tagger_model") or self.config.get("repo_id") or "Makki2104/animetimm/eva02_large_patch14_448.dbv4-full"
        self.general_threshold = self.config.get("general_threshold", 0.25)
        self.character_threshold = self.config.get("character_threshold", 0.8)
    
    @property
    def name(self) -> str:
        return "tagger_imgutils_generic"
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            from imgutils.generic import multilabel_timm_predict
            return True
        except ImportError:
            return False

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'TaggerImgutilsGenericWorker':
        """從預設設定建立 Worker"""
        preset = MODEL_PRESETS.get(preset_name, MODEL_PRESETS["EVA02_Large_Animetimm"])
        config = {**preset, **kwargs}
        # 將 repo_id 對應到 tagger_model 以符合 UI 慣例
        if "repo_id" in config:
            config["tagger_model"] = config.pop("repo_id")
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
            from imgutils.generic import multilabel_timm_predict
            from imgutils.tagging import tags_to_text, remove_underline
            
            # 執行預測
            # 根據使用者提供的 snippet，fmt 參數控制回傳內容
            # repo_id 支援 "user/repo" 或 "user/repo/subdir" 格式 (取決於 imgutils 實作)
            general, character, rating = multilabel_timm_predict(
                img,
                repo_id=self.repo_id,
                fmt=("general", "character", "rating"),
            )
            
            # 過濾閾值
            general = {k: v for k, v in general.items() if v >= self.general_threshold}
            character = {k: v for k, v in character.items() if v >= self.character_threshold}
            
            # 正規化標籤 (移除底線)
            features = {remove_underline(k): v for k, v in general.items()}
            chars = {remove_underline(k): v for k, v in character.items()}
            
            # 轉換為文字
            all_tags = {**features, **chars}
            tags_text = tags_to_text(all_tags)
            
            # 更新 ImageData
            image_data.tagger_tags = tags_text
            image_data.tagger_rating = rating # rating 是一組分數，如 {'safe': 0.9, ...}
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
