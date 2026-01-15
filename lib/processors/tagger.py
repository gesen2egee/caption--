
from typing import Tuple, Any
from .base import BaseProcessor
from ..data import ImageContext
from ..services.tagger import call_wd14
from ..services.common import unload_all_models

class TaggerProcessor(BaseProcessor):
    def process(self, ctx: ImageContext) -> Tuple[bool, Any]:
        """
        執行 WD14 標註並寫入 Sidecar JSON。
        """
        img = ctx.get_image()
        
        # 呼叫 Service
        rating, features, chars = call_wd14(img, self.settings)
        
        # 處理結果
        rating_key = max(rating, key=rating.get)
        rating_tag = f"rating:{rating_key}"
        
        # 組合標籤字串 (Rating + Chars + Features)
        # 這裡照舊邏輯組合，方便顯示
        char_tags = list(chars.keys())
        feature_tags = list(features.keys())
        all_tags_list = [rating_tag] + char_tags + feature_tags
        tags_str = ", ".join(all_tags_list)
        
        # 更新 Sidecar (資料持久化)
        sc = ctx.sidecar
        sc["rating"] = rating_key # 存原始 rating 鍵值 (如 general, explicit)
        
        # 這裡我們可以存分開的 list，也可以存合在一起的字串。
        # 依照舊程式邏輯，JSON 裡通常存 `tagger_tags` 作為字串?
        # 舊程式: sidecar["tagger_tags"] = tags_str (在 callback 裡做的)
        sc["tagger_tags"] = tags_str 
        
        ctx.save_sidecar()
        
        return True, tags_str

    def cleanup(self):
        unload_all_models()
