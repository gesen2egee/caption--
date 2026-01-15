
from typing import Any, Tuple
from ..data import ImageContext, AppSettings

class BaseProcessor:
    """
    所有批量處理邏輯的基底類別。
    子類別必須實作 process 方法。
    """
    def __init__(self, settings: AppSettings):
        self.settings = settings
    
    def prepare(self):
        """Batch 開始前執行 (例如載入模型)"""
        pass

    def cleanup(self):
        """Batch 結束後執行 (例如卸載模型)"""
        pass

    def process(self, ctx: ImageContext) -> Tuple[bool, Any]:
        """
        處理單張圖片。
        回傳: (is_changed: bool, result_data: Any)
        - is_changed: 是否有更動 (用於統計或後續處理)
        - result_data: 要透過 Signal 傳回 UI 的數據 (例如生成的標籤字串)
        """
        raise NotImplementedError
