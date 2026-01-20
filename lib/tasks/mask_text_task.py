from typing import List, Optional
from lib.core.dataclasses import ImageData
from lib.workers.base import BaseWorker, WorkerInput
from lib.workers.mask_text_local import MaskTextLocalWorker
from lib.tasks.base import SingleImageTask, BatchTask

class MaskTextTask(SingleImageTask):
    def __init__(self, image: ImageData, config: dict = None):
        super().__init__(image, config)
    
    def create_worker(self) -> BaseWorker:
        return MaskTextLocalWorker(self.config)

class BatchMaskTextTask(BatchTask):
    def __init__(self, images: List[ImageData], config: dict = None):
        super().__init__(images, config)
    
    def create_worker(self) -> BaseWorker:
        return MaskTextLocalWorker(self.config)
    
    def _should_process(self, image: ImageData) -> bool:
        # 檢查是否已處理
        if self.config.get("mask_batch_skip_once_processed", True) and image.masked_text:
            return False
            
        # 檢查是否僅處理包含 background 標籤 (複用去背的設定? 或是需要新的?)
        # 假設 mask text 有自己的設定 mask_batch_only_if_has_background_tag?
        # 通常去文字依賴於OCR，不一定依賴 background tag
        # 但 UI 上可能有共用設置
        
        return True
