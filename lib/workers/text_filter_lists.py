# -*- coding: utf-8 -*-
"""
文字黑白名單過濾 Worker

過濾標籤或句子中的特徵內容。
"""
from typing import Optional, Dict, List

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput


class TextFilterListsWorker(BaseWorker):
    """
    文字黑白名單過濾 Worker
    
    根據黑白名單判斷文字是否為特徵內容。
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # 從 config 讀取黑白名單
        self.blacklist = self.config.get("blacklist", [])
        self.whitelist = self.config.get("whitelist", [])
    
    @property
    def name(self) -> str:
        return "text_filter_lists"
    
    def is_filtered(self, text: str) -> bool:
        """
        判斷文字是否符合過濾條件。
        
        邏輯：包含任何黑名單詞，且不包含任何白名單詞。
        """
        if not text:
            return False
        
        t = text.strip().lower()
        
        # 檢查黑名單
        has_blacklist = any(bw.lower() in t for bw in self.blacklist if bw)
        if not has_blacklist:
            return False
        
        # 檢查白名單
        has_whitelist = any(ww.lower() in t for ww in self.whitelist if ww)
        if has_whitelist:
            return False
        
        return True
    
    def filter_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """
        過濾標籤列表。
        
        Returns:
            {"keep": [...], "remove": [...]}
        """
        keep = []
        remove = []
        
        for tag in tags:
            if self.is_filtered(tag):
                remove.append(tag)
            else:
                keep.append(tag)
        
        return {"keep": keep, "remove": remove}
    
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行過濾"""
        try:
            # 從 extra 取得要過濾的文字或標籤
            text = input_data.extra.get("text")
            tags = input_data.extra.get("tags", [])
            
            if text:
                is_filtered = self.is_filtered(text)
                return WorkerOutput(
                    success=True,
                    result_data={"is_filtered": is_filtered, "text": text}
                )
            
            if tags:
                result = self.filter_tags(tags)
                return WorkerOutput(
                    success=True,
                    result_data=result
                )
            
            return WorkerOutput(success=False, error="缺少 text 或 tags 輸入")
            
        except Exception as e:
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.extra.get("text") and not input_data.extra.get("tags"):
            return "缺少 text 或 tags 輸入"
        return None
