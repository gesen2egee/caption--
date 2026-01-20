# -*- coding: utf-8 -*-
"""
BaseTask 抽象基類

所有 Task 都應該繼承此類別並實作 execute 方法。
Task 負責：
1. Skip 判斷
2. Worker 選擇與配置
3. 前後處理邏輯
4. 結果整合回 ImageData
"""
from abc import ABC, abstractmethod
from typing import Tuple

from lib.pipeline.context import TaskContext, TaskResult


class BaseTask(ABC):
    """
    Task 抽象基類
    
    定義一個完整的任務，包含 skip 判斷、執行邏輯和結果整合。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Task 名稱，用於日誌和信號"""
        pass
    
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """
        判斷是否應該跳過此圖片
        
        Args:
            context: 執行上下文
            
        Returns:
            (should_skip, skip_reason)
        """
        return False, ""
    
    @abstractmethod
    def execute(self, context: TaskContext) -> TaskResult:
        """
        執行任務
        
        Args:
            context: 執行上下文
            
        Returns:
            TaskResult: 執行結果
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
