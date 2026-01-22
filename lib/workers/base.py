# -*- coding: utf-8 -*-
"""
Worker 基礎類別和資料結構

Worker 是純粹的功能/模型包裝，不涉及 UI 互動。
命名規範: 功能分類_來源_local或api.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta


@dataclass
class WorkerInput:
    """
    Worker 統一輸入資料結構
    
    使用核心 dataclass，Worker 根據需要解析使用。
    沒有的欄位為 None。
    """
    image: Optional[ImageData] = None         # 圖片資料 (單張)
    images: Optional[List[ImageData]] = None  # 圖片資料列表 (批量)
    settings: Optional[Settings] = None       # 應用程式設定
    prompt: Optional[Prompt] = None           # Pipeline 指令
    folder: Optional[FolderMeta] = None       # 資料夾 Meta (未來用)
    
    # 額外參數 (Worker 特定)
    extra: Dict = field(default_factory=dict)


@dataclass
class WorkerOutput:
    """
    Worker 統一輸出資料結構
    """
    success: bool                             # 是否成功
    
    # 輸出資料
    image: Optional[ImageData] = None         # 更新後的圖片資料
    images: Optional[List[ImageData]] = None  # 更新後的圖片資料列表
    result_text: Optional[str] = None         # 文字結果
    result_data: Optional[Dict] = None        # 結構化資料結果
    
    # 狀態
    error: Optional[str] = None               # 錯誤訊息
    skipped: bool = False                     # 是否跳過 (不算失敗)
    skip_reason: Optional[str] = None         # 跳過原因
    
    # 元資料
    metadata: Dict = field(default_factory=dict)


class BaseWorker(ABC):
    """
    Worker 抽象基礎類別
    
    所有 Worker 都應該繼承此類別並實作 process 方法。
    Worker 是無狀態的純功能單元，不涉及 UI 互動。
    """
    
    # Metadata (子類別需覆寫)
    category: str = "OTHER"   # "TAGGER", "LLM", "UNMASK", "MASK_TEXT", "RESTORE"
    display_name: str = "Base Worker"
    description: str = ""
    default_config: Dict = {}
    
    def __init__(self, config: Dict = None):
        """
        初始化 Worker
        
        Args:
            config: Worker 設定 (模型參數、閾值等)
        """
        self.config = config or {}
        # Merge with defaults if not present
        for k, v in self.default_config.items():
            if k not in self.config:
                self.config[k] = v
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Worker 唯一識別名 (system name)"""
        pass

    @classmethod
    def is_available(cls) -> bool:
        """
        檢查此 Worker 是否可用 (例如檢查依賴、模型檔案是否存在)
        
        Returns:
            bool: 是否可用
        """
        return True
    
    @abstractmethod
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """
        執行處理
        
        Args:
            input_data: 輸入資料
            
        Returns:
            WorkerOutput: 處理結果
        """
        pass
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        """
        驗證輸入資料
        
        Returns:
            錯誤訊息，若驗證通過則回傳 None
        """
        return None
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
