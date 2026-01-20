# -*- coding: utf-8 -*-
"""
Pipeline 上下文與結果資料結構

TaskContext: 傳入 Task 的統一容器
TaskResult: Task 回傳的統一結果
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta
from lib.workers.base import WorkerInput


@dataclass
class TaskContext:
    """
    Task 執行上下文
    
    統一的資料容器，包含執行 Task 所需的所有資訊。
    即使某些欄位沒用到，也一律傳遞以保持一致性。
    """
    image: ImageData                          # 必填：當前處理的圖片
    settings: Settings                        # 必填：應用程式設定
    prompt: Optional[Prompt] = None           # Pipeline 指令（LLM 用）
    folder: Optional[FolderMeta] = None       # 資料夾 Meta（未來用）
    
    # 額外參數 (Task 特定)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_worker_input(self) -> WorkerInput:
        """轉換為 Worker 需要的格式"""
        return WorkerInput(
            image=self.image,
            settings=self.settings,
            prompt=self.prompt,
            folder=self.folder,
            extra=self.extra,
        )


@dataclass
class TaskResult:
    """
    Task 執行結果
    
    統一的結果結構，包含執行狀態和輸出資料。
    """
    success: bool                             # 是否成功
    
    # 輸出資料
    image: Optional[ImageData] = None         # 更新後的圖片資料
    result_text: Optional[str] = None         # 文字結果（Tagger/LLM）
    result_data: Optional[Dict] = None        # 結構化資料結果
    
    # 狀態
    skipped: bool = False                     # 是否跳過（不算失敗）
    skip_reason: Optional[str] = None         # 跳過原因
    error: Optional[str] = None               # 錯誤訊息
    
    # 元資料
    metadata: Dict[str, Any] = field(default_factory=dict)
