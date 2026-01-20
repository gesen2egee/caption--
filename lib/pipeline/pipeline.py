# -*- coding: utf-8 -*-
"""
Pipeline 類別

Pipeline 定義執行哪些 Batch，管理整體流程。
每個功能按鈕都是一個 Pipeline。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from PyQt6.QtCore import QThread, pyqtSignal

from lib.core.dataclasses import ImageData, Settings
from lib.pipeline.batch import Batch, TaskType
from lib.workers.base import WorkerOutput


class Pipeline(QThread):
    """
    Pipeline 類別
    
    管理多個 Batch 的執行流程。
    繼承 QThread 以便在背景執行。
    """
    
    # 信號
    batch_progress = pyqtSignal(int, int, str)    # Batch 內進度 (current, total, filename)
    batch_done = pyqtSignal(int, int)              # Batch 完成 (batch_index, total_batches)
    image_done = pyqtSignal(str, object)           # 單圖完成 (image_path, WorkerOutput)
    pipeline_done = pyqtSignal(list)               # Pipeline 完成 (all_results)
    error = pyqtSignal(str)                        # 錯誤
    
    def __init__(self, 
                 name: str,
                 batches: List[Batch],
                 settings: Settings = None,
                 parent=None):
        super().__init__(parent)
        self.name = name
        self.batches = batches
        self.settings = settings
        self._stop = False
        self._all_results: List[WorkerOutput] = []
    
    def stop(self):
        """請求中止所有 Batch"""
        self._stop = True
        for batch in self.batches:
            batch.stop()
    
    def is_stopped(self) -> bool:
        """是否已中止"""
        return self._stop
    
    @property
    def all_results(self) -> List[WorkerOutput]:
        """取得所有結果"""
        return self._all_results
    
    def run(self):
        """執行 Pipeline"""
        try:
            total_batches = len(self.batches)
            
            for i, batch in enumerate(self.batches):
                if self._stop:
                    break
                
                # 連接 Batch 信號
                batch.progress.connect(
                    lambda c, t, f: self.batch_progress.emit(c, t, f)
                )
                batch.image_done.connect(
                    lambda p, o: self._on_image_done(p, o)
                )
                
                # 執行 Batch（同步）
                batch.run()
                
                # 收集結果
                self._all_results.extend(batch.results)
                
                # 發送 Batch 完成信號
                self.batch_done.emit(i + 1, total_batches)
            
            self.pipeline_done.emit(self._all_results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _on_image_done(self, image_path: str, output: WorkerOutput):
        """處理單圖完成"""
        self.image_done.emit(image_path, output)


# ============================================================
# 便捷工廠方法
# ============================================================

def create_single_image_pipeline(
    task_type: TaskType,
    image: ImageData,
    settings: Settings = None,
    options: Dict[str, Any] = None,
    name: str = None,
) -> Pipeline:
    """
    建立單圖 Pipeline
    
    這是最常見的使用情境：對當前圖片執行一個任務。
    """
    batch = Batch(
        task_type=task_type,
        images=[image],
        settings=settings,
        options=options,
    )
    return Pipeline(
        name=name or f"single_{task_type.value}",
        batches=[batch],
        settings=settings,
    )


def create_batch_pipeline(
    task_type: TaskType,
    images: List[ImageData],
    settings: Settings = None,
    options: Dict[str, Any] = None,
    name: str = None,
) -> Pipeline:
    """
    建立批量 Pipeline
    
    對所有圖片執行一個任務。
    """
    batch = Batch(
        task_type=task_type,
        images=images,
        settings=settings,
        options=options,
    )
    return Pipeline(
        name=name or f"batch_{task_type.value}",
        batches=[batch],
        settings=settings,
    )


def create_multi_task_pipeline(
    task_types: List[TaskType],
    images: List[ImageData],
    settings: Settings = None,
    options: Dict[str, Any] = None,
    name: str = None,
) -> Pipeline:
    """
    建立多任務 Pipeline
    
    對所有圖片依序執行多個任務（例如：先標籤後 LLM）。
    """
    batches = [
        Batch(
            task_type=task_type,
            images=images,
            settings=settings,
            options=options,
        )
        for task_type in task_types
    ]
    return Pipeline(
        name=name or f"multi_{'_'.join(t.value for t in task_types)}",
        batches=batches,
        settings=settings,
    )
