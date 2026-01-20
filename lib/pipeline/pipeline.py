# -*- coding: utf-8 -*-
"""
Pipeline 類別

Pipeline 負責在背景執行緒中遍歷圖片列表並執行 Task。
統一處理單張和批量，不再區分。
"""
from typing import List, Optional, Dict, Any, Type

from PyQt6.QtCore import QThread, pyqtSignal

from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta
from lib.pipeline.context import TaskContext, TaskResult
from lib.pipeline.tasks.base_task import BaseTask


class Pipeline(QThread):
    """
    Pipeline 類別
    
    在背景執行緒中遍歷圖片列表，對每張圖片執行指定的 Task。
    不區分單張/批量，統一處理。
    """
    
    # 信號
    progress = pyqtSignal(int, int, str)       # 進度 (current, total, filename)
    image_done = pyqtSignal(str, object)       # 單圖完成 (image_path, TaskResult)
    pipeline_done = pyqtSignal(list)           # Pipeline 完成 (all_results)
    error = pyqtSignal(str)                    # 錯誤
    
    def __init__(self, 
                 task: BaseTask,
                 images: List[ImageData],
                 settings: Settings,
                 prompt: Optional[Prompt] = None,
                 folder: Optional[FolderMeta] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 parent=None):
        super().__init__(parent)
        self.task = task
        self.images = images
        self.settings = settings
        self.prompt = prompt
        self.folder = folder
        self.extra = extra or {}
        self._stop = False
        self._all_results: List[TaskResult] = []
    
    @property
    def name(self) -> str:
        """Pipeline 名稱（來自 Task）"""
        return self.task.name
    
    def stop(self):
        """請求中止"""
        self._stop = True
    
    def is_stopped(self) -> bool:
        """是否已中止"""
        return self._stop
    
    @property
    def all_results(self) -> List[TaskResult]:
        """取得所有結果"""
        return self._all_results
    
    def run(self):
        """執行 Pipeline"""
        try:
            total = len(self.images)
            
            for i, image in enumerate(self.images):
                if self._stop:
                    break
                
                # 發送進度
                self.progress.emit(i + 1, total, image.filename)
                
                # 建立上下文
                context = TaskContext(
                    image=image,
                    settings=self.settings,
                    prompt=self.prompt,
                    folder=self.folder,
                    extra=self.extra,
                )
                
                # 執行 Task
                result = self.task.execute(context)
                self._all_results.append(result)
                
                # 發送單圖完成信號
                self.image_done.emit(image.path, result)
            
            self.pipeline_done.emit(self._all_results)
            
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# 便捷工廠方法
# ============================================================

def create_pipeline(
    task: BaseTask,
    images: List[ImageData],
    settings: Settings,
    prompt: Optional[Prompt] = None,
    folder: Optional[FolderMeta] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    建立 Pipeline
    
    統一的建立方法，不區分單張/批量。
    """
    return Pipeline(
        task=task,
        images=images,
        settings=settings,
        prompt=prompt,
        folder=folder,
        extra=extra,
    )
