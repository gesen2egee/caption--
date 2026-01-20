# -*- coding: utf-8 -*-
"""
Pipeline Manager

管理 Pipeline 的執行，提供 UI 整合介面。
"""
from typing import Optional, Callable, List, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal

from lib.core.dataclasses import ImageData, Settings
from lib.pipeline.batch import Batch, TaskType
from lib.pipeline.pipeline import (
    Pipeline,
    create_single_image_pipeline,
    create_batch_pipeline,
    create_multi_task_pipeline,
)
from lib.workers.base import WorkerOutput
from lib.utils.sidecar import load_image_sidecar


class PipelineManager(QObject):
    """
    Pipeline 管理器
    
    提供給 UI 使用的高階介面，管理 Pipeline 執行。
    """
    
    # 信號（轉發給 UI）
    progress = pyqtSignal(int, int, str)          # 進度
    image_done = pyqtSignal(str, object)          # 單圖完成
    pipeline_done = pyqtSignal(str, list)         # Pipeline 完成 (name, results)
    error = pyqtSignal(str)                        # 錯誤
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pipeline: Optional[Pipeline] = None
        self._settings: Optional[Settings] = None
    
    def set_settings(self, settings: Settings):
        """設定當前設定"""
        self._settings = settings
    
    def is_running(self) -> bool:
        """是否有 Pipeline 正在執行"""
        return self._current_pipeline is not None and self._current_pipeline.isRunning()
    
    def stop(self):
        """中止當前 Pipeline"""
        if self._current_pipeline:
            self._current_pipeline.stop()
    
    # ============================================================
    # 便捷方法 - 單圖操作
    # ============================================================
    
    def run_tagger(self, image: ImageData):
        """執行單圖標籤"""
        self._run_single(TaskType.TAGGER, image)
    
    def run_llm(self, image: ImageData, prompt_mode: str = "default", user_prompt: str = None, system_prompt: str = None):
        """執行單圖 LLM"""
        options = {"prompt_mode": prompt_mode}
        if user_prompt: options["user_prompt"] = user_prompt
        if system_prompt: options["system_prompt"] = system_prompt
        self._run_single(TaskType.LLM, image, options=options)
    
    def run_unmask(self, image: ImageData):
        """執行單圖去背"""
        self._run_single(TaskType.UNMASK, image)
    
    def run_mask_text(self, image: ImageData):
        """執行單圖去文字"""
        self._run_single(TaskType.MASK_TEXT, image)
    
    def run_restore(self, image: ImageData):
        """執行單圖還原"""
        self._run_single(TaskType.RESTORE, image)
    
    # ============================================================
    # 便捷方法 - 批量操作
    # ============================================================
    
    def run_batch_tagger(self, images: List[ImageData]):
        """執行批量標籤"""
        self._run_batch(TaskType.TAGGER, images)
    
    def run_batch_llm(self, images: List[ImageData], prompt_mode: str = "default", user_prompt: str = None, system_prompt: str = None):
        """執行批量 LLM"""
        options = {"prompt_mode": prompt_mode}
        if user_prompt: options["user_prompt"] = user_prompt
        if system_prompt: options["system_prompt"] = system_prompt
        self._run_batch(TaskType.LLM, images, options=options)
    
    def run_batch_unmask(self, images: List[ImageData]):
        """執行批量去背"""
        self._run_batch(TaskType.UNMASK, images)
    
    def run_batch_mask_text(self, images: List[ImageData]):
        """執行批量去文字"""
        self._run_batch(TaskType.MASK_TEXT, images)
    
    def run_batch_restore(self, images: List[ImageData]):
        """執行批量還原"""
        self._run_batch(TaskType.RESTORE, images)
    
    # ============================================================
    # 便捷方法 - 複合操作
    # ============================================================
    
    def run_tagger_then_llm(self, images: List[ImageData]):
        """先標籤後 LLM"""
        self._run_multi([TaskType.TAGGER, TaskType.LLM], images)
    
    # ============================================================
    # 內部方法
    # ============================================================
    
    def _run_single(self, task_type: TaskType, image: ImageData, 
                    options: Dict[str, Any] = None):
        """執行單圖 Pipeline"""
        pipeline = create_single_image_pipeline(
            task_type=task_type,
            image=image,
            settings=self._settings,
            options=options,
        )
        self._run_pipeline(pipeline)
    
    def _run_batch(self, task_type: TaskType, images: List[ImageData],
                   options: Dict[str, Any] = None):
        """執行批量 Pipeline"""
        pipeline = create_batch_pipeline(
            task_type=task_type,
            images=images,
            settings=self._settings,
            options=options,
        )
        self._run_pipeline(pipeline)
    
    def _run_multi(self, task_types: List[TaskType], images: List[ImageData],
                   options: Dict[str, Any] = None):
        """執行多任務 Pipeline"""
        pipeline = create_multi_task_pipeline(
            task_types=task_types,
            images=images,
            settings=self._settings,
            options=options,
        )
        self._run_pipeline(pipeline)
    
    def _run_pipeline(self, pipeline: Pipeline):
        """執行 Pipeline"""
        if self.is_running():
            self.error.emit("已有 Pipeline 正在執行")
            return
        
        self._current_pipeline = pipeline
        
        # 連接信號
        pipeline.batch_progress.connect(self.progress.emit)
        pipeline.image_done.connect(self.image_done.emit)
        pipeline.pipeline_done.connect(
            lambda results: self._on_done(pipeline.name, results)
        )
        pipeline.error.connect(self.error.emit)
        
        # 啟動
        pipeline.start()
    
    def _on_done(self, name: str, results: List[WorkerOutput]):
        """Pipeline 完成"""
        self._current_pipeline = None
        self.pipeline_done.emit(name, results)


# ============================================================
# 輔助函數
# ============================================================

def create_image_data_from_path(image_path: str) -> ImageData:
    """從檔案路徑建立 ImageData"""
    sidecar = load_image_sidecar(image_path)
    return ImageData(
        path=image_path,
        tagger_tags=sidecar.get("tagger_tags"),
        nl_pages=sidecar.get("nl_pages", []),
        masked_background=sidecar.get("masked_background", False),
        masked_text=sidecar.get("masked_text", False),
        raw_image_rel_path=sidecar.get("raw_image_rel_path") or sidecar.get("raw_backup_path"),
    )


def create_image_data_list(image_paths: List[str]) -> List[ImageData]:
    """從檔案路徑列表建立 ImageData 列表"""
    return [create_image_data_from_path(p) for p in image_paths]
