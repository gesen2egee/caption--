# -*- coding: utf-8 -*-
"""
Pipeline Manager

管理 Pipeline 的執行，提供 UI 整合介面。
使用統一的 Task API，不區分單張/批量。
"""
from typing import Optional, List, Dict, Any, Type

from PyQt6.QtCore import QObject, pyqtSignal

from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta
from lib.pipeline.pipeline import Pipeline, create_pipeline
from lib.pipeline.context import TaskResult
from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.tasks import TaggerTask, LLMTask, UnmaskTask, MaskTextTask, RestoreTask
from lib.utils.sidecar import load_image_sidecar


class PipelineManager(QObject):
    """
    Pipeline 管理器
    
    提供給 UI 使用的高階介面，管理 Pipeline 執行。
    使用統一的 Task API，UI 決定圖片列表（單張或多張）。
    """
    
    # 信號（轉發給 UI）
    progress = pyqtSignal(int, int, str)          # 進度 (current, total, filename)
    image_done = pyqtSignal(str, object)          # 單圖完成 (image_path, TaskResult)
    pipeline_done = pyqtSignal(str, list)         # Pipeline 完成 (name, results)
    error = pyqtSignal(str)                        # 錯誤
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pipeline: Optional[Pipeline] = None
        self._settings: Optional[Settings] = None
        self._prompt: Optional[Prompt] = None
        self._folder: Optional[FolderMeta] = None
    
    def set_settings(self, settings: Settings):
        """設定當前設定"""
        self._settings = settings
    
    def set_prompt(self, prompt: Prompt):
        """設定 Pipeline 指令"""
        self._prompt = prompt
    
    def set_folder(self, folder: FolderMeta):
        """設定資料夾 Meta"""
        self._folder = folder
    
    def is_running(self) -> bool:
        """是否有 Pipeline 正在執行"""
        return self._current_pipeline is not None and self._current_pipeline.isRunning()
    
    def stop(self):
        """中止當前 Pipeline"""
        if self._current_pipeline:
            self._current_pipeline.stop()
    
    # ============================================================
    # 統一 Task 執行介面
    # ============================================================
    
    def run_task(self, task: BaseTask, images: List[ImageData], 
                 extra: Optional[Dict[str, Any]] = None):
        """
        執行 Task
        
        統一的執行入口，UI 決定傳入的圖片列表（單張或多張）。
        
        Args:
            task: Task 實例
            images: 圖片列表（單張時為 [image]）
            extra: 額外參數
        """
        if self.is_running():
            self.error.emit("已有 Pipeline 正在執行")
            return
        
        pipeline = create_pipeline(
            task=task,
            images=images,
            settings=self._settings,
            prompt=self._prompt,
            folder=self._folder,
            extra=extra,
        )
        self._run_pipeline(pipeline)
    
    # ============================================================
    # 便捷方法（內部建立 Task 實例）
    # ============================================================
    
    def run_tagger(self, images: List[ImageData]):
        """執行標籤"""
        self.run_task(TaggerTask(), images)
    
    def run_llm(self, images: List[ImageData], user_prompt: str = None, system_prompt: str = None):
        """執行 LLM"""
        extra = {}
        if user_prompt:
            extra["user_prompt"] = user_prompt
        if system_prompt:
            extra["system_prompt"] = system_prompt
        self.run_task(LLMTask(), images, extra=extra)
    
    def run_unmask(self, images: List[ImageData]):
        """執行去背"""
        self.run_task(UnmaskTask(), images)
    
    def run_mask_text(self, images: List[ImageData]):
        """執行去文字"""
        self.run_task(MaskTextTask(), images)
    
    def run_restore(self, images: List[ImageData]):
        """執行還原"""
        self.run_task(RestoreTask(), images)
    
    # ============================================================
    # 內部方法
    # ============================================================
    
    def _run_pipeline(self, pipeline: Pipeline):
        """執行 Pipeline"""
        self._current_pipeline = pipeline
        
        # 連接信號
        pipeline.progress.connect(self.progress.emit)
        pipeline.image_done.connect(self.image_done.emit)
        pipeline.pipeline_done.connect(
            lambda results: self._on_done(pipeline.name, results)
        )
        pipeline.error.connect(self.error.emit)
        
        # 啟動
        pipeline.start()
    
    def _on_done(self, name: str, results: List[TaskResult]):
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
