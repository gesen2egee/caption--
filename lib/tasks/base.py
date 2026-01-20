# -*- coding: utf-8 -*-
"""
Task 基礎類別

Task 負責業務邏輯：判斷、中止控制、更新 UI、銜接 Worker。
Task 繼承 QThread，發送信號給 UI。
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal

from lib.core.dataclasses import ImageData, Settings
from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput


class BaseTask(QThread):
    """
    Task 抽象基礎類別
    
    Task 負責：
    1. 簡單判斷（是否處理、是否跳過）
    2. 中止控制
    3. 更新 UI（透過 signals）
    4. 銜接 Worker
    """
    
    # 通用信號
    progress = pyqtSignal(int, int, str)      # 進度 (current, total, message)
    per_image = pyqtSignal(str, str)          # 每張圖片完成 (image_path, result)
    done = pyqtSignal()                        # 任務完成
    error = pyqtSignal(str)                    # 錯誤
    
    def __init__(self, settings: Settings = None):
        super().__init__()
        self.settings = settings
        self._stop = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Task 名稱"""
        pass
    
    def stop(self):
        """請求中止任務"""
        self._stop = True
    
    def is_stopped(self) -> bool:
        """檢查是否已請求中止"""
        return self._stop
    
    @abstractmethod
    def run(self):
        """執行任務（由 QThread 調用）"""
        pass


class SingleImageTask(BaseTask):
    """
    單張圖片任務基礎類別
    """
    
    # 單圖完成信號
    finished_single = pyqtSignal(str)         # 完成 (result)
    
    def __init__(self, image: ImageData, settings: Settings = None):
        super().__init__(settings)
        self.image = image
    
    @abstractmethod
    def get_worker(self) -> BaseWorker:
        """取得要使用的 Worker"""
        pass
    
    def should_process(self) -> bool:
        """判斷是否應該處理（可覆寫）"""
        return True
    
    def prepare_input(self) -> WorkerInput:
        """準備 Worker 輸入（可覆寫）"""
        return WorkerInput(
            image=self.image,
            settings=self.settings,
        )
    
    def handle_output(self, output: WorkerOutput):
        """處理 Worker 輸出（可覆寫）"""
        if output.success:
            if output.image:
                self.image = output.image
            self.finished_single.emit(output.result_text or "")
        else:
            self.error.emit(output.error or "未知錯誤")
    
    def run(self):
        """執行任務"""
        try:
            if not self.should_process():
                self.finished_single.emit("")
                return
            
            worker = self.get_worker()
            input_data = self.prepare_input()
            output = worker.process(input_data)
            self.handle_output(output)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()


class BatchTask(BaseTask):
    """
    批量任務基礎類別
    """
    
    def __init__(self, images: List[ImageData], settings: Settings = None):
        super().__init__(settings)
        self.images = images
        self.results: List[WorkerOutput] = []
    
    @abstractmethod
    def get_worker(self) -> BaseWorker:
        """取得要使用的 Worker"""
        pass
    
    def should_process_image(self, image: ImageData) -> bool:
        """判斷單張圖片是否應該處理（可覆寫）"""
        return True
    
    def prepare_input(self, image: ImageData) -> WorkerInput:
        """準備單張圖片的 Worker 輸入（可覆寫）"""
        return WorkerInput(
            image=image,
            settings=self.settings,
        )
    
    def handle_output(self, image: ImageData, output: WorkerOutput):
        """處理單張圖片的 Worker 輸出（可覆寫）"""
        self.results.append(output)
        if output.success and not output.skipped:
            self.per_image.emit(image.path, output.result_text or "")
        elif output.skipped:
            self.per_image.emit(image.path, f"[跳過] {output.skip_reason or ''}")
    
    def run(self):
        """執行批量任務"""
        try:
            worker = self.get_worker()
            total = len(self.images)
            
            for i, image in enumerate(self.images):
                if self.is_stopped():
                    break
                
                self.progress.emit(i + 1, total, image.filename)
                
                if not self.should_process_image(image):
                    self.per_image.emit(image.path, "[跳過]")
                    continue
                
                input_data = self.prepare_input(image)
                output = worker.process(input_data)
                self.handle_output(image, output)
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()
