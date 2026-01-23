# -*- coding: utf-8 -*-
"""
BaseTask - Integrated Task & Pipeline

Merges Task logic with the execution pipeline.
Each Task is now a QThread that manages its own execution loop over a list of images.
"""
import time
import traceback
from typing import List, Optional, Dict, Any, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from lib.pipeline.context import TaskContext, TaskResult
from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta


class BaseTask(QThread):
    """
    BaseTask (Threaded)
    
    Contains:
    - Thread execution loop (formerly Pipeline)
    - Signal definitions
    - Abstract execute method for workers
    """
    
    # Signals
    progress = pyqtSignal(int, int, str, float)       # current, total, filename, speed
    image_done = pyqtSignal(str, object)              # image_path, TaskResult
    batch_done = pyqtSignal(list)                     # all_results
    error = pyqtSignal(str)                           # error message
    
    def __init__(self, 
                 images: List[ImageData],
                 settings: Settings,
                 prompt: Optional[Prompt] = None,
                 folder: Optional[FolderMeta] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 parent=None):
        super().__init__(parent)
        self.images = images
        self.settings = settings
        self.prompt = prompt
        self.folder = folder
        self.extra = extra or {}
        
        self._stop_event = False
        self._all_results: List[TaskResult] = []

    @property
    def name(self) -> str:
        """Task name for logs"""
        raise NotImplementedError
    
    def stop(self):
        """Request stop"""
        self._stop_event = True
        
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """
        Determine if the image should be skipped.
        Returns (should_skip, reason)
        """
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """
        Process a single image. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def run(self):
        """
        Main execution loop (Runs in background thread)
        """
        try:
            total = len(self.images)
            start_time = time.time()
            self._all_results = []
            
            for i, image in enumerate(self.images):
                if self._stop_event:
                    break
                
                # Speed Calculation
                elapsed = time.time() - start_time
                speed = (i) / elapsed if elapsed > 0 and i > 0 else 0.0
                
                # Emit progress BEFORE processing (or allow UI to show "Processing...")
                self.progress.emit(i + 1, total, image.filename, speed)
                
                context = TaskContext(
                    image=image,
                    settings=self.settings,
                    prompt=self.prompt,
                    folder=self.folder,
                    extra=self.extra,
                )
                
                # Execute Logic
                # Note: 'execute' methods in subclasses often handle their own 'should_skip' checks 
                # and return a skipped TaskResult. We trust the subclass to handle this.
                result = self.execute(context)
                self._all_results.append(result)
                
                self.image_done.emit(image.path, result)
            
            self.batch_done.emit(self._all_results)
            
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
