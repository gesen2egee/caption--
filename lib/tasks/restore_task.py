# -*- coding: utf-8 -*-
"""
還原任務

單圖和批量還原原圖任務。
"""
from typing import List

from lib.tasks.base import SingleImageTask, BatchTask
from lib.workers.base import BaseWorker
from lib.workers.image_restore_raw import ImageRestoreRawWorker
from lib.core.dataclasses import ImageData, Settings


class RestoreTask(SingleImageTask):
    """
    單圖還原任務
    """
    
    @property
    def name(self) -> str:
        return "restore_single"
    
    def get_worker(self) -> BaseWorker:
        return ImageRestoreRawWorker()


class BatchRestoreTask(BatchTask):
    """
    批量還原任務
    """
    
    @property
    def name(self) -> str:
        return "restore_batch"
    
    def get_worker(self) -> BaseWorker:
        return ImageRestoreRawWorker()
