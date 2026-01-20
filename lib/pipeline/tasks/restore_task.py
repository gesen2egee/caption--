# -*- coding: utf-8 -*-
"""
RestoreTask - 還原任務

從 raw_image 備份還原原圖。
"""
import traceback
from typing import Tuple

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar


class RestoreTask(BaseTask):
    """
    還原任務
    
    從 raw_image 資料夾還原原圖。
    """
    
    @property
    def name(self) -> str:
        return "restore"
    
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """檢查是否有備份"""
        sidecar = load_image_sidecar(context.image.path)
        raw_rel = sidecar.get("raw_backup_path") or sidecar.get("raw_image_rel_path")
        if not raw_rel:
            return True, "找不到原圖備份紀錄"
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """執行還原"""
        try:
            # 1. Skip 判斷
            should_skip, skip_reason = self.should_skip(context)
            if should_skip:
                return TaskResult(
                    success=True,
                    skipped=True,
                    skip_reason=skip_reason,
                    image=context.image,
                )
            
            # 2. 建立並呼叫 Worker
            from lib.workers.image_restore_raw import ImageRestoreRawWorker
            
            worker = ImageRestoreRawWorker()
            worker_output = worker.process(context.to_worker_input())
            
            if not worker_output.success:
                return TaskResult(
                    success=False,
                    error=worker_output.error,
                    image=context.image,
                )
            
            if worker_output.skipped:
                return TaskResult(
                    success=True,
                    skipped=True,
                    skip_reason=worker_output.skip_reason,
                    image=context.image,
                )
            
            # 3. 更新 ImageData
            context.image.masked_background = False
            context.image.masked_text = False
            
            return TaskResult(
                success=True,
                image=context.image,
                result_data=worker_output.result_data,
            )
            
        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                image=context.image,
            )
