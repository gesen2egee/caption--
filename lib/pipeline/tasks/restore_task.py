# -*- coding: utf-8 -*-
"""
RestoreTask - 還原任務

從 raw_image 備份還原原圖。
"""
import traceback
from typing import Tuple

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.runtime.errors import build_runtime_error_info
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar
from lib.workers import invoke_worker


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
        if context.extra.get("force_execution", False):
            return False, ""
            
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
            
            worker_output = invoke_worker(
                "RESTORE",
                "image_restore_raw",
                config={},
                worker_input=context.to_worker_input(),
                settings=context.settings,
            )
            
            if not worker_output.success:
                return TaskResult(
                    success=False,
                    error=worker_output.error,
                    error_info=worker_output.error_info,
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
            result_path = ""
            if worker_output.result_data:
                result_path = str(worker_output.result_data.get("result_path", "") or "")
            if result_path:
                context.image.path = result_path
            
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
                error_info=build_runtime_error_info(e, source="pipeline.task.restore"),
                image=context.image,
            )
