# -*- coding: utf-8 -*-
"""
TaggerTask - 自動標籤任務

使用 WD14 Tagger 對圖片進行標籤識別。
"""
import traceback
from typing import Tuple

from lib.pipeline.tasks.base_task import BaseTask
from lib.pipeline.context import TaskContext, TaskResult
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar


class TaggerTask(BaseTask):
    """
    自動標籤任務
    
    使用 imgutils.tagging 的 WD14 模型進行圖片標籤識別。
    """
    
    @property
    def name(self) -> str:
        return "tagger"
    
    def should_skip(self, context: TaskContext) -> Tuple[bool, str]:
        """Tagger 預設不跳過任何圖片"""
        return False, ""
    
    def execute(self, context: TaskContext) -> TaskResult:
        """執行標籤識別"""
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
            from lib.workers.tagger_imgutils_tagging_local import TaggerImgutilsTaggingLocalWorker
            
            config = {}
            if context.settings:
                config = {
                    "model_name": context.settings.tagger_model,
                    "general_threshold": context.settings.general_threshold,
                    "general_mcut_enabled": context.settings.general_mcut_enabled,
                    "character_threshold": context.settings.character_threshold,
                    "character_mcut_enabled": context.settings.character_mcut_enabled,
                    "drop_overlap": context.settings.drop_overlap,
                }
            
            worker = TaggerImgutilsTaggingLocalWorker(config)
            worker_output = worker.process(context.to_worker_input())
            
            if not worker_output.success:
                return TaskResult(
                    success=False,
                    error=worker_output.error,
                    image=context.image,
                )
            
            # 3. 後處理：儲存結果到 sidecar
            if worker_output.result_text:
                sidecar = load_image_sidecar(context.image.path)
                sidecar["tagger_tags"] = worker_output.result_text
                if worker_output.result_data:
                    sidecar["tagger_rating"] = worker_output.result_data.get("rating")
                    sidecar["tagger_features"] = worker_output.result_data.get("features")
                    sidecar["tagger_chars"] = worker_output.result_data.get("chars")
                save_image_sidecar(context.image.path, sidecar)
                
                # 更新 ImageData
                context.image.tagger_tags = worker_output.result_text
                if worker_output.result_data:
                    context.image.tagger_rating = worker_output.result_data.get("rating")
                    context.image.tagger_features = worker_output.result_data.get("features")
                    context.image.tagger_chars = worker_output.result_data.get("chars")
            
            return TaskResult(
                success=True,
                image=context.image,
                result_text=worker_output.result_text,
                result_data=worker_output.result_data,
            )
            
        except Exception as e:
            traceback.print_exc()
            return TaskResult(
                success=False,
                error=str(e),
                image=context.image,
            )
