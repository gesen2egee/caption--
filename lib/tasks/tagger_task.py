# -*- coding: utf-8 -*-
"""
Tagger 任務

單圖和批量標籤識別任務。
"""
from typing import List

from lib.tasks.base import SingleImageTask, BatchTask
from lib.workers.base import BaseWorker
from lib.workers.tagger_imgutils_tagging_local import TaggerImgutilsTaggingLocalWorker
from lib.core.dataclasses import ImageData, Settings


class TaggerTask(SingleImageTask):
    """
    單圖標籤識別任務
    """
    
    @property
    def name(self) -> str:
        return "tagger_single"
    
    def get_worker(self) -> BaseWorker:
        config = {}
        if self.settings:
            config = {
                "model_name": self.settings.tagger_model,
                "general_threshold": self.settings.general_threshold,
                "general_mcut_enabled": self.settings.general_mcut_enabled,
                "character_threshold": self.settings.character_threshold,
                "character_mcut_enabled": self.settings.character_mcut_enabled,
                "drop_overlap": self.settings.drop_overlap,
            }
        return TaggerImgutilsTaggingLocalWorker(config)


class BatchTaggerTask(BatchTask):
    """
    批量標籤識別任務
    """
    
    @property
    def name(self) -> str:
        return "tagger_batch"
    
    def get_worker(self) -> BaseWorker:
        config = {}
        if self.settings:
            config = {
                "model_name": self.settings.tagger_model,
                "general_threshold": self.settings.general_threshold,
                "general_mcut_enabled": self.settings.general_mcut_enabled,
                "character_threshold": self.settings.character_threshold,
                "character_mcut_enabled": self.settings.character_mcut_enabled,
                "drop_overlap": self.settings.drop_overlap,
            }
        return TaggerImgutilsTaggingLocalWorker(config)
