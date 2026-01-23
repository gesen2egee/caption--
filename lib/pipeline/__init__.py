# -*- coding: utf-8 -*-
"""
Caption 神器 - Pipeline 模組

Pipeline → Task → Worker 架構
統一的任務執行框架，不區分單張/批量。
"""
from lib.pipeline.context import TaskContext, TaskResult
from lib.utils.file_ops import create_image_data_from_path, create_image_data_list

from lib.pipeline.tasks import (
    BaseTask,
    TaggerTask,
    LLMTask,
    UnmaskTask,
    MaskTextTask,
    RestoreTask,
)

__all__ = [
    # Context
    "TaskContext",
    "TaskResult",
    # Functions
    "create_image_data_from_path",
    "create_image_data_list",
    # Tasks
    "BaseTask",
    "TaggerTask",
    "LLMTask",
    "UnmaskTask",
    "MaskTextTask",
    "RestoreTask",
]
