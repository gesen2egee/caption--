# -*- coding: utf-8 -*-
"""
Caption 神器 - Pipeline 模組

Pipeline → Batch → Task → Worker 架構
"""
from lib.pipeline.batch import Batch, TaskType
from lib.pipeline.pipeline import (
    Pipeline,
    create_single_image_pipeline,
    create_batch_pipeline,
    create_multi_task_pipeline,
)
from lib.pipeline.manager import (
    PipelineManager,
    create_image_data_from_path,
    create_image_data_list,
)
