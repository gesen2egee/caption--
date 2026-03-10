# -*- coding: utf-8 -*-
"""
圖片還原 Worker

從 raw_image 還原原圖。
"""
import os
import shutil
import traceback
from typing import Optional, Dict

from lib.workers.base import BaseWorker, WorkerInput, WorkerOutput
from lib.utils.sidecar import load_image_sidecar, save_image_sidecar


class ImageRestoreRawWorker(BaseWorker):
    """
    圖片還原 Worker
    
    從 raw_image 資料夾還原原圖。
    """
    
    category = "RESTORE"
    display_name = "Restore from Raw"
    description = "Restore image from raw_image backup folder"
    default_config = {}
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "image_restore_raw"
    
    def process(self, input_data: WorkerInput) -> WorkerOutput:
        """執行還原"""
        try:
            # 驗證輸入
            if not input_data.image:
                return WorkerOutput(success=False, error="缺少圖片資料")
            
            image_data = input_data.image
            image_path = image_data.path
            
            # 讀取 sidecar
            sidecar = load_image_sidecar(image_path)
            raw_rel = sidecar.get("raw_backup_path") or sidecar.get("raw_image_rel_path")
            
            if not raw_rel:
                return WorkerOutput(
                    success=True,
                    skipped=True,
                    skip_reason="找不到原圖備份紀錄",
                    image=image_data,
                )
            
            # 計算絕對路徑
            src_dir = os.path.dirname(image_path)
            raw_abs = os.path.normpath(os.path.join(src_dir, raw_rel))
            
            if not os.path.exists(raw_abs):
                return WorkerOutput(
                    success=True,
                    skipped=True,
                    skip_reason=f"原圖備份不存在: {raw_rel}",
                    image=image_data,
                )
            
            # 還原時要回到原始副檔名/檔名，避免把 PNG 原圖直接覆回 .webp 路徑。
            restore_target = os.path.join(src_dir, os.path.basename(raw_abs))

            # 複製備份回資料集目錄的原始檔名位置。
            shutil.copy2(raw_abs, restore_target)

            # 若目前顯示的是處理後的新檔名，移除它，避免資料夾留下還原後的殘留處理檔。
            removed_processed_path = None
            if os.path.normcase(os.path.normpath(restore_target)) != os.path.normcase(os.path.normpath(image_path)):
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        removed_processed_path = image_path
                    except Exception:
                        removed_processed_path = None

            # 更新 sidecar
            sidecar["masked_background"] = False
            sidecar["masked_text"] = False
            save_image_sidecar(restore_target, sidecar)

            # 更新 ImageData
            image_data.masked_background = False
            image_data.masked_text = False
            image_data.path = restore_target

            return WorkerOutput(
                success=True,
                image=image_data,
                result_data={
                    "original_path": image_path,
                    "result_path": restore_target,
                    "raw_backup_path": raw_abs,
                    "removed_processed_path": removed_processed_path,
                },
            )
            
        except Exception as e:
            traceback.print_exc()
            return WorkerOutput(success=False, error=str(e))
    
    def validate_input(self, input_data: WorkerInput) -> Optional[str]:
        if not input_data.image:
            return "缺少圖片資料"
        return None
