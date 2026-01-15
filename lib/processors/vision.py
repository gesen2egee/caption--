
import os
import shutil
from typing import Tuple, Any
from .base import BaseProcessor
from ..data import ImageContext, AppSettings
from ..services.vision import get_remover_model, process_background_removal, detect_text_boxes
from ..services.common import unload_all_models
from ..utils import (
    load_image_sidecar, save_image_sidecar, image_sidecar_json_path,
    backup_raw_image, delete_matching_npz
)

class UnmaskProcessor(BaseProcessor):
    def __init__(self, settings: AppSettings):
        super().__init__(settings)
        self.remover = None

    def prepare(self):
        """載入透明背景移除模型 (Heavy Load)"""
        try:
            self.remover = get_remover_model(self.settings)
        except ImportError:
            raise RuntimeError("transparent_background library not found.")

    def process(self, ctx: ImageContext) -> Tuple[bool, Any]:
        """
        執行去背。
        回傳: (True, new_path) if successful, (False, None) if skipped/failed.
        """
        if not self.remover:
            raise RuntimeError("Remover not initialized.")

        # 1. 檢查是否跳過 (Skip Logic)
        # 1a. 是否已處理過
        if bool(self.settings.get("mask_batch_skip_once_processed", True)):
            if ctx.sidecar.get("masked_background", False):
                return False, None

        # 1b. 檢查 Scenery Tag
        if bool(self.settings.get("mask_batch_skip_if_scenery_tag", True)):
            tags = ctx.sidecar.get("tagger_tags", "").lower()
            if "indoors" in tags or "outdoors" in tags:
                return False, None

        # 1c. 檢查 Background Tag
        if bool(self.settings.get("mask_batch_only_if_has_background_tag", False)):
            tags = ctx.sidecar.get("tagger_tags", "").lower()
            # 簡單檢查單字
            if "background" not in tags:
                return False, None

        # 2. 備份原圖 (Backup)
        backup_success = backup_raw_image(ctx.path)
        
        # 3. 處理圖片 (Processing)
        try:
            # 為了避免鎖定，先讀入記憶體
            img = ctx.get_image(mode=None) # Keep original mode
            
            # 呼叫 Service 進行數學運算
            result_img = process_background_removal(img, self.settings, self.remover)
            
            # 4. 存檔 (Save)
            # 決定副檔名 (WebP 優先)
            fmt = str(self.settings.get("mask_default_format", "webp")).lower().strip(".")
            if fmt not in ("webp", "png"):
                fmt = "webp"
                
            base_no_ext = os.path.splitext(ctx.path)[0]
            new_path = base_no_ext + f".{fmt}"
            
            # Save
            if fmt == "png":
                result_img.save(new_path, "PNG")
            else:
                result_img.save(new_path, "WEBP", quality=100)
                
            # 5. 清理舊檔 (Cleanup Old File)
            # 如果副檔名改變了 (例如 jpg -> webp)，且新舊路徑不同
            new_abs = os.path.abspath(new_path)
            old_abs = os.path.abspath(ctx.path)
            
            if new_abs != old_abs:
                try:
                    # 刪除舊圖 (因為已備份)
                    os.remove(old_abs)
                    
                    # 遷移 Sidecar
                    old_json = image_sidecar_json_path(old_abs)
                    new_json = image_sidecar_json_path(new_abs)
                    if os.path.exists(old_json) and old_json != new_json:
                        shutil.move(old_json, new_json)
                    
                    # 刪除 NPZ
                    if self.settings.get("mask_delete_npz_on_move", True):
                        delete_matching_npz(old_abs)
                        
                except Exception as e:
                    print(f"Error cleaning up old file {old_abs}: {e}")

            # 6. 更新 Sidecar 標記
            # 注意: 此時 ctx.path 仍指向舊路徑，如果你要reload sidecar會有問題
            # 但我們手動讀取新路徑的 sidecar
            _sc = load_image_sidecar(new_path)
            _sc["masked_background"] = True
            save_image_sidecar(new_path, _sc)
            
            # 更新 Context 物件狀態 (雖然 Worker 迴圈跑完了，但為了正確性)
            ctx.path = new_path 
            ctx.reload_sidecar()
            
            return True, new_path

        except Exception as e:
            print(f"Failed to unmask {ctx.filename}: {e}")
            return False, None

    def cleanup(self):
        unload_all_models()
        self.remover = None
