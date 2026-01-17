"""
Batch Base Mixin

負責處理：
- 通用批量處理完成回調
- 通用批量處理錯誤回調
- 圖片路徑替換輔助方法

依賴的屬性：
- self.image_files, self.all_image_files: list
- self.current_image_path: str
- self.hide_progress()
- self.load_image()
- self.btn_* (各種按鈕狀態)
"""

from PyQt6.QtWidgets import QMessageBox
import os
from lib.services.common import unload_all_models


class BatchBaseMixin:
    """批量處理基礎 Mixin"""
    
    def on_batch_done(self, msg="Batch Process Completed"):
        """批量處理完成回調"""
        self.hide_progress()
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(False)
            self.btn_cancel_batch.setEnabled(False)
        
        # 批次完成後刷新當前圖片顯示
        if hasattr(self, 'load_image') and hasattr(self, 'current_image_path') and self.current_image_path:
            self.load_image()
        
        QMessageBox.information(self, "Batch", msg)
        
        # 非同步執行模型卸載，避免同步卡頓
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, unload_all_models)

    def on_batch_error(self, title, err):
        """批量處理錯誤回調"""
        self.recover_ui_state()
        self.hide_progress()
        self.statusBar().showMessage(f"Batch Error ({title}): {err}", 8000)

    def recover_ui_state(self):
        """恢復 UI 按鈕狀態"""
        if hasattr(self, 'btn_batch_tagger'): self.btn_batch_tagger.setEnabled(True)
        if hasattr(self, 'btn_batch_tagger_to_txt'): self.btn_batch_tagger_to_txt.setEnabled(True)
        if hasattr(self, 'btn_auto_tag'): self.btn_auto_tag.setEnabled(True)
        if hasattr(self, 'btn_batch_llm'): self.btn_batch_llm.setEnabled(True)
        if hasattr(self, 'btn_batch_llm_to_txt'): self.btn_batch_llm_to_txt.setEnabled(True)
        if hasattr(self, 'btn_run_llm'): self.btn_run_llm.setEnabled(True)
        if hasattr(self, '_is_batch_to_txt'): self._is_batch_to_txt = False

    def _replace_image_path_in_list(self, old_path, new_path):
        """
        在圖片列表中替換路徑 (當副檔名改變時使用)
        """
        if not old_path or not new_path:
            return
            
        old_abs = os.path.abspath(old_path)
        new_abs = os.path.abspath(new_path)
        
        # 1. 更新當前圖片路徑 (最重要，防止後續 IO 失敗)
        is_current = False
        if self.current_image_path and os.path.abspath(self.current_image_path) == old_abs:
            self.current_image_path = new_path
            is_current = True
        
        # 2. 更新 image_files 列表
        found_in_list = False
        for i, p in enumerate(self.image_files):
            if os.path.abspath(p) == old_abs:
                self.image_files[i] = new_path
                if is_current:
                    self.current_index = i
                found_in_list = True
        
        # 3. 更新 all_image_files
        if hasattr(self, 'all_image_files'):
            for i, p in enumerate(self.all_image_files):
                if os.path.abspath(p) == old_abs:
                    self.all_image_files[i] = new_path

        # 4. 如果是當前圖片且更換了副檔名，立即通知所有相關 mixin
        if is_current and old_abs != new_abs:
             print(f"[Core] Image path synced: {old_path} -> {new_path}")
