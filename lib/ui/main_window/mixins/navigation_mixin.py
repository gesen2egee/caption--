"""
圖片導航 Mixin

負責處理：
- 上一張/下一張圖片
- 刪除當前圖片
- 圖片索引管理

依賴的屬性：
- self.image_files: list - 圖片列表
- self.current_index: int - 當前索引
- self.current_image_path: str - 當前圖片路徑
"""

from PyQt6.QtWidgets import QMessageBox
import os
from lib.utils import delete_matching_npz


class NavigationMixin:
    """圖片導航 Mixin"""
    
    def prev_image(self):
        """上一張圖片"""
        if not self.image_files:
            return
        new_idx = (self.current_index - 1) % len(self.image_files)
        self.load_image(new_idx)

    def next_image(self):
        """下一張圖片"""
        if not self.image_files:
            return
        new_idx = (self.current_index + 1) % len(self.image_files)
        self.load_image(new_idx)

    def delete_current_image(self):
        """刪除當前圖片"""
        if not self.current_image_path:
            return
        
        reply = QMessageBox.question(
            self, "確認刪除",
            f"確定要刪除這張圖片嗎？\n{os.path.basename(self.current_image_path)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # 刪除圖片和相關檔案
            if os.path.exists(self.current_image_path):
                os.remove(self.current_image_path)
            
            # 刪除 txt
            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                os.remove(txt_path)
            
            # 刪除 json
            json_path = os.path.splitext(self.current_image_path)[0] + ".json"
            if os.path.exists(json_path):
                os.remove(json_path)
            
            # 刪除 npz
            delete_matching_npz(self.current_image_path)
            
            # 從列表中移除
            if self.current_image_path in self.image_files:
                self.image_files.remove(self.current_image_path)
            if hasattr(self, 'all_image_files') and self.current_image_path in self.all_image_files:
                self.all_image_files.remove(self.current_image_path)
            
            # 載入下一張
            if self.image_files:
                new_idx = min(self.current_index, len(self.image_files) - 1)
                self.load_image(new_idx)
            else:
                self.current_index = -1
                self.current_image_path = ""
                
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"刪除失敗: {e}")
