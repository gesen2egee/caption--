"""
圖片導航 Mixin

負責處理：
- 上一張/下一張圖片
- 跳轉到指定索引
- 刪除當前圖片
- 鍵盤左右鍵導航
- 滑鼠滾輪導航

依賴的屬性：
- self.image_files, self.current_index
- self.current_image_path
- self.load_image()
- self.index_input - 索引輸入框
"""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt
import os
from lib.utils import delete_matching_npz


class NavigationMixin:
    """圖片導航 Mixin"""
    
    def prev_image(self):
        """上一張圖片"""
        if not self.image_files:
            return
        
        count = len(self.image_files)
        # 如果過濾器啟用，使用過濾列表導航
        if self.filter_active:
             if self.current_image_path in self.filtered_image_files:
                 idx = self.filtered_image_files.index(self.current_image_path)
                 new_idx = (idx - 1) % len(self.filtered_image_files)
                 target = self.filtered_image_files[new_idx]
                 self.current_index = self.image_files.index(target)
             else:
                 # 當前圖片不在過濾結果中，重置到第一個
                 if self.filtered_image_files:
                      target = self.filtered_image_files[0]
                      self.current_index = self.image_files.index(target)
        else:
            self.current_index = (self.current_index - 1) % count
            
        self.load_image()

    def next_image(self):
        """下一張圖片"""
        if not self.image_files:
            return
        
        count = len(self.image_files)
        if self.filter_active:
             if self.current_image_path in self.filtered_image_files:
                 idx = self.filtered_image_files.index(self.current_image_path)
                 new_idx = (idx + 1) % len(self.filtered_image_files)
                 target = self.filtered_image_files[new_idx]
                 self.current_index = self.image_files.index(target)
             else:
                 if self.filtered_image_files:
                      target = self.filtered_image_files[0]
                      self.current_index = self.image_files.index(target)
        else:
            self.current_index = (self.current_index + 1) % count
            
        self.load_image()

    def jump_to_index(self):
        """跳轉到指定索引"""
        try:
            val = int(self.index_input.text())
            target_idx = val - 1
            
            if self.filter_active:
                if 0 <= target_idx < len(self.filtered_image_files):
                    target_path = self.filtered_image_files[target_idx]
                    self.current_index = self.image_files.index(target_path)
                    self.load_image()
                else:
                    self.load_image() # Reset to current
            else:
                if 0 <= target_idx < len(self.image_files):
                    self.current_index = target_idx
                    self.load_image()
                else:
                    self.load_image() # Reset to current
        except Exception:
            self.load_image()
            
    def delete_current_image(self):
        """刪除當前圖片"""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return
            
        reply = QMessageBox.question(
            self, 
            self.tr("msg_confirm_delete_title"), 
            self.tr("msg_confirm_delete_content").format(os.path.basename(self.current_image_path)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # 1. 刪除相關檔案 (.txt, .caption, .boorutag, .json)
                base, ext = os.path.splitext(self.current_image_path)
                for suffix in [".txt", ".caption", ".boorutag", ".json"]:
                    p = base + suffix
                    if os.path.exists(p):
                        os.remove(p)
                
                # 2. 刪除 .npz
                delete_matching_npz(self.current_image_path)
                
                # 3. 刪除圖片本身
                os.remove(self.current_image_path)
                
                self.statusBar().showMessage(f"Deleted: {os.path.basename(self.current_image_path)}")
                
                # 4. 更新列表並導航
                # Remove from lists
                if self.current_image_path in self.image_files:
                    self.image_files.remove(self.current_image_path)
                if self.current_image_path in self.all_image_files:
                    self.all_image_files.remove(self.current_image_path)
                if self.current_image_path in self.filtered_image_files:
                    self.filtered_image_files.remove(self.current_image_path)
                
                if not self.image_files:
                    self.current_image_path = ""
                    self.current_index = -1
                    if hasattr(self, 'image_label'):
                        self.image_label.clear()
                        self.image_label.setText(self.tr("label_no_image"))
                    return

                if self.current_index >= len(self.image_files):
                    self.current_index = len(self.image_files) - 1
                
                # Update UI
                self.load_image()
                if hasattr(self, 'update_file_list_ui'):
                    self.update_file_list_ui()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete: {e}")

    def wheelEvent(self, event):
        """滑鼠滾輪切換圖片"""
        # 檢查滑鼠是否在圖片區域
        pos = event.position().toPoint()
        widget = self.childAt(pos)
        if hasattr(self, 'image_label') and (widget is self.image_label or (widget and self.image_label.isAncestorOf(widget))):
            dy = event.angleDelta().y()
            if dy > 0:
                self.prev_image()
            elif dy < 0:
                self.next_image()
            event.accept()
            return
        super().wheelEvent(event)
