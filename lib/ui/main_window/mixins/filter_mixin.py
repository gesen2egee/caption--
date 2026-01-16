"""
篩選功能 Mixin

負責處理：
- Danbooru 標籤與文字篩選邏輯
- 應用篩選結果
- 獲取圖片內容以供篩選

依賴的屬性：
- self.all_image_files: list
- self.image_files: list (當前使用的列表)
- self.filtered_image_files: list
- self.filter_active: bool
- self.current_index: int
- self.filter_input, self.chk_filter_tags, self.chk_filter_text
- self.load_image()
- self.update_file_list_ui() (optional)
"""

from PyQt6.QtWidgets import QApplication
from lib.utils import DanbooruQueryFilter, normalize_for_match, load_image_sidecar
import os

class FilterMixin:
    """篩選功能 Mixin"""
    
    def apply_filter(self):
        """應用篩選條件"""
        query = self.filter_input.text().strip()
        if not query:
            # 清除篩選
            self.filter_active = False
            self.image_files = list(self.all_image_files)
            self.filtered_image_files = []
            
            # Restore index
            if self.current_image_path in self.image_files:
                self.current_index = self.image_files.index(self.current_image_path)
            else:
                self.current_index = 0 if self.image_files else -1
            
            if hasattr(self, 'update_file_list_ui'):
                self.update_file_list_ui() # Update visual list if exists
            
            self.load_image()
            return
            
        # 執行篩選
        self.statusBar().showMessage("Filtering...")
        QApplication.processEvents() if 'QApplication' in globals() else None
        
        matcher = DanbooruQueryFilter(query)
        res = []
        
        for p in self.all_image_files:
            content = self._get_image_content_for_filter(p)
            if matcher.matches(content):
                res.append(p)
                
        self.filtered_image_files = res
        self.image_files = res
        self.filter_active = True
        
        self.current_index = 0 if res else -1
        if res:
             self.current_image_path = res[0]
        else:
             self.current_image_path = ""
             
        if hasattr(self, 'update_file_list_ui'):
            self.update_file_list_ui()
            
        self.load_image()
        self.statusBar().showMessage(f"Filter done. Found {len(res)} images.")

    def _get_image_content_for_filter(self, image_path: str) -> str:
        """Get combined content (tags + text) for filtering."""
        content_parts = []
        
        if hasattr(self, 'chk_filter_tags') and self.chk_filter_tags.isChecked():
            # Get tags from sidecar
            sidecar = load_image_sidecar(image_path)
            tags = sidecar.get("tagger_tags", "")
            content_parts.append(tags)
        
        if hasattr(self, 'chk_filter_text') and self.chk_filter_text.isChecked():
            # Get text from .txt file
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content_parts.append(f.read())
                except Exception:
                    pass
        
        return " ".join(content_parts)
