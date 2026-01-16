"""
篩選功能 Mixin

負責處理：
- Danbooru 查詢篩選
- 篩選應用和清除

依賴的屬性：
- self.filter_active: bool - 篩選是否啟用
- self.all_image_files: list - 所有圖片列表
- self.filtered_image_files: list - 篩選後的圖片列表
- self.image_files: list - 當前圖片列表
- self.filter_edit - 篩選輸入框
"""

from lib.utils import DanbooruQueryFilter, load_image_sidecar
import os


class FilterMixin:
    """篩選功能 Mixin"""
    
    def apply_filter(self):
        """應用 Danbooru 篩選"""
        query = self.filter_edit.text().strip()
        if not query:
            self.clear_filter()
            return
        
        try:
            filter_obj = DanbooruQueryFilter(query)
            self.filtered_image_files = []
            
            for img_path in self.all_image_files:
                # 載入 sidecar 和 txt 內容
                sidecar = load_image_sidecar(img_path)
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                txt_content = ""
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            txt_content = f.read()
                    except:
                        pass
                
                # 組合搜尋內容
                search_content = f"{txt_content} {sidecar.get('tagger_tags', '')}"
                
                if filter_obj.matches(search_content):
                    self.filtered_image_files.append(img_path)
            
            self.image_files = self.filtered_image_files
            self.filter_active = True
            
            if self.image_files:
                self.load_image(0)
            else:
                self.current_index = -1
                self.current_image_path = ""
                
        except Exception as e:
            print(f"Filter error: {e}")

    def clear_filter(self):
        """清除篩選"""
        self.filter_active = False
        self.image_files = list(self.all_image_files)
        self.filtered_image_files = []
        
        if self.image_files:
            self.load_image(0)
