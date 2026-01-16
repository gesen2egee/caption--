"""
檔案管理 Mixin

負責處理：
- 開啟目錄
- 刷新檔案列表
- 掃描圖片檔案
- 第一張/最後一張圖片導航

依賴的屬性：
- self.root_dir_path: str - 根目錄路徑
- self.image_files: list - 圖片檔案列表
- self.current_index: int - 當前索引
- self.settings: dict - 應用程式設定
"""

from PyQt6.QtWidgets import QFileDialog
import os
from natsort import natsorted


class FileMixin:
    """檔案管理 Mixin"""
    
    def open_directory(self):
        """開啟目錄對話框並載入圖片"""
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾", self.root_dir_path or "")
        if folder:
            self.root_dir_path = folder
            self.settings["last_open_dir"] = folder
            self.refresh_file_list()

    def refresh_file_list(self):
        """刷新檔案列表"""
        if not self.root_dir_path or not os.path.isdir(self.root_dir_path):
            return
        
        self.image_files = self._scan_images(self.root_dir_path)
        self.all_image_files = list(self.image_files)
        
        if self.image_files:
            self.current_index = 0
            self.load_image(0)
        else:
            self.current_index = -1
            self.current_image_path = ""

    def _scan_images(self, folder):
        """掃描資料夾中的圖片檔案"""
        exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        files = []
        for f in os.listdir(folder):
            if f.lower().endswith(exts):
                files.append(os.path.join(folder, f))
        return natsorted(files)

    def first_image(self):
        """跳到第一張圖片"""
        if self.image_files:
            self.load_image(0)

    def last_image(self):
        """跳到最後一張圖片"""
        if self.image_files:
            self.load_image(len(self.image_files) - 1)
