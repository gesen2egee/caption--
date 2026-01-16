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
        """
        遞迴掃描資料夾中的圖片檔案
        
        排除以下資料夾：no_used, unmask, raw_image, mask
        """
        exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        excluded_folders = {'no_used', 'unmask', 'raw_image', 'mask'}
        files = []
        
        try:
            # 掃描當前資料夾的圖片
            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.lower().endswith(exts):
                    files.append(entry.path)
            
            # 遞迴掃描子資料夾
            for entry in os.scandir(folder):
                if entry.is_dir():
                    # 檢查是否為排除的資料夾
                    if entry.name.lower() not in excluded_folders:
                        # 遞迴掃描子資料夾
                        subfolder_files = self._scan_images(entry.path)
                        files.extend(subfolder_files)
        except PermissionError:
            # 忽略無權限的資料夾
            pass
        except Exception as e:
            print(f"[Scan] 掃描資料夾時發生錯誤 {folder}: {e}")
        
        return natsorted(files)

    def first_image(self):
        """跳到第一張圖片"""
        if self.image_files:
            self.load_image(0)

    def last_image(self):
        """跳到最後一張圖片"""
        if self.image_files:
            self.load_image(len(self.image_files) - 1)
