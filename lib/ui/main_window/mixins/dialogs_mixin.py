"""
對話框管理 Mixin

負責處理各種對話框的開啟和管理：
- 設定對話框
- 查找替換對話框  
- 手繪橡皮擦對話框

依賴的屬性：
- self.settings: dict - 應用程式設定
- self.current_image_path: str - 當前圖片路徑
- self.apply_theme() - 應用主題方法
- self.retranslate_ui() - 重新翻譯 UI 方法
"""

from PyQt6.QtWidgets import QMessageBox
from lib.ui.dialogs import SettingsDialog, AdvancedFindReplaceDialog, StrokeEraseDialog
from lib.data import save_app_settings
from lib.utils import backup_raw_image
from PIL import Image
import numpy as np


class DialogsMixin:
    """對話框管理 Mixin"""
    
    def open_settings(self):
        """開啟設定對話框"""
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            new_cfg = dlg.get_config()
            self.settings.update(new_cfg)
            save_app_settings(self.settings)
            self.apply_theme()
            self.retranslate_ui()

    def open_find_replace(self):
        """開啟查找替換對話框"""
        dlg = AdvancedFindReplaceDialog(self)
        if dlg.exec():
            opts = dlg.get_options()
            self._do_find_replace(opts)

    def open_stroke_eraser(self):
        """開啟手繪橡皮擦對話框"""
        if not self.current_image_path:
            return
        
        try:
            dlg = StrokeEraseDialog(self.current_image_path, self)
            if dlg.exec():
                mask_img, pen_width = dlg.get_result()
                self.stroke_erase_to_webp(mask_img, pen_width)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"無法開啟手繪橡皮擦: {e}")

    def stroke_erase_to_webp(self, mask_qimage, pen_width):
        """
        使用手繪遮罩處理圖片
        
        Args:
            mask_qimage: QImage 格式的遮罩
            pen_width: 筆畫寬度
        """
        if not self.current_image_path:
            return
        
        try:
            # 備份原圖
            backup_raw_image(self.current_image_path)
            
            # 載入原圖
            img = Image.open(self.current_image_path).convert("RGBA")
            
            # 轉換 QImage 遮罩為 numpy array
            w, h = mask_qimage.width(), mask_qimage.height()
            ptr = mask_qimage.bits()
            ptr.setsize(w * h)
            mask_arr = np.frombuffer(ptr, np.uint8).reshape((h, w))
            
            # 應用遮罩
            img_arr = np.array(img)
            alpha = img_arr[:, :, 3]
            # 將遮罩區域的 alpha 設為 0
            alpha[mask_arr > 127] = 0
            img_arr[:, :, 3] = alpha
            
            # 儲存為 WebP
            result = Image.fromarray(img_arr, "RGBA")
            result.save(self.current_image_path, "WebP", lossless=True)
            
            # 重新載入圖片
            self.load_image(self.current_index)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"手繪橡皮擦處理失敗: {e}")
