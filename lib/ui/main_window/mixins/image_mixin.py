"""
圖片載入和顯示 Mixin

負責處理：
- 圖片載入
- 圖片顯示更新
- 視圖模式切換
- 文本載入

依賴的屬性：
- self.current_image_path, self.current_index
- self.image_label - 圖片顯示標籤
- self.txt_edit - 文本編輯器
- self.current_view_mode, self.temp_view_mode
"""

from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image
import os


class ImageMixin:
    """圖片載入和顯示 Mixin"""
    
    def load_image(self, index=None):
        """載入圖片"""
        if index is not None:
            if not self.image_files or index < 0 or index >= len(self.image_files):
                return
            self.current_index = index
            self.current_image_path = self.image_files[index]
        
        if not self.current_image_path:
            return
        
        # 載入文本
        txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.txt_edit.blockSignals(True)
                self.txt_edit.setPlainText(content)
                self.txt_edit.blockSignals(False)
            except:
                self.txt_edit.setPlainText("")
        else:
            self.txt_edit.setPlainText("")
        
        # 載入標籤
        if hasattr(self, 'build_top_tags_for_current_image'):
            self.top_tags = self.build_top_tags_for_current_image()
        if hasattr(self, 'load_tagger_tags_for_current_image'):
            self.tagger_tags = self.load_tagger_tags_for_current_image()
        
        # 載入 NL
        if hasattr(self, 'load_nl_for_current_image'):
            self.nl_pages = self.load_nl_pages_for_image(self.current_image_path)
            self.nl_latest = self.load_nl_for_current_image()
            self.nl_page_index = len(self.nl_pages) - 1 if self.nl_pages else 0
        
        # 刷新 UI
        if hasattr(self, 'refresh_tags_tab'):
            self.refresh_tags_tab()
        if hasattr(self, 'refresh_nl_tab'):
            self.refresh_nl_tab()
        
        # 更新圖片顯示
        self.update_image_display()
        
        # 更新狀態欄
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(
                f"{self.current_index + 1}/{len(self.image_files)} - {os.path.basename(self.current_image_path)}"
            )

    def update_image_display(self):
        """更新圖片顯示"""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return
        
        try:
            # 決定視圖模式
            view_mode = self.temp_view_mode if self.temp_view_mode is not None else self.current_view_mode
            
            # 載入圖片
            img = Image.open(self.current_image_path)
            
            if view_mode == 1:  # RGB
                img = img.convert("RGB")
            elif view_mode == 2:  # Alpha
                if img.mode == "RGBA":
                    alpha = img.split()[3]
                    img = alpha.convert("RGB")
            
            # 轉換為 QPixmap
            img_bytes = img.tobytes()
            qimg = QPixmap.fromImage(
                img.toqimage() if hasattr(img, 'toqimage') else 
                QPixmap(self.current_image_path)
            )
            
            # 縮放以適應視窗
            if hasattr(self, 'image_label'):
                scaled = qimg.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
        except Exception as e:
            print(f"Image display error: {e}")

    def on_view_mode_changed(self, index):
        """視圖模式變更"""
        self.current_view_mode = index
        self.update_image_display()

    def load_nl_pages_for_image(self, image_path):
        """載入 NL 頁面"""
        from lib.utils import load_image_sidecar
        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if isinstance(pages, list):
            return [p for p in pages if p and str(p).strip()]
        return []

    def load_nl_for_current_image(self):
        """載入當前圖片的 NL"""
        pages = self.load_nl_pages_for_image(self.current_image_path)
        return pages[-1] if pages else ""

    def save_nl_for_image(self, image_path, content):
        """儲存 NL 到圖片"""
        if not content:
            return
        
        content = str(content).strip()
        if not content:
            return
        
        from lib.utils import load_image_sidecar, save_image_sidecar
        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if not isinstance(pages, list):
            pages = []
        pages.append(content)
        sidecar["nl_pages"] = pages
        save_image_sidecar(image_path, sidecar)
