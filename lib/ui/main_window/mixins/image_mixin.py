"""
圖片載入和顯示 Mixin

負責處理：
- 圖片載入
- 圖片顯示更新
- 視圖模式切換 (Original/RGB/Alpha)
- Context Menu & Clipboard
- 縮放處理

依賴的屬性：
- self.current_image_path, self.current_index
- self.image_label - 圖片顯示標籤
- self.txt_edit - 文本編輯器
- self.current_view_mode, self.temp_view_mode
"""

from PyQt6.QtWidgets import QLabel, QMenu, QApplication
from PyQt6.QtGui import QPixmap, QImage, QAction, QDesktopServices
from PyQt6.QtCore import Qt, QPoint, QUrl, QTimer
from PIL import Image
import os
from lib.utils import load_image_sidecar


class ImageMixin:
    """圖片載入和顯示 Mixin"""
    
    def load_image(self, index=None):
        """載入圖片"""
        if index is not None:
            if not self.image_files or index < 0 or index >= len(self.image_files):
                return
            self.current_index = index
        
        # 根據 current_index 更新 current_image_path
        if self.image_files and 0 <= self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
        
        if not self.current_image_path:
            return
        
        # 確定資料夾路徑以判斷是否需要刷新自定義標籤
        folder_path = os.path.dirname(self.current_image_path)
        
        # 載入文本
        txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
        content = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                pass
        self.txt_edit.blockSignals(True)
        self.txt_edit.setPlainText(content)
        self.txt_edit.blockSignals(False)
        
        # 僅在資料夾變更時載入資料夾自定義標籤
        if not hasattr(self, 'current_folder_path') or self.current_folder_path != folder_path:
            self.current_folder_path = folder_path
            if hasattr(self, 'load_folder_custom_tags'):
                self.custom_tags = self.load_folder_custom_tags(self.current_folder_path)
        
        # 載入標籤數據 (此處僅讀取數據，不刷新 UI)
        if hasattr(self, 'build_top_tags_for_current_image'):
            self.top_tags = self.build_top_tags_for_current_image()
        if hasattr(self, 'load_tagger_tags_for_current_image'):
            self.tagger_tags = self.load_tagger_tags_for_current_image()
        
        # 載入 NL 數據 (此處僅讀取數據，不刷新 UI)
        if hasattr(self, 'load_nl_for_current_image'):
            self.nl_pages = self.load_nl_pages_for_image(self.current_image_path)
            self.nl_latest = self.load_nl_for_current_image()
            self.nl_page_index = len(self.nl_pages) - 1 if self.nl_pages else 0
        
        # 集中刷新 UI 標籤流 (放在最後一次性完成)
        if hasattr(self, 'refresh_tags_tab'):
            self.refresh_tags_tab()
        if hasattr(self, 'refresh_nl_tab'):
            self.refresh_nl_tab()
        
        # 更新圖片顯示 (Load pixmap)
        self.current_pixmap = QPixmap(self.current_image_path)
        # 快取 sidecar 避免在 _get_processed_pixmap 中重複讀取
        self._current_sidecar = load_image_sidecar(self.current_image_path)
        self.update_image_display()
        
        # 更新頂部導航欄 (Top Info Bar)
        if hasattr(self, 'index_input'):
            self.index_input.blockSignals(True)
            self.index_input.setText(str(self.current_index + 1))
            self.index_input.blockSignals(False)
        if hasattr(self, 'total_label'):
            self.total_label.setText(f"/ {len(self.image_files)}")
        if hasattr(self, 'filename_label'):
            self.filename_label.setText(f": {os.path.basename(self.current_image_path)}")

        # 更新狀態欄
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(
                f"{self.current_index + 1}/{len(self.image_files)} - {os.path.basename(self.current_image_path)}"
            )

    def update_image_display(self):
        """更新圖片顯示"""
        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return
            
        try:
            # Get processed pixmap based on view mode
            scaled_pixmap = self._get_processed_pixmap()
            if scaled_pixmap.isNull():
                return
            
            # 縮放以適應視窗
            if hasattr(self, 'image_label'):
                scaled = scaled_pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
        except Exception as e:
            print(f"Image display error: {e}")

    def _get_processed_pixmap(self) -> QPixmap:
        """根據模式獲取處理後的 Pixmap"""
        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return QPixmap()

        # 決定模式
        mode = self.current_view_mode
        if self.temp_view_mode is not None:
            mode = self.temp_view_mode

        # 0: Original (+Mask overlay)
        if mode == 0:
            sidecar = getattr(self, '_current_sidecar', {})
            if not sidecar:
                sidecar = load_image_sidecar(self.current_image_path)
            
            rel_mask = sidecar.get("mask_map_rel_path", "")
            if rel_mask:
                mask_abs = os.path.normpath(os.path.join(os.path.dirname(self.current_image_path), rel_mask))
                if os.path.exists(mask_abs):
                    # 合成
                    try:
                        img_q = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                        mask_q = QImage(mask_abs).convertToFormat(QImage.Format.Format_Alpha8)
                        if img_q.size() == mask_q.size():
                            img_q.setAlphaChannel(mask_q)
                            return QPixmap.fromImage(img_q)
                    except:
                        pass
            return self.current_pixmap
        
        # 1: RGB Only
        if mode == 1:
            img = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
            return QPixmap.fromImage(img)
            
        # 2: Alpha Only
        if mode == 2:
            img = self.current_pixmap.toImage()
            if img.hasAlphaChannel():
                alpha = img.convertToFormat(QImage.Format.Format_Alpha8)
                # Hack to display alpha as grayscale
                ptr = alpha.constBits()
                ptr.setsize(alpha.sizeInBytes())
                gray = QImage(ptr, alpha.width(), alpha.height(), alpha.bytesPerLine(), QImage.Format.Format_Grayscale8)
                return QPixmap.fromImage(gray.copy())
            else:
                white = QPixmap(img.size())
                white.fill(Qt.GlobalColor.white)
                return white
        
        return self.current_pixmap

    def on_view_mode_changed(self, index):
        """視圖模式變更"""
        self.current_view_mode = index
        self.update_image_display()

    def load_nl_pages_for_image(self, image_path):
        """載入 NL 頁面 (Helper)"""
        from lib.utils import load_image_sidecar
        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if isinstance(pages, list):
            return [p for p in pages if p and str(p).strip()]
        return []

    def load_nl_for_current_image(self):
        """載入當前圖片 NL (Helper)"""
        pages = self.load_nl_pages_for_image(self.current_image_path)
        return pages[-1] if pages else ""

    def save_nl_for_image(self, image_path, content):
        """儲存 NL (Helper)"""
        if not content: return
        content = str(content).strip()
        if not content: return
        
        from lib.utils import load_image_sidecar, save_image_sidecar
        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if not isinstance(pages, list): pages = []
        pages.append(content)
        sidecar["nl_pages"] = pages
        save_image_sidecar(image_path, sidecar)
        
    def show_image_context_menu(self, pos: QPoint):
        """顯示圖片右鍵菜單"""
        if not self.current_image_path:
            return

        menu = QMenu(self)
        
        action_copy_img = QAction(self.tr("複製圖片 (Copy Image)") if hasattr(self, "tr") else "Copy Image", self)
        action_copy_img.triggered.connect(self._ctx_copy_image)
        menu.addAction(action_copy_img)
        
        action_copy_path = QAction(self.tr("複製路徑 (Copy Path)") if hasattr(self, "tr") else "Copy Path", self)
        action_copy_path.triggered.connect(self._ctx_copy_path)
        menu.addAction(action_copy_path)
        
        menu.addSeparator()
        
        action_open_dir = QAction(self.tr("打開檔案所在目錄 (Open Folder)") if hasattr(self, "tr") else "Open Folder", self)
        action_open_dir.triggered.connect(self._ctx_open_folder)
        menu.addAction(action_open_dir)
        
        menu.exec(self.image_label.mapToGlobal(pos))

    def _ctx_copy_image(self):
        if hasattr(self, 'current_pixmap') and not self.current_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.current_pixmap)
            if hasattr(self, 'statusBar'): self.statusBar().showMessage("圖片已複製到剪貼簿", 2000)

    def _ctx_copy_path(self):
        if self.current_image_path:
            QApplication.clipboard().setText(os.path.abspath(self.current_image_path))
            if hasattr(self, 'statusBar'): self.statusBar().showMessage("路徑已複製到剪貼簿", 2000)
    
    def _ctx_open_folder(self):
        if self.current_image_path:
            folder = os.path.dirname(self.current_image_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def resizeEvent(self, event):
        """視窗縮放事件"""
        super().resizeEvent(event)
        QTimer.singleShot(10, self.update_image_display)
