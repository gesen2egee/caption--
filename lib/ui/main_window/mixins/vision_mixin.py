"""
視覺處理 Mixin

負責處理：
- 單圖/批量去背 (Unmask)
- 單圖/批量文字遮罩 (Mask Text)
- 單圖/批量還原 (Restore)

依賴的屬性：
- self.app_settings: AppSettings
- self.current_image_path: str
- self.batch_*_thread: GenericBatchWorker
- self.show_progress()
"""

from PyQt6.QtWidgets import QMessageBox
from lib.data import ImageContext
from lib.processors.vision import UnmaskProcessor, TextMaskProcessor, RestoreProcessor
from lib.workers.batch import GenericBatchWorker
import os

try:
    from imgutils.ocr import detect_text_with_ocr
except ImportError:
    detect_text_with_ocr = None


class VisionMixin:
    """視覺處理 Mixin"""
    
    # ==========================
    # Unmask (Background Removal)
    # ==========================
    def unmask_current_image(self):
        """單圖去背"""
        if not self.current_image_path:
            return
        
        try:
            from transparent_background import Remover
        except ImportError:
            QMessageBox.warning(self, "Unmask", "transparent_background.Remover not available")
            return
            
        ctx = ImageContext(self.current_image_path)
        proc = UnmaskProcessor(self.app_settings, is_batch=False)
        
        self.batch_unmask_thread = GenericBatchWorker([ctx], proc)
        self.batch_unmask_thread.progress.connect(self.show_progress)
        self.batch_unmask_thread.item_done.connect(self.on_batch_unmask_per_image)
        self.batch_unmask_thread.finished_all.connect(lambda: self.on_batch_done("單圖去背完成"))
        self.batch_unmask_thread.error.connect(lambda e: QMessageBox.warning(self, "Error", f"Unmask 失敗: {e}"))
        self.batch_unmask_thread.start()

    def run_batch_unmask_background(self):
        """批量去背"""
        if not self.image_files:
            return
        
        try:
            from transparent_background import Remover
        except ImportError:
            QMessageBox.warning(self, "Unmask", "transparent_background.Remover not available")
            return
            
        contexts = [ImageContext(p, load_sidecar=True) for p in self.image_files]
        proc = UnmaskProcessor(self.app_settings, is_batch=True)
        
        self.batch_unmask_thread = GenericBatchWorker(contexts, proc)
        self.batch_unmask_thread.progress.connect(self.show_progress)
        self.batch_unmask_thread.item_done.connect(self.on_batch_unmask_per_image) # Use same callback
        self.batch_unmask_thread.finished_all.connect(lambda: self.on_batch_done("批量去背完成"))
        self.batch_unmask_thread.error.connect(lambda e: self.on_batch_error("Mask", e))
        self.batch_unmask_thread.start()

    def on_batch_unmask_per_image(self, old_path: str, new_path: str):
        """去背完成回調（單圖/批量共用）"""
        self._replace_image_path_in_list(old_path, new_path)
        # 如果是當前圖片，重新載入以顯示結果
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
            self.load_image()

    # ==========================
    # Mask Text (OCR)
    # ==========================
    def mask_text_current_image(self):
        """單圖文字遮罩"""
        if not self.current_image_path:
            return
        if not bool(self.app_settings.get("mask_batch_detect_text_enabled", True)):
            QMessageBox.information(self, "Info", "OCR text detection is disabled in settings.")
            return

        if detect_text_with_ocr is None:
             QMessageBox.warning(self, "Mask Text", self.tr("setting_mask_ocr_hint"))
             return

        ctx = ImageContext(self.current_image_path)
        proc = TextMaskProcessor(self.app_settings, is_batch=False)
        
        self.batch_mask_text_thread = GenericBatchWorker([ctx], proc)
        self.batch_mask_text_thread.progress.connect(self.show_progress)
        self.batch_mask_text_thread.item_done.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.finished_all.connect(lambda: self.on_batch_done("Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Mask Text", e))
        self.batch_mask_text_thread.start()

    def run_batch_mask_text(self):
        """批量文字遮罩"""
        if not self.image_files:
            return
        if not bool(self.app_settings.get("mask_batch_detect_text_enabled", True)):
            QMessageBox.information(self, "Info", "OCR text detection is disabled in settings.")
            return
        if detect_text_with_ocr is None:
             QMessageBox.warning(self, "Mask Text", self.tr("setting_mask_ocr_hint"))
             return
             
        contexts = [ImageContext(p, load_sidecar=True) for p in self.image_files]
        proc = TextMaskProcessor(self.app_settings, is_batch=True)
        
        self.batch_mask_text_thread = GenericBatchWorker(contexts, proc)
        self.batch_mask_text_thread.progress.connect(self.show_progress)
        self.batch_mask_text_thread.item_done.connect(self.on_batch_mask_text_per_image)
        self.batch_mask_text_thread.finished_all.connect(lambda: self.on_batch_done("批量 Mask Text 完成"))
        self.batch_mask_text_thread.error.connect(lambda e: self.on_batch_error("Mask Text", e))
        self.batch_mask_text_thread.start()

    def on_batch_mask_text_per_image(self, old_path, new_path):
        """文字遮罩完成回調"""
        if old_path != new_path:
            self._replace_image_path_in_list(old_path, new_path)
            if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
                self.load_image()

    # ==========================
    # Restore
    # ==========================
    def restore_current_image(self):
        """單圖還原"""
        if not self.current_image_path:
            return
        
        ctx = ImageContext(self.current_image_path)
        proc = RestoreProcessor(self.app_settings)
        
        self.batch_restore_thread = GenericBatchWorker([ctx], proc)
        self.batch_restore_thread.item_done.connect(lambda c, r: self.load_image())
        self.batch_restore_thread.error.connect(lambda e: self.statusBar().showMessage(f"Restore failed: {e}", 3000))
        self.batch_restore_thread.start()

    def run_batch_restore(self):
        """批量還原"""
        if not self.image_files:
            return
            
        contexts = [ImageContext(p) for p in self.image_files]
        proc = RestoreProcessor(self.app_settings)
        
        self.batch_restore_thread = GenericBatchWorker(contexts, proc)
        self.batch_restore_thread.progress.connect(self.show_progress)
        self.batch_restore_thread.item_done.connect(lambda c, r: None) # No path change
        self.batch_restore_thread.finished_all.connect(lambda: self.on_batch_done("批量還原完成"))
        self.batch_restore_thread.error.connect(lambda e: self.on_batch_error("Restore", e))
        self.batch_restore_thread.start()
