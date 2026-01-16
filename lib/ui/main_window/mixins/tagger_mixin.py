"""
Tagger 功能 Mixin

負責處理：
- 單圖 Tagger 執行
- 批量 Tagger 執行
- 批量 Tagger 轉 txt
- Tagger 回調和後處理

依賴的屬性：
- self.app_settings: AppSettings
- self.current_image_path: str
- self.batch_tagger_thread: GenericBatchWorker
- self.on_batch_done()
- self.on_batch_error()
- self.show_progress()
- self.flow_tagger.sync_state()
"""

from PyQt6.QtWidgets import QMessageBox
from lib.data import ImageContext
from lib.processors.tagger import TaggerProcessor
from lib.workers.batch import GenericBatchWorker
from lib.workers.tagger_worker import TaggerWorker
import os
import json


class TaggerMixin:
    """Tagger 功能 Mixin"""
    
    def auto_tag_current_image(self):
        """單圖自動打標"""
        if not self.current_image_path:
            return
            
        self.statusBar().showMessage(self.tr("msg_tagging"))
        
        # 禁用按鈕防止重複點擊
        if hasattr(self, 'btn_auto_tag'):
            self.btn_auto_tag.setEnabled(False)
        
        ctx = ImageContext(self.current_image_path)
        self.tagger_thread = TaggerWorker(ctx, self.app_settings)
        self.tagger_thread.finished.connect(self.on_auto_tag_finished)
        self.tagger_thread.error.connect(self.on_auto_tag_error)
        self.tagger_thread.start()
    
    def on_auto_tag_error(self, err):
        """單圖打標錯誤回調"""
        if hasattr(self, 'btn_auto_tag'):
            self.btn_auto_tag.setEnabled(True)
        QMessageBox.warning(self, self.tr("error.error_tagger_failed"), str(err))

    def on_auto_tag_finished(self, ctx: ImageContext):
        """單圖打標完成回調"""
        # 恢復按鈕狀態
        if hasattr(self, 'btn_auto_tag'):
            self.btn_auto_tag.setEnabled(True)
            
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(self.tr("status.status_tagging_done"), 3000)
            
        # Update UI if it's still the current image
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(ctx.path):
            # 重新載入整個圖片以刷新所有內容
            self.load_image()
    
    def run_batch_tagger(self, to_txt=False):
        """批量打標"""
        if not self.image_files:
            return
            
        self._is_batch_to_txt = to_txt
        contexts = [ImageContext(p) for p in self.image_files]
        
        # Disable buttons
        self.btn_batch_tagger.setEnabled(False)
        self.btn_batch_tagger_to_txt.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)
        
        proc = TaggerProcessor(self.app_settings)
        
        self.batch_tagger_thread = GenericBatchWorker(contexts, proc)
        self.batch_tagger_thread.progress.connect(self.show_progress)
        self.batch_tagger_thread.item_done.connect(self.on_batch_tagger_per_image)
        self.batch_tagger_thread.finished_all.connect(self.on_batch_tagger_all_done)
        self.batch_tagger_thread.error.connect(lambda e: self.on_batch_error("Tagger", e))
        self.batch_tagger_thread.start()

    def on_batch_tagger_per_image(self, old_path, new_path):
        """批量打標單圖完成回調"""
        # 如果需要寫入 txt
        if self._is_batch_to_txt:
            # 讀取 sidecar 中的 tagger_tags
            from lib.utils import load_image_sidecar
            sidecar = load_image_sidecar(new_path)
            tags = sidecar.get("tagger_tags", "")
            
            if tags:
                # 使用統一的寫入方法，包含過濾和格式化
                if hasattr(self, 'write_batch_result_to_txt'):
                    self.write_batch_result_to_txt(new_path, tags, is_tagger=True)
                else:
                    # Fallback: 直接寫入
                    if self.english_force_lowercase:
                        tags = tags.lower()
                    if self.settings.get("text_auto_format", True):
                        parts = [p.strip() for p in tags.split(",") if p.strip()]
                        tags = ", ".join(parts)
                    
                    txt_path = os.path.splitext(new_path)[0] + ".txt"
                    try:
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(tags)
                    except Exception as e:
                        print(f"Failed to write txt for {new_path}: {e}")

        # 如果是當前圖片，刷新顯示
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
            if hasattr(self, 'load_image'):
                self.load_image()

    def on_batch_tagger_all_done(self):
        """批量打標全部完成回調"""
        self.btn_batch_tagger.setEnabled(True)
        self.btn_batch_tagger_to_txt.setEnabled(True)
        self.btn_auto_tag.setEnabled(True)
        self._is_batch_to_txt = False
        
        # 批次完成後刷新當前圖片顯示
        if hasattr(self, 'load_image') and self.current_image_path:
            self.load_image()
        
        self.on_batch_done("Batch Tagger Completed")
