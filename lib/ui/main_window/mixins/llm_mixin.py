"""
LLM 功能 Mixin

負責處理：
- 單圖 LLM 執行
- 批量 LLM 執行
- 批量 LLM 轉 txt
- LLM 回調和後處理

依賴的屬性：
- self.app_settings: AppSettings
- self.current_image_path: str
- self.batch_llm_thread: GenericBatchWorker
- self.on_batch_done()
- self.on_batch_error()
- self.show_progress()
- self.prompt_edit - 提示詞編輯器
"""

from PyQt6.QtWidgets import QMessageBox
from lib.data import ImageContext
from lib.processors.llm import LLMProcessor
from lib.workers.batch import GenericBatchWorker
from lib.workers.llm_worker import VisionLLMWorker
import os


class LLMMixin:
    """LLM 功能 Mixin"""
    
    def run_llm_single(self):
        """單圖運行 LLM"""
        if not self.current_image_path:
            return
            
        custom_prompt = self.prompt_edit.toPlainText()
        
        # Update settings if needed (temporary override or just verify)
        # Here we just pass context
        
        self.statusBar().showMessage("Running LLM...")
        self.btn_run_llm.setEnabled(False)
        if hasattr(self, 'bot_label'):
             self.bot_label.setText(f"<b>{self.tr('label_txt_content')} (Running...)</b>")

        ctx = ImageContext(self.current_image_path)
        # Use current prompt in UI
        ctx.user_prompt = custom_prompt
        
        self.llm_thread = VisionLLMWorker(ctx, self.app_settings)
        self.llm_thread.finished.connect(self.on_llm_single_finished)
        self.llm_thread.error.connect(self.on_llm_single_error)
        self.llm_thread.start()

    def on_llm_single_finished(self, ctx: ImageContext):
        """單圖 LLM 完成回調"""
        self.btn_run_llm.setEnabled(True)
        if hasattr(self, 'bot_label'):
            self.bot_label.setText(f"<b>{self.tr('label_txt_content')}</b>")
        self.statusBar().showMessage("LLM Done", 3000)
        
        # Update UI if current
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(ctx.path):
             # Append result
             result_text = ctx.llm_result or ""
             if result_text:
                 self.nl_latest = result_text
                 # Add to history (nl_pages)
                 self.save_nl_for_image(self.current_image_path, result_text)
                 
                 # Reload to update paging
                 if hasattr(self, 'load_nl_for_current_image'):
                     self.nl_pages = self.load_nl_pages_for_image(self.current_image_path)
                     self.nl_page_index = len(self.nl_pages) - 1
                     self.refresh_nl_tab()
                     self.update_nl_page_controls()

    def on_llm_single_error(self, err):
        """單圖 LLM 錯誤回調"""
        self.btn_run_llm.setEnabled(True)
        if hasattr(self, 'bot_label'):
            self.bot_label.setText(f"<b>{self.tr('label_txt_content')} (Error)</b>")
        QMessageBox.warning(self, "LLM Error", str(err))

    def run_batch_llm(self, to_txt=False):
        """批量運行 LLM"""
        if not self.image_files:
            return
            
        self._is_batch_to_txt = to_txt
        contexts = [ImageContext(p) for p in self.image_files]
        
        # Override user prompt from UI? 
        # Ideally batch uses settings, or current UI prompt?
        # Typically batch uses the prompt template from settings.
        # But if user edited the prompt box, maybe they want to use that?
        # For consistency with legacy code, we might check if we should pass generic overrides.
        # But Processor reads from AppSettings. Let's stick to AppSettings for now unless logic requires otherwise.
        
        # Disable buttons
        self.btn_batch_llm.setEnabled(False)
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.btn_run_llm.setEnabled(False)
        
        proc = LLMProcessor(self.app_settings)
        
        self.batch_llm_thread = GenericBatchWorker(contexts, proc)
        self.batch_llm_thread.progress.connect(self.show_progress)
        self.batch_llm_thread.item_done.connect(self.on_batch_llm_per_image)
        self.batch_llm_thread.finished_all.connect(self.on_batch_llm_all_done)
        self.batch_llm_thread.error.connect(lambda e: self.on_batch_error("LLM", e))
        self.batch_llm_thread.start()

    def on_batch_llm_per_image(self, old_path, new_path):
        """批量 LLM 單圖完成回調"""
        # 如果需要寫入 txt
        if self._is_batch_to_txt:
            # 讀取 sidecar 中的最新 nl_pages
            from lib.utils import load_image_sidecar
            sidecar = load_image_sidecar(new_path)
            pages = sidecar.get("nl_pages", [])
            if pages:
                latest = pages[-1]
                # Write to txt
                txt_path = os.path.splitext(new_path)[0] + ".txt"
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(latest)
                except Exception as e:
                     print(f"Failed to write txt for {new_path}: {e}")

        # 如果是當前圖片，刷新顯示
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
             self.load_image()

    def on_batch_llm_all_done(self):
        """批量 LLM 全部完成回調"""
        self.btn_batch_llm.setEnabled(True)
        self.btn_batch_llm_to_txt.setEnabled(True)
        self.btn_run_llm.setEnabled(True)
        self._is_batch_to_txt = False
        self.on_batch_done("Batch LLM Completed")
