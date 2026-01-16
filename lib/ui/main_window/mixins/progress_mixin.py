"""
批量處理進度管理 Mixin

負責處理：
- 進度條顯示和隱藏
- 批量處理取消

依賴的屬性：
- self.progress_bar - 進度條元件
- self.btn_cancel_batch - 取消按鈕
- self.batch_*_thread - 各種批量處理執行緒
"""


class ProgressMixin:
    """批量處理進度管理 Mixin"""
    
    def show_progress(self):
        """顯示進度條和取消按鈕"""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        if hasattr(self, 'btn_cancel_batch') and self.btn_cancel_batch:
            self.btn_cancel_batch.setVisible(True)

    def hide_progress(self):
        """隱藏進度條和取消按鈕"""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setVisible(False)
        if hasattr(self, 'btn_cancel_batch') and self.btn_cancel_batch:
            self.btn_cancel_batch.setVisible(False)

    def cancel_batch(self):
        """取消當前批量處理"""
        # 嘗試停止所有可能正在運行的批量執行緒
        for thread_attr in ['batch_tagger_thread', 'batch_llm_thread', 
                            'batch_unmask_thread', 'batch_mask_text_thread',
                            'batch_restore_thread']:
            thread = getattr(self, thread_attr, None)
            if thread and thread.isRunning():
                thread.requestInterruption()
                thread.wait(1000)  # 等待最多 1 秒
