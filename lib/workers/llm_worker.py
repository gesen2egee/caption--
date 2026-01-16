
from PyQt6.QtCore import QThread, pyqtSignal
from ..data import ImageContext, AppSettings
from ..processors.llm import LLMProcessor
import traceback

class VisionLLMWorker(QThread):
    """
    專門用於單圖 LLM 生成的 Worker。
    """
    finished = pyqtSignal(object) # ImageContext
    error = pyqtSignal(str)

    def __init__(self, ctx: ImageContext, settings: AppSettings):
        super().__init__()
        self.ctx = ctx
        self.settings = settings

    def run(self):
        try:
            # 使用 LLMProcessor 執行
            # 單圖模式下，ctx 可能帶有 UI override 的 user_prompt
            override_prompt = getattr(self.ctx, 'user_prompt', None)
            
            proc = LLMProcessor(self.settings, override_user_prompt=override_prompt, is_batch=False)
            proc.prepare()
            
            success, result = proc.process(self.ctx)
            if success:
                self.ctx.llm_result = result
                self.finished.emit(self.ctx)
            else:
                self.error.emit("LLM failed to generate content.")
                
            proc.cleanup()
        except Exception:
            self.error.emit(traceback.format_exc())
