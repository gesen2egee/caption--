
from PyQt6.QtCore import QThread, pyqtSignal
from ..data import ImageContext, AppSettings
from ..processors.tagger import TaggerProcessor
import traceback

class TaggerWorker(QThread):
    """
    專門用於單圖打標的 Worker。
    為了相容性，它會發送 ImageContext 物件。
    """
    finished = pyqtSignal(object) # ImageContext
    error = pyqtSignal(str)

    def __init__(self, ctx: ImageContext, settings: AppSettings):
        super().__init__()
        self.ctx = ctx
        self.settings = settings

    def run(self):
        try:
            # 使用 TaggerProcessor 執行
            proc = TaggerProcessor(self.settings)
            proc.prepare()
            
            success, result = proc.process(self.ctx)
            # 將結果填入 ctx (processor 已經填入 sidecar 了，但我們確保記憶體一致)
            if success:
                self.ctx.tagger_tags_list = [t.strip() for t in result.split(",") if t.strip()]
                self.finished.emit(self.ctx)
            else:
                self.error.emit("Tagger failed to process image.")
                
            proc.cleanup()
        except Exception:
            self.error.emit(traceback.format_exc())
