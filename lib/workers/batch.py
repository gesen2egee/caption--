
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List

from ..data import ImageContext
from ..processors.base import BaseProcessor

class GenericBatchWorker(QThread):
    """
    通用的批量處理執行緒。
    將具體邏輯委派給傳入的 Processor 執行。
    """
    progress = pyqtSignal(int, int, str)   # index, total, filename
    item_done = pyqtSignal(str, object)    # path, result_data (Tag string, or New Path, etc)
    finished_all = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, contexts: List[ImageContext], processor: BaseProcessor):
        super().__init__()
        self.contexts = contexts
        self.processor = processor
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self.processor.prepare()
            total = len(self.contexts)
            
            for i, ctx in enumerate(self.contexts, start=1):
                if self._stop_flag:
                    break
                
                # Emit progress before processing
                self.progress.emit(i, total, ctx.filename)
                
                try:
                    changed, result = self.processor.process(ctx)
                    
                    # 只要處理成功，就發送 item_done (即便 changed=False，有時也需要回報結果)
                    # Processor 可以透過回傳 result=None 來決定不發送
                    if result is not None:
                        self.item_done.emit(ctx.path, result)
                        
                except Exception as e:
                    print(f"[Batch Error] {ctx.filename}: {e}")
                    # 個別圖片錯誤不中斷整個 Batch，但可考慮發送錯誤信號
                    # self.error.emit(f"{ctx.filename}: {str(e)}")
            
            self.finished_all.emit()
            
        except Exception as e:
            self.error.emit(traceback.format_exc())
            
        finally:
            self.processor.cleanup()
