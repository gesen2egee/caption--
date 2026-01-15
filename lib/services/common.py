
import gc
import traceback

def unload_all_models():
    """ 
    強制執行垃圾回收，並在支援 Torch 的環境下排空 CUDA 快取。
    這有助於在完成 WD14、OCR 或去背景任務後釋放記憶體。
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
