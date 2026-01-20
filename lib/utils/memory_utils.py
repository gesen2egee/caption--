# -*- coding: utf-8 -*-
import gc

def unload_all_models():
    """
    強制釋放 GPU 記憶體與 Python 記憶體。
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    gc.collect()
