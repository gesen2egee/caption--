# -*- coding: utf-8 -*-
import sys
import os
import warnings
import gc

# silence some noisy third-party warnings
os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_fwd")
warnings.filterwarnings("ignore", message="Failed to import flet")
warnings.filterwarnings("ignore", message="Token indices sequence length")

# [GPU Fix] 嘗試載入 pip 安裝的 NVIDIA dll
if os.name == 'nt':
    try:
        import nvidia.cudnn
        import nvidia.cublas
        libs = [
            os.path.dirname(nvidia.cudnn.__file__),
            os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin"),
            os.path.dirname(nvidia.cublas.__file__),
            os.path.join(os.path.dirname(nvidia.cublas.__file__), "bin"),
        ]
        for lib in libs:
            if os.path.exists(lib):
                os.add_dll_directory(lib)
    except Exception:
        pass

from PyQt6.QtWidgets import QApplication

# Import the refactored MainWindow
from lib.ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())