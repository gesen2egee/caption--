import os
# Read original file
try:
    with open('lib/ui/main_window.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
except FileNotFoundError:
    print("File not found")
    exit(1)

# Logic to find classes
start_idx = 0
end_idx = len(lines)

for i, line in enumerate(lines):
    if line.strip().startswith('class MainWindow(QMainWindow):'):
        start_idx = i
        break

# Find end (remove if __name__)
for i in range(len(lines)-1, 0, -1):
    if line.strip().startswith('if __name__ == "__main__":'):
        end_idx = i
        break
    if lines[i].strip().startswith('if __name__ == "__main__":'):
        end_idx = i
        break

content_lines = lines[start_idx:end_idx]

imports = """# -*- coding: utf-8 -*-
import os
import sys
import shutil
import json
import re
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QLineEdit, QLabel, QPushButton, QCheckBox, QComboBox, 
    QScrollArea, QTabWidget, QTextEdit, QPlainTextEdit, 
    QProgressBar, QMessageBox, QFrame, QMenu, QFileDialog, 
    QInputDialog, QApplication, QStyle, QLayout
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize, QUrl, QBuffer, QIODevice, QByteArray, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QAction, QKeySequence, QShortcut, QDesktopServices, QPalette, QBrush, QIcon, QTextCursor

from natsort import natsorted
from PIL import Image, ImageChops

from lib.core.settings import (
    load_app_settings, save_app_settings, DEFAULT_APP_SETTINGS, 
    DEFAULT_CUSTOM_TAGS, DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CUSTOM_PROMPT_TEMPLATE
)
from lib.core.dataclasses import Settings, ImageData
from lib.locales import load_locale, locale_tr
from lib.utils.file_ops import (
    load_image_sidecar, save_image_sidecar, 
    backup_raw_image, restore_raw_image, has_raw_backup,
    delete_matching_npz, get_raw_image_dir
)
from lib.utils.parsing import (
    smart_parse_tags, extract_bracket_content, is_basic_character_tag, 
    try_tags_to_text_list, cleanup_csv_like_text
)
from lib.utils.boorutag import parse_boorutag_meta
from lib.utils.query_filter import DanbooruQueryFilter

from lib.ui.themes import THEME_STYLES
from lib.ui.components.tag_flow import TagFlowWidget, TagButton
from lib.ui.components.stroke import StrokeEraseDialog, create_checkerboard_png_bytes
from lib.ui.dialogs.find_replace import AdvancedFindReplaceDialog
from lib.ui.dialogs.settings_dialog import SettingsDialog

from lib.pipeline.manager import PipelineManager

def unload_all_models():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

try:
    from transparent_background import Remover
except ImportError:
    Remover = None

try:
    from transformers import AutoTokenizer, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPTokenizer = None

"""

with open('lib/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(imports)
    f.writelines(content_lines)

print("Migration successful")
