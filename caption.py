# ============================================================
#  Caption 神器
# ============================================================

import sys
import os
import shutil
import csv
import base64
import re
import json
import traceback
import warnings
import inspect
import gc  # 新增垃圾回收支援

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

from pathlib import Path
from io import BytesIO
from urllib.request import urlopen, Request
from natsort import natsorted

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QSplitter,
    QScrollArea, QLineEdit, QDialog, QFormLayout, QComboBox,
    QCheckBox, QMessageBox, QPlainTextEdit, QInputDialog,
    QRadioButton, QGroupBox, QSizePolicy, QTabWidget,
    QFrame, QProgressBar, QSlider, QSpinBox, QDoubleSpinBox,
    QMenu
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, 
    QBuffer, QIODevice, QByteArray, QTimer
)
from PyQt6.QtGui import (
    QPixmap, QKeySequence, QAction, QShortcut, QFont,
    QPalette, QBrush, QPainter, QPen, QColor, QImage, QTextCursor,
    QCursor, QDesktopServices, QIcon
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, 
    QBuffer, QIODevice, QByteArray, QTimer, QUrl
)

from PIL import Image

# optional: clip_anytokenizer for token counting
# 嘗試匯入 transformers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not found, falling back to regex token counting.")

# [Refactor Imports]
from lib.data import AppSettings, ImageContext, load_app_settings, save_app_settings
from lib.const import (
    DEFAULT_CUSTOM_TAGS, LOCALIZATION, THEME_STYLES, 
    DEFAULT_APP_SETTINGS, TEMPLATE_DEFAULT, DEFAULT_SYSTEM_PROMPT
)
from lib.workers.batch import GenericBatchWorker
from lib.processors.tagger import TaggerProcessor
from lib.processors.llm import LLMProcessor
from lib.processors.vision import UnmaskProcessor, TextMaskProcessor, RestoreProcessor
from lib.utils import (
    create_checkerboard_png_bytes, delete_matching_npz, image_sidecar_json_path,
    load_image_sidecar, save_image_sidecar, backup_raw_image, restore_raw_image,
    has_raw_backup, get_raw_image_dir, delete_raw_backup, ensure_tags_csv,
    load_translations, parse_boorutag_meta, smart_parse_tags, extract_bracket_content,
    is_basic_character_tag, cleanup_csv_like_text, split_csv_like_text,
    try_tags_to_text_list, remove_underline, DanbooruQueryFilter, normalize_for_match
)
from lib.services.tagger import call_wd14
from lib.services.common import unload_all_models

# Optional: OCR text detection (used in MainWindow.mask_text_current_image)
try:
    from imgutils.ocr import detect_text_with_ocr
except ImportError:
    detect_text_with_ocr = None

# Optional: CLIPTokenizer for token counting
try:
    from clip_anytokenizer import CLIPTokenizer
except ImportError:
    CLIPTokenizer = None

os.environ['ONNX_MODE'] = 'gpu'

# ==========================================
#  Configuration & Globals
# Note: Utility functions (load_app_settings, call_wd14, etc.) 
# have been moved to lib.utils, lib.services, and lib.data.

# Danbooru filter and Utilities have been moved to lib.utils

# ==========================================
#  Main Window
# ==========================================
# UI classes (StrokeCanvas, StrokeEraseDialog, TagButton, TagFlowWidget,
# AdvancedFindReplaceDialog, SettingsDialog) have been moved to lib.ui

# Import Mixins
from lib.ui.main_window.mixins.shortcuts_mixin import ShortcutsMixin
from lib.ui.main_window.mixins.theme_mixin import ThemeMixin
from lib.ui.main_window.mixins.nl_mixin import NLMixin
from lib.ui.main_window.mixins.dialogs_mixin import DialogsMixin
from lib.ui.main_window.mixins.progress_mixin import ProgressMixin
from lib.ui.main_window.mixins.file_mixin import FileMixin
from lib.ui.main_window.mixins.filter_mixin import FilterMixin
from lib.ui.main_window.mixins.navigation_mixin import NavigationMixin
from lib.ui.main_window.mixins.text_edit_mixin import TextEditMixin
from lib.ui.main_window.mixins.tags_mixin import TagsMixin
from lib.ui.main_window.mixins.image_mixin import ImageMixin
from lib.ui.main_window.mixins.batch_base_mixin import BatchBaseMixin
from lib.ui.main_window.mixins.vision_mixin import VisionMixin
from lib.ui.main_window.mixins.tagger_mixin import TaggerMixin
from lib.ui.main_window.mixins.llm_mixin import LLMMixin
from lib.ui.main_window.mixins.app_core_mixin import AppCoreMixin
from lib.ui.main_window.mixins.batch_export_mixin import BatchExportMixin

class MainWindow(ShortcutsMixin, ThemeMixin, NLMixin, DialogsMixin, ProgressMixin, 
                 FileMixin, FilterMixin, NavigationMixin, TextEditMixin, TagsMixin,
                 ImageMixin, BatchBaseMixin, VisionMixin, TaggerMixin, LLMMixin, 
                 BatchExportMixin, AppCoreMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Captioning Assistant")
        self._clip_tokenizer = None
        self.resize(1600, 1000)

        self.settings = load_app_settings()
        self.app_settings = AppSettings(self.settings)

        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", DEFAULT_APP_SETTINGS["llm_api_key"]))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        self.default_user_prompt_template = self.app_settings.user_prompt_template
        # self.custom_prompt_template = ... [Removed]
        self.current_prompt_mode = "default"
        self.default_custom_tags_global = list(self.settings.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
        self.english_force_lowercase = bool(self.settings.get("english_force_lowercase", True))

        self.current_view_mode = 0  # 0=Original, 1=RGB, 2=Alpha
        self.temp_view_mode = None  # For N/M keys override
        
        self.translations_csv = load_translations()

        self.image_files = []
        self.current_index = -1
        self.current_image_path = ""
        self.current_folder_path = ""

        self.top_tags = []
        self.custom_tags = []
        self.tagger_tags = []

        self.root_dir_path = ""
        
        # Filter state
        self.filter_active = False
        self.filtered_image_files = []
        self.all_image_files = []  # Original unfiltered list

        self.nl_pages = []
        self.nl_page_index = 0
        self.nl_latest = ""

        self.batch_tagger_thread = None
        self.batch_llm_thread = None
        self.batch_unmask_thread = None
        self.batch_mask_text_thread = None
        self.tagger_thread = None # Add missing single threads
        self.llm_thread = None

        self.init_ui()
        self.apply_theme()
        self.setup_shortcuts()
        self._hf_tokenizer = None

        # Auto-load last directory
        last_dir = self.settings.get("last_open_dir", "")
        if last_dir and os.path.exists(last_dir):
            self.root_dir_path = last_dir
            self.refresh_file_list()

        # Check CUDA availability
        try:
            import torch
            if not torch.cuda.is_available():
                # Use singleShot to show message after UI is fully loaded
                QTimer.singleShot(1000, lambda: QMessageBox.warning(
                    self, 
                    "CUDA Warning", 
                    "偵測不到 NVIDIA GPU (CUDA)。\n\n這可能是因為 venv 中的 PyTorch 版本錯誤。\n請執行根目錄下的 'fix_torch_gpu.bat' 來修復。\n\n目前將使用 CPU 執行，速度會非常慢。"
                ))
        except ImportError:
            pass

    def init_ui(self):
        self.setup_ui_components()

    def wheelEvent(self, event):
        pos = event.position().toPoint()
        widget = self.childAt(pos)
        if widget is self.image_label or (widget and self.image_label.isAncestorOf(widget)):
            dy = event.angleDelta().y()
            if dy > 0:
                self.prev_image()
            elif dy < 0:
                self.next_image()
            event.accept()
            return
        super().wheelEvent(event)

    # ==========================
    # Logic: Storage & Init
    # ==========================
    def _get_image_content_for_filter(self, image_path: str) -> str:
        """Get combined content (tags + text) for filtering."""
        content_parts = []
        
        if self.chk_filter_tags.isChecked():
            # Get tags from sidecar
            sidecar = load_image_sidecar(image_path)
            tags = sidecar.get("tagger_tags", "")
            content_parts.append(tags)
        
        if self.chk_filter_text.isChecked():
            # Get text from .txt file
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content_parts.append(f.read())
                except Exception:
                    pass
        
        return " ".join(content_parts)

    def jump_to_index(self):
        try:
            val = int(self.index_input.text())
            target_idx = val - 1
            
            if self.filter_active:
                if 0 <= target_idx < len(self.filtered_image_files):
                    target_path = self.filtered_image_files[target_idx]
                    self.current_index = self.image_files.index(target_path)
                    self.load_image()
                else:
                    self.load_image() # Reset to current
            else:
                if 0 <= target_idx < len(self.image_files):
                    self.current_index = target_idx
                    self.load_image()
                else:
                    self.load_image() # Reset to current
        except Exception:
            self.load_image()

    def _get_processed_pixmap(self) -> QPixmap:
        """
        根據目前的檢視模式 (Original/RGB/Alpha) 回傳對應的 QPixmap。
        優先順序: 
        1. 暫時按鍵 (temp_view_mode: N/M)
        2. 下拉選單 (current_view_mode)
        """
        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return QPixmap()

        # 決定要用的模式 (0=Orig, 1=RGB, 2=Alpha)
        mode = self.current_view_mode
        if self.temp_view_mode is not None:
            mode = self.temp_view_mode

        # 如果是「原圖模式」，檢查是否有外部遮罩 (mask/*.png)
        if mode == 0:
            sidecar = load_image_sidecar(self.current_image_path)
            rel_mask = sidecar.get("mask_map_rel_path", "")
            if rel_mask:
                mask_abs = os.path.normpath(os.path.join(os.path.dirname(self.current_image_path), rel_mask))
                if os.path.exists(mask_abs):
                    # 動態合成
                    img_q = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                    mask_q = QImage(mask_abs).convertToFormat(QImage.Format.Format_Alpha8)
                    if img_q.size() == mask_q.size():
                        img_q.setAlphaChannel(mask_q)
                        return QPixmap.fromImage(img_q)

            return self.current_pixmap
        
        # 轉換處理
        img = self.current_pixmap.toImage()
        
        if mode == 1: # RGB Only (Force Opaque)
            # 轉換為 RGB888 (丟棄 Alpha)
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            return QPixmap.fromImage(img)
            
        elif mode == 2: # Alpha Only
            if img.hasAlphaChannel():
                # Convert to Alpha8 (data is 8-bit alpha)
                alpha_img = img.convertToFormat(QImage.Format.Format_Alpha8)
                # Interpret the data as Grayscale8
                # Note: We must ensure the data persists or is copied.
                # Constructing QImage from inputs shares memory? 
                # Safe way: Convert Alpha8 to RGBA, then fill RGB with Alpha? No.
                
                # Using PIL is robust and easy if we can convert.
                ptr = alpha_img.constBits()
                ptr.setsize(alpha_img.sizeInBytes())
                # Create Grayscale QImage from the bytes
                gray_img = QImage(ptr, alpha_img.width(), alpha_img.height(), alpha_img.bytesPerLine(), QImage.Format.Format_Grayscale8)
                return QPixmap.fromImage(gray_img.copy()) # copy to detach
            else:
                # No Alpha -> White
                white = QPixmap(img.size())
                white.fill(Qt.GlobalColor.white)
                return white
        
        return self.current_pixmap

    def show_image_context_menu(self, pos: QPoint):
        if not self.current_image_path:
            return

        menu = QMenu(self)
        
        # 1. 複製圖片
        action_copy_img = QAction("複製圖片 (Copy Image)", self)
        action_copy_img.triggered.connect(self._ctx_copy_image)
        menu.addAction(action_copy_img)
        
        # 2. 複製路徑
        action_copy_path = QAction("複製路徑 (Copy Path)", self)
        action_copy_path.triggered.connect(self._ctx_copy_path)
        menu.addAction(action_copy_path)
        
        menu.addSeparator()
        
        # 3. 開啟檔案位置
        action_open_dir = QAction("打開檔案所在目錄 (Open Folder)", self)
        action_open_dir.triggered.connect(self._ctx_open_folder)
        menu.addAction(action_open_dir)
        
        menu.exec(self.image_label.mapToGlobal(pos))

    def _ctx_copy_image(self):
        if hasattr(self, 'current_pixmap') and not self.current_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.current_pixmap)
            self.statusBar().showMessage("圖片已複製到剪貼簿", 2000)

    def _ctx_copy_path(self):
        if self.current_image_path:
            QApplication.clipboard().setText(os.path.abspath(self.current_image_path))
            self.statusBar().showMessage("路徑已複製到剪貼簿", 2000)
    
    def _ctx_open_folder(self):
        if self.current_image_path:
            folder = os.path.dirname(self.current_image_path)
            # 使用 QDesktopServices 開啟目錄
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 使用 QTimer 避免縮放時過於頻繁的重繪造成卡頓
        QTimer.singleShot(10, self.update_image_display)

    def _get_clip_tokenizer(self):
        if CLIPTokenizer is None:
            return None
        if self._clip_tokenizer is None:
            try:
                self._clip_tokenizer = CLIPTokenizer("openai/clip-vit-large-patch14")
            except Exception:
                self._clip_tokenizer = None
        return self._clip_tokenizer

    def folder_custom_tags_path(self, folder_path):
        return os.path.join(folder_path, ".custom_tags.json")

    def run_tagger(self):
        if not self.current_image_path:
            return
        self.btn_auto_tag.setEnabled(False)
        self.btn_auto_tag.setText("Tagging...")

        # [Refactor] Use GenericBatchWorker for single image
        ctx = ImageContext(self.current_image_path)
        # Check settings for tagger? TaggerProcessor uses self.settings internally
        proc = TaggerProcessor(self.app_settings)
        
        self.tagger_thread = GenericBatchWorker([ctx], proc)
        # Adapt output to match legacy slot: item_done(ctx, result) -> on_tagger_finished(result)
        self.tagger_thread.item_done.connect(lambda ctx, res: self.on_tagger_finished(res))
        self.tagger_thread.error.connect(self.on_tagger_error_no_popup)
        # We don't need finished_all for single item logic if we use item_done, 
        # but cleanup happens in finished_all usually? 
        # Actually worker cleans up processor in run() after loop.
        self.tagger_thread.start()

    def run_llm_generation(self):
        if not self.current_image_path:
            return

        tags_text = self.build_llm_tags_context_for_image(self.current_image_path)
        user_prompt = self.prompt_edit.toPlainText()

        self.btn_run_llm.setEnabled(False)
        self.btn_run_llm.setText("Running LLM...")

        # Check empty tags logic
        if "{tags}" in user_prompt and not tags_text.strip():
            reply = QMessageBox.question(
                self, "Warning", 
                "Prompt 包含 {tags} 但目前沒有標籤資料。\n確定要繼續嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.btn_run_llm.setEnabled(True)
                self.btn_batch_llm.setEnabled(True)
                return

        # [Refactor] Use GenericBatchWorker for single image
        ctx = ImageContext(self.current_image_path)
        
        # Determine strict settings usage. Legacy used self.settings dict.
        # Now using self.app_settings and processor
        proc = LLMProcessor(self.app_settings, override_user_prompt=user_prompt, is_batch=False)
        
        self.llm_thread = GenericBatchWorker([ctx], proc)
        # item_done -> (ctx, result_content)
        self.llm_thread.item_done.connect(lambda ctx, res: self.on_llm_finished_latest_only(res))
        self.llm_thread.error.connect(self.on_llm_error_no_popup)
        self.llm_thread.start()

    def on_tagger_error_no_popup(self, e):
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText("Auto Tag (WD14)")
        self.statusBar().showMessage(f"Tagger error: {e}", 5000)

    def on_tagger_finished(self, raw_tags_str):
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText("Auto Tag (WD14)")

        self.save_tagger_tags_for_image(self.current_image_path, raw_tags_str)

        parts = [x.strip() for x in raw_tags_str.split(",") if x.strip()]
        parts = try_tags_to_text_list(parts)
        parts = [t.replace("_", " ").strip() for t in parts if t.strip()]

        self.tagger_tags = parts
        self.refresh_tags_tab()
        self.on_text_changed()

    def build_llm_tags_context_for_image(self, image_path: str) -> str:
        top_tags = []
        try:
            hints = []
            tags_from_meta = []

            meta_path = str(image_path) + ".boorutag"
            if os.path.isfile(meta_path):
                tags_meta, hint_info = parse_boorutag_meta(meta_path)
                tags_from_meta.extend(tags_meta)
                hints.extend(hint_info)

            parent = Path(image_path).parent.name
            if "_" in parent:
                folder_hint = parent.split("_", 1)[1]
                if "{" not in folder_hint:
                    folder_hint = f"{{{folder_hint}}}"
                hints.append(folder_hint)

            initial_keywords = []
            for h in hints:
                initial_keywords.extend(extract_bracket_content(h))

            combined = initial_keywords + tags_from_meta
            seen = set()
            top_tags = [x for x in combined if not (x in seen or seen.add(x))]
            top_tags = [str(t).replace("_", " ").strip() for t in top_tags if str(t).strip()]
        except Exception:
            top_tags = []

        tagger_parts = []
        sidecar = load_image_sidecar(image_path)
        raw = sidecar.get("tagger_tags", "")
        if raw:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
            parts = try_tags_to_text_list(parts)
            tagger_parts = [t.replace("_", " ").strip() for t in parts if t.strip()]

        all_tags = []
        seen2 = set()
        for t in (top_tags + tagger_parts):
            if t and t not in seen2:
                seen2.add(t)
                all_tags.append(t)

        return "\n".join(all_tags)

    @staticmethod
    def extract_llm_content_and_postprocess(full_text: str, force_lowercase: bool = True) -> str:
        pattern = r"===處理結果開始===(.*?)===處理結果結束==="
        match = re.search(pattern, full_text, re.DOTALL)
        if not match:
            return ""

        content = match.group(1).strip()
        lines = [l.rstrip() for l in content.splitlines() if l.strip()]

        out_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("(") or s.startswith("（"):
                out_lines.append(s)
                continue

            s = s.replace(", ", " ")
            s = s.rstrip(".").strip()
            if force_lowercase:
                s = s.lower()
            out_lines.append(s)

        return "\n".join(out_lines)

    def on_llm_finished_latest_only(self, full_text):
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText("Run LLM")

        content = self.extract_llm_content_and_postprocess(full_text, self.english_force_lowercase)
        if not content:
            self.statusBar().showMessage("LLM Parser: 找不到 === markers 或內容為空", 5000)
            print(full_text)
            return

        self.save_nl_for_image(self.current_image_path, content)

        if not self.nl_pages:
            self.nl_pages = []
        self.nl_pages.append(content)
        self.nl_page_index = len(self.nl_pages) - 1
        self.nl_latest = content
        self.refresh_nl_tab()
        self.update_nl_page_controls()
        self.on_text_changed()

    def on_llm_error_no_popup(self, err):
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText("Run LLM")
        self.statusBar().showMessage(f"LLM Error: {err}", 8000)

    # ==========================
    # Tools: Unmask (Remove BG) / Stroke Eraser
    # ==========================
    def _qimage_to_pil_l(qimg: QImage) -> Image.Image:
        # 轉換格式
        q = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
        
        # 使用 QBuffer 將 QImage 存為 PNG 格式的 Bytes
        # 這樣可以由 Qt 自動處理所有的 Stride/Padding/Alignment 問題
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        q.save(buf, "PNG")
        
        # 使用 PIL 從記憶體中讀取
        return Image.open(BytesIO(ba.data()))

    def on_batch_tagger_done(self):
        self.btn_batch_tagger.setEnabled(True)
        self.btn_batch_tagger_to_txt.setEnabled(True)
        self.btn_auto_tag.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage("Batch Tagger 完成", 5000)

    

    def on_batch_llm_done(self):
        self.btn_batch_llm.setEnabled(True)
        self.btn_batch_llm_to_txt.setEnabled(True)
        self.btn_run_llm.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage("Batch LLM 完成", 5000)

    def _image_has_background_tag(self, image_path: str) -> bool:
        """
        判斷是否為「背景圖」。
        檢查 .txt 分類以及 sidecar JSON 標籤。
        """
        try:
            # 1. 優先檢查 Sidecar JSON (如果有 Tagger 結果就抓得到)
            sidecar = load_image_sidecar(image_path)
            # 檢查 tagger_tags (標籤器結果) 或 tags_context (原本的標籤)
            tags_all = (sidecar.get("tagger_tags", "") + " " + sidecar.get("tags_context", "")).lower()
            if "background" in tags_all:
                return True

            # 2. 檢查 .txt 檔案 (後備方案)
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                return "background" in content
        except Exception:
            pass
        return False

    def on_batch_restore_per_image(self, old_path, new_path):
        if old_path != new_path:
            self._replace_image_path_in_list(old_path, new_path)
            # Reload if current image is affected
            if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(new_path):
                self.load_image()

    def on_reset_prompt(self):
        """Reset prompt editor to the currently active template."""
        try:
            # Re-fetch from app_settings to ensure latest
            self.default_user_prompt_template = self.app_settings.user_prompt_template
            self.prompt_edit.setPlainText(self.default_user_prompt_template)
            self.statusBar().showMessage(self.tr("msg_prompt_reset"), 2000)
        except Exception:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())