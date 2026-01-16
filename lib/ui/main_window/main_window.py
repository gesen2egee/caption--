"""
Main Window Module

這是應用程式的主視窗類別，通過組合多個 Mixin 來實現完整功能。
"""
import os
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QApplication
from PyQt6.QtCore import QTimer

# Import Config & Utils
from lib.data import AppSettings, load_app_settings
from lib.const import DEFAULT_APP_SETTINGS, DEFAULT_CUSTOM_TAGS
from lib.utils import load_translations

# Import Mixins
from lib.ui.main_window.mixins import (
    ShortcutsMixin, ThemeMixin, NLMixin, DialogsMixin, ProgressMixin,
    FileMixin, FilterMixin, NavigationMixin, TextEditMixin, TagsMixin,
    ImageMixin, BatchBaseMixin, VisionMixin, TaggerMixin, LLMMixin,
    BatchExportMixin, AppCoreMixin
)

class MainWindow(ShortcutsMixin, ThemeMixin, NLMixin, DialogsMixin, ProgressMixin, 
                 FileMixin, FilterMixin, NavigationMixin, TextEditMixin, TagsMixin,
                 ImageMixin, BatchBaseMixin, VisionMixin, TaggerMixin, LLMMixin, 
                 BatchExportMixin, AppCoreMixin, QMainWindow):
    """
    Caption 神器主視窗
    
    繼承順序決定了 Mixin 的覆蓋順序。AppCoreMixin 提供了基礎 UI 初始化。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Captioning Assistant")
        self._clip_tokenizer = None
        self.resize(1600, 1000)

        # 載入設定
        self.settings = load_app_settings()
        self.app_settings = AppSettings(self.settings)

        # 初始化 LLM 參數 (Legacy Support)
        self.llm_base_url = str(self.settings.get("llm_base_url", DEFAULT_APP_SETTINGS["llm_base_url"]))
        self.api_key = str(self.settings.get("llm_api_key", DEFAULT_APP_SETTINGS["llm_api_key"]))
        self.model_name = str(self.settings.get("llm_model", DEFAULT_APP_SETTINGS["llm_model"]))
        self.llm_system_prompt = str(self.settings.get("llm_system_prompt", DEFAULT_APP_SETTINGS["llm_system_prompt"]))
        
        self.default_user_prompt_template = self.app_settings.user_prompt_template
        self.current_prompt_mode = "default"
        self.default_custom_tags_global = list(self.settings.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS)))
        self.english_force_lowercase = bool(self.settings.get("english_force_lowercase", True))

        # View Mode
        self.current_view_mode = 0  # 0=Original, 1=RGB, 2=Alpha
        self.temp_view_mode = None  # For N/M keys override
        
        self.translations_csv = {} 

        # State Variables
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
        self.all_image_files = [] 

        self.nl_pages = []
        self.nl_page_index = 0
        self.nl_latest = ""

        # Thread References
        self.batch_tagger_thread = None
        self.batch_llm_thread = None
        self.batch_unmask_thread = None
        self.batch_mask_text_thread = None
        self.tagger_thread = None
        self.llm_thread = None
        self.batch_restore_thread = None

        # 載入標籤翻譯（在 UI 初始化前載入，確保翻譯可用）
        from lib.utils import load_translations
        try:
            self.translations_csv = load_translations()
        except Exception as e:
            print(f"[Warning] 載入標籤翻譯失敗: {e}")
            self.translations_csv = {}

        # UI Init
        self.setup_ui_components() # From AppCoreMixin
            
        self._hf_tokenizer = None

        # Auto-load last directory
        last_dir = self.settings.get("last_open_dir", "")
        if last_dir and os.path.exists(last_dir):
            self.root_dir_path = last_dir
            # Delay loading to prevent freeze on startup
            QTimer.singleShot(100, self.refresh_file_list) 

        # Lazy Init Models (background) to prevent startup freeze
        QTimer.singleShot(500, self._lazy_init_models)

    def _lazy_init_models(self):
        """非同步初始化模型，避免卡死介面"""
        # Check CUDA
        try:
            import torch
            if not torch.cuda.is_available():
                QMessageBox.warning(
                    self, 
                    "CUDA Warning", 
                    "偵測不到 NVIDIA GPU (CUDA)。\n\n這可能是因為 venv 中的 PyTorch 版本錯誤。\n請執行根目錄下的 'fix_torch_gpu.bat' 來修復。\n\n目前將使用 CPU 執行，速度會非常慢。"
                )
        except ImportError:
            pass

        # Pre-load Tokenizer (This might take time, but now it's AFTER UI displays)
        if hasattr(self, '_get_clip_tokenizer'):
            self.statusBar().showMessage("Loading Clip Tokenizer...")
            QApplication.processEvents()
            self._get_clip_tokenizer()
            self.statusBar().showMessage("Ready", 3000)
