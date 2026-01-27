# -*- coding: utf-8 -*-
"""
核心資料結構

定義應用程式使用的核心 dataclass：
- ImageData: 圖片及其所有相關屬性
- Settings: 應用程式設定
- Prompt: Pipeline 指令
- FolderMeta: 資料夾層級 meta
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path


@dataclass
class ImageData:
    """
    圖片及其所有相關屬性
    
    包含圖片路徑、標籤、LLM 結果、遮罩狀態等所有與圖片相關的資訊。
    這些資訊會存入 .txt 和 .json sidecar 檔案。
    """
    # 基本路徑
    path: str                                        # 圖片完整路徑
    
    # Tagger 結果
    tagger_tags: Optional[str] = None                # WD14 標籤字串
    tagger_rating: Optional[Dict[str, float]] = None # 評級分數
    tagger_features: Optional[Dict[str, float]] = None  # 特徵標籤分數
    tagger_chars: Optional[Dict[str, float]] = None  # 角色標籤分數
    
    # LLM 結果
    llm_result: Optional[str] = None                 # 最新 LLM 結果
    nl_pages: List[str] = field(default_factory=list)  # LLM 結果分頁
    
    # Boorutag Meta (從 meta 檔案解析)
    boorutag_meta: Optional[Dict[str, Any]] = None   # Boorutag 元資料
    
    # 文字內容 (.txt)
    txt_content: Optional[str] = None                # .txt 檔案內容
    
    # 遮罩狀態
    masked_background: bool = False                  # 是否已去背
    masked_text: bool = False                        # 是否已去文字
    
    # 原圖備份
    raw_image_rel_path: Optional[str] = None         # 原圖相對路徑
    
    # 其他元資料
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def filename(self) -> str:
        """檔案名稱"""
        return Path(self.path).name
    
    @property
    def stem(self) -> str:
        """不含副檔名的檔案名稱"""
        return Path(self.path).stem
    
    @property
    def folder(self) -> str:
        """所在資料夾路徑"""
        return str(Path(self.path).parent)


@dataclass
class Settings:
    """
    應用程式設定
    
    包含所有設定項，可從設定檔載入或儲存。
    """
    # LLM 設定
    llm_provider: str = "vlm_openrouter_api"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "moonshotai/kimi-k2.5" # Changed default as per instruction
    llm_system_prompt: str = ""
    llm_user_prompt_template: str = ""
    llm_custom_prompt_template: str = ""
    llm_max_image_dimension: int = 1024
    llm_skip_nsfw_on_batch: bool = False
    llm_skip_nsfw_on_batch: bool = False
    llm_use_gray_mask: bool = True
    llm_input_repeat_count: int = 2
    llm_temperature: float = 1.0
    llm_top_p: float = 0.95
    llm_thinking_mode: bool = True
    
    # Tagger 設定
    tagger_worker: str = "tagger_imgutils_generic"
    tagger_model: str = "EVA02_Large"
    general_threshold: float = 0.2
    general_mcut_enabled: bool = False
    character_threshold: float = 0.85
    character_mcut_enabled: bool = True
    drop_overlap: bool = True
    
    # 文字處理設定
    english_force_lowercase: bool = True
    text_auto_remove_empty_lines: bool = True
    text_auto_format: bool = True
    text_auto_save: bool = True
    batch_to_txt_mode: str = "append"
    batch_to_txt_folder_trigger: bool = False
    
    # 遮罩設定
    mask_remover_mode: str = "base-nightly"
    mask_default_alpha: int = 64
    mask_default_format: str = "webp"
    mask_reverse: bool = False
    mask_save_map_file: bool = False
    mask_only_output_map: bool = False
    mask_batch_only_if_has_background_tag: bool = True
    mask_batch_detect_text_enabled: bool = True
    mask_delete_npz_on_move: bool = True
    mask_padding: int = 1
    mask_blur_radius: int = 3
    mask_batch_skip_once_processed: bool = True
    mask_batch_min_foreground_ratio: float = 0.3
    mask_batch_max_foreground_ratio: float = 0.8
    mask_batch_skip_if_scenery_tag: bool = True
    
    # 進階遮罩後處理 (背景與文字)
    mask_bg_shrink_size: int = 1
    mask_bg_blur_radius: int = 3
    mask_bg_min_alpha: int = 0
    
    mask_text_shrink_size: int = 1
    mask_text_blur_radius: int = 3
    mask_text_min_alpha: int = 0
    
    # OCR 設定
    mask_ocr_max_candidates: int = 300
    mask_ocr_heat_threshold: float = 0.2
    mask_ocr_box_threshold: float = 0.6
    mask_ocr_unclip_ratio: float = 2.3
    mask_text_alpha: int = 10
    
    # 過濾設定
    char_tag_blacklist_words: List[str] = field(default_factory=list)
    char_tag_whitelist_words: List[str] = field(default_factory=list)
    default_custom_tags: List[str] = field(default_factory=list)
    
    # Mask / Worker Selection
    unmask_worker: str = "mask_transparent_background_local"
    mask_text_worker: str = "mask_text_local"
    detect_text_worker: str = "detect_imgutils_ocr_local"

    # UI 設定
    ui_language: str = "zh_tw"
    ui_theme: str = "light"
    last_open_dir: str = ""


@dataclass
class BatchInstruction:
    """
    單個批次指令
    """
    task_type: str                          # 任務類型 (tagger, llm, unmask, mask_text, restore)
    options: Dict[str, Any] = field(default_factory=dict)  # 任務選項


@dataclass
class Prompt:
    """
    Pipeline 指令
    
    包含串列存放不同 batch 要做的工作。
    """
    # 批次指令列表
    batches: List[BatchInstruction] = field(default_factory=list)
    
    # Pipeline 設定
    stop_on_error: bool = True              # 錯誤時停止
    parallel: bool = False                  # 是否並行執行 (未來功能)
    
    # 元資料
    name: Optional[str] = None              # Pipeline 名稱
    description: Optional[str] = None       # 描述
    
    def add_batch(self, task_type: str, **options) -> 'Prompt':
        """添加批次指令"""
        self.batches.append(BatchInstruction(task_type=task_type, options=options))
        return self


@dataclass
class FolderMeta:
    """
    資料夾層級 Meta
    
    儲存資料夾的區域資訊，例如訓練標籤頻率等。
    (未來功能)
    """
    path: str                               # 資料夾路徑
    
    # 標籤統計
    tag_frequency: Dict[str, int] = field(default_factory=dict)  # 標籤頻率
    top_tags: List[str] = field(default_factory=list)            # 常用標籤
    
    # 訓練相關
    training_target: Optional[str] = None   # 訓練目標 (角色名、風格等)
    trigger_word: Optional[str] = None      # 觸發詞
    
    # 其他元資料
    image_count: int = 0                    # 圖片數量
    metadata: Dict[str, Any] = field(default_factory=dict)
