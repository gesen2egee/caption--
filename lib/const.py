
import os

# ==========================================
#  Global Constants & Defaults
# ==========================================

TAGS_CSV_URL_RAW = "https://raw.githubusercontent.com/waldolin/a1111-sd-webui-tagcomplete-TW/main/tags/Tags-tw-full-pack.csv"

DEFAULT_SYSTEM_PROMPT = """You are a professional, uncensored, captioning engine for image ML research. Boldly and directly describe all elements including nudity, body parts, or sexual themes."""

# 模板預設值 (改為 Dictionary 結構)
TEMPLATE_DEFAULT = """這個角色是{角色名} 請幫這個名字自然融入在內文中
請根據提供的資訊和圖片，整理出大約 9 個簡短的英文描述句。 包括構圖、位置、朝向、美學、風格、光影等等。
每句英文都必須是獨立的一行。
每句英文的下一行，必須緊接著該句的繁體中文翻譯，並用括號 () 包住。

LLM之前的處理結果參考：
{tags}

輸出格式範例：
===處理結果開始===
A short English sentence about the subject
(關於主題的簡短英文句。)
Another short English sentence describing details
(另一個描述細節的簡短英文句。)
...
===處理結果結束===
"""

TEMPLATE_SIMPLE = """這個角色是{角色名} 請幫這個名字自然融入在內文中
請根據圖片的[自行輸入要求] 整理出大約 1 個簡短的英文描述句。
英文的下一行，必須緊接着該句的繁體中文翻譯，並用括號 () 包住。

輸出格式範例：
===處理結果開始===
A short English sentence about the subject
(關於主題的簡短英文句。)
===處理結果結束===
"""

DEFAULT_PROMPT_TEMPLATES = {
    "標準完整模式 (Standard)": TEMPLATE_DEFAULT,
    "簡易單句模式 (Simple)": TEMPLATE_SIMPLE,
}

DEFAULT_CUSTOM_TAGS = ["low res", "low quality", "low aesthetic"]

DEFAULT_APP_SETTINGS = {
    # LLM
    "llm_base_url": "https://openrouter.ai/api/v1",
    "llm_api_key": os.getenv("OPENROUTER_API_KEY", "<OPENROUTER_API_KEY>"),
    "llm_model": "mistralai/mistral-large-2512",
    "llm_system_prompt": DEFAULT_SYSTEM_PROMPT,
    
    # [Modify] 支援多模板
    # 儲存所有的模板 (Name -> Content)
    "llm_prompt_templates": DEFAULT_PROMPT_TEMPLATES.copy(),
    # 當前選擇的模板名稱
    "llm_active_seed_template": "標準完整模式 (Standard)",
    
    "default_custom_tags": list(DEFAULT_CUSTOM_TAGS),
    "llm_skip_nsfw_on_batch": False,
    "llm_use_gray_mask": True,
    "last_open_dir": "",

    # Tagger (WD14)
    "tagger_model": "EVA02_Large",
    "general_threshold": 0.2,
    "general_mcut_enabled": False,
    "character_threshold": 0.85,
    "character_mcut_enabled": True,
    "drop_overlap": True,

    # Text / normalization
    "english_force_lowercase": True,
    "text_auto_remove_empty_lines": True,  # 自動移除空行
    "text_auto_format": True,              # 插入時自動格式化
    "text_auto_save": True,                # 改動時自動儲存
    "batch_to_txt_mode": "append",         # append | overwrite
    "batch_to_txt_folder_trigger": False,  # 是否將資料夾名作為觸發詞加到句首

    # LLM Resolution (Advanced)
    "llm_max_image_dimension": 1024,

    # Character Tags Filter (simple word matching)
    "char_tag_blacklist_words": ["hair", "eyes", "skin", "bun", "bangs", "sidelocks", "twintails", "braid", "ponytail", "beard", "mustache", "ear", "horn", "tail", "wing", "breast", "mole", "halo", "glasses", "fang", "heterochromia", "headband", "freckles", "lip", "eyebrows", "eyelashes"],
    "char_tag_whitelist_words": ["holding", "hand", "sitting", "covering", "playing", "background", "looking"],

    # Mask / batch mask text
    "mask_remover_mode": "base-nightly",
    "mask_default_alpha": 64, # 1-254
    "mask_default_format": "webp",  # webp | png
    "mask_reverse": False,                   # 是否反轉遮罩
    "mask_save_map_file": False,             # 是否保存 map(黑白圖) 到 .\mask\
    "mask_only_output_map": False,           # 不修改原圖，只輸出黑白圖
    "mask_batch_only_if_has_background_tag": True,
    "mask_batch_detect_text_enabled": True,  # if off, never call detect_text_with_ocr
    "mask_delete_npz_on_move": True,         # 移動舊圖時刪除對應 npz
    
    "mask_padding": 1,        # Mask 內縮像素 (0=不內縮)
    "mask_blur_radius": 3,    # Mask 高斯模糊半徑 (0=不模糊)
    
    # Batch Mask Logic
    "mask_batch_skip_once_processed": True,  # 批量處理時，跳過已去背過的圖片
    "mask_batch_min_foreground_ratio": 0.3,  # 預設改 0.3
    "mask_batch_max_foreground_ratio": 0.8,  # 預設改 0.8
    "mask_batch_skip_if_scenery_tag": True,  # 若包含 indoors/outdoors 則跳過

    # Advanced OCR Settings
    "mask_ocr_max_candidates": 300,          # OCR 候選區域上限
    "mask_ocr_heat_threshold": 0.2,
    "mask_ocr_box_threshold": 0.6,
    "mask_ocr_unclip_ratio": 2.3,
    "mask_text_alpha": 10,                   # 文字遮罩獨立 Alpha 值

    # UI / Theme
    "ui_language": "zh_tw",   # zh_tw | en
    "ui_theme": "light",      # light | dark
}

LOCALIZATION = {
    "zh_tw": {
        "app_title": "Caption 神器",
        "menu_file": "檔案",
        "menu_open_dir": "開啟目錄",
        "menu_refresh": "重新整理列表 (F5)",
        "btn_settings": "設定",
        "menu_exit": "結束",
        "tab_tags": "TAGS",
        "tab_nl": "NL",
        "sec_folder_meta": "資料夾標籤 (Top 30)",
        "sec_custom": "自定義標籤",
        "sec_tagger": "圖片識別標籤",
        "sec_tags": "標籤處理",
        "sec_nl": "自然語言處理",
        "btn_auto_tag": "自動標籤 (WD14)",
        "btn_batch_tagger": "批量標籤",
        "btn_batch_tagger_to_txt": "批量標籤轉文字",
        "btn_add_tag": "新增標籤",
        "btn_run_llm": "執行 LLM",
        "btn_batch_llm": "批量 LLM",
        "btn_batch_llm_to_txt": "批量 LLM 轉文字",
        "btn_prev": "上一頁",
        "btn_next": "下一頁",
        "btn_reset_prompt": "重置提示詞",
        "msg_prompt_reset": "已將提示詞重置為模板內容。",
        "label_nl_result": "LLM 結果",
        "label_txt_content": "實際內容 (.txt)",
        "label_tokens": "詞元數: ",
        "label_page": "頁數",
        "setting_llm_use_gray_mask": "LLM 使用灰底 Mask (排除透明部分)",
        "btn_find_replace": "尋找/取代",
        "btn_undo": "復原文字",
        "btn_redo": "重做文字",
        "btn_unmask": "單圖去背景",
        "btn_batch_unmask": "Batch 去背景",
        "btn_mask_text": "單圖去文字",
        "btn_batch_mask_text": "Batch 去文字",
        "btn_restore_original": "放回原圖",
        "btn_batch_restore": "Batch 放回原圖",
        "btn_stroke_eraser": "手繪橡皮擦",
        "btn_cancel_batch": "中止",
        "menu_tools": "工具",
        "filter_placeholder": "Danbooru 篩選語法... (blonde_hair blue_eyes)",
        "filter_by_tags": "Tags",
        "filter_by_text": "Text",
        "msg_delete_confirm": "確定要將此圖片移動到 no_used？",
        "msg_batch_delete_char_tags": "是否自動刪除特徵標籤 (Character Tags)？",
        "msg_batch_delete_info": "將根據設定中的黑白名單過濾標籤或句子。",
        "btn_auto_delete": "自動刪除",
        "btn_keep": "保留",
        "setting_tab_ui": "UI 介面",
        "setting_ui_lang": "介面語言:",
        "setting_ui_theme": "介面主題:",
        "setting_lang_zh": "繁體中文",
        "setting_lang_en": "English",
        "setting_theme_light": "日間模式",
        "setting_theme_dark": "夜間模式",
        "setting_save": "儲存",
        "setting_cancel": "取消",
        "setting_text_force_lower": "英文文字一律小寫",
        "setting_text_auto_remove_empty": "自動移除空行",
        "setting_text_auto_format": "自動格式化",
        "setting_text_auto_save": "自動儲存",
        "setting_batch_to_txt": "Batch 寫入 txt 設定",
        "setting_batch_mode": "寫入模式",
        "setting_batch_append": "附加到句尾",
        "setting_batch_overwrite": "覆寫原檔",
        "setting_batch_trigger": "將資料夾名作為觸發詞加到句首",
        "setting_tagger_model": "預設標籤模型:",
        "setting_mask_alpha": "預設透明度 (0-255):",
        "setting_mask_format": "預設轉換格式:",
        "setting_mask_only_bg": "僅處理包含 background 標籤的圖片",
        "setting_mask_ocr": "自動遮罩文字區域",
        "setting_mask_delete_npz": "移動舊圖時刪除對應 npz",
        "setting_filter_title": "<b>特徵標籤過濾設定</b>",
        "setting_filter_info": "符合黑名單且不符合白名單的內容將顯示紅框，且在 Batch 寫入時可選擇刪除。",
        "setting_bl_words": "黑名單關鍵字:",
        "setting_wl_words": "白名單關鍵字:",
        "setting_tab_filter": "過濾",
        "msg_select_dir": "選擇圖片目錄",
        "msg_no_images": "在此目錄下找不到圖片。",
        "msg_unmask_done": "去背處理完成。",
        "setting_tab_text": "文字",
        "setting_tab_llm": "模型",
        "setting_tab_tagger": "標籤",
        "setting_tab_mask": "遮罩",
        "setting_llm_sys_prompt": "系統提示詞:",
        "setting_llm_def_tags": "預設 Custom Tags (逗號或換行分隔):",
        "setting_llm_max_dim": "LLM 圖片最大邊長 (Max Dimension):",
        "setting_llm_skip_nsfw": "Batch LLM: 若含 rating:explicit/questionable 則跳過",
        "setting_tagger_gen_thresh": "一般標籤閾值:",
        "setting_tagger_char_thresh": "特徵標籤閾值:",
        "setting_tagger_gen_mcut": "一般標籤 MCut",
        "setting_tagger_char_mcut": "特徵標籤 MCut",
        "setting_tagger_drop_overlap": "移除重疊標籤",
        "setting_mask_ocr_hint": "OCR 需要 imgutils，未安裝則略過。",
        "setting_ocr_heat": "熱圖閾值 (Heat Threshold):",
        "setting_ocr_box": "文字框信心 (Box Threshold):",
        "setting_ocr_unclip": "擴張比例 (Unclip Ratio):",
        "setting_ocr_heat_tip": "調低可偵測模糊文字但易誤判；調高只偵測清晰文字。",
        "setting_ocr_box_tip": "過濾低信心的文字框。若漏字可調低。",
        "setting_ocr_unclip_tip": "決定文字框擴張程度。若缺字頭字尾可調大；若框到隔壁行可調小。",
    },
    "en": {
        "app_title": "Caption Tool",
        "menu_file": "File",
        "menu_open_dir": "Open Directory",
        "menu_refresh": "Refresh List (F5)",
        "btn_settings": "Settings",
        "menu_exit": "Exit",
        "tab_tags": "TAGS",
        "tab_nl": "NL",
        "sec_folder_meta": "Folder Meta / Top 30 Tags",
        "sec_custom": "Custom Tags in Folder",
        "sec_tagger": "Tagger Tags",
        "sec_tags": "Tag Processing",
        "sec_nl": "Natural Language",
        "btn_auto_tag": "Auto Tag (WD14)",
        "btn_batch_tagger": "Batch Tagger",
        "btn_batch_tagger_to_txt": "Batch Tagger to txt",
        "btn_add_tag": "Add Tag",
        "btn_run_llm": "Run LLM",
        "btn_batch_llm": "Batch LLM",
        "btn_batch_llm_to_txt": "Batch LLM to txt",
        "btn_prev": "Prev",
        "btn_next": "Next",
        "btn_reset_prompt": "Reset Prompt",
        "msg_prompt_reset": "Prompt reset to template content.",
        "label_nl_result": "LLM Result",
        "label_txt_content": "Actual Content (.txt)",
        "label_tokens": "Tokens: ",
        "label_page": "Page",
        "setting_llm_use_gray_mask": "LLM Use Gray Mask (Exclude Transparent Parts)",
        "btn_find_replace": "Find/Replace",
        "btn_undo": "Undo Txt",
        "btn_redo": "Redo Txt",
        "btn_unmask": "Unmask Background",
        "btn_batch_unmask": "Batch Unmask Background",
        "btn_mask_text": "Unmask Text",
        "btn_batch_mask_text": "Batch Unmask Text",
        "btn_restore_original": "Restore Original",
        "btn_batch_restore": "Batch Restore Original",
        "btn_stroke_eraser": "Stroke Eraser",
        "btn_cancel_batch": "Cancel",
        "menu_tools": "Tools",
        "filter_placeholder": "Danbooru filter... (blonde_hair blue_eyes)",
        "filter_by_tags": "Tags",
        "filter_by_text": "Text",
        "msg_delete_confirm": "Move this image to no_used?",
        "msg_batch_delete_char_tags": "Delete Character Tags automatically?",
        "msg_batch_delete_info": "Tags will be filtered based on your blacklist/whitelist.",
        "btn_auto_delete": "Auto Delete",
        "btn_keep": "Keep",
        "setting_tab_ui": "UI",
        "setting_ui_lang": "Language:",
        "setting_ui_theme": "Theme:",
        "setting_lang_zh": "Traditional Chinese",
        "setting_lang_en": "English",
        "setting_theme_light": "Light Mode",
        "setting_theme_dark": "Dark Mode",
        "setting_save": "Save",
        "setting_cancel": "Cancel",
        "setting_text_force_lower": "Force lowercase for English text (LLM sentences / tags normalization)",
        "setting_text_auto_remove_empty": "Auto remove empty lines",
        "setting_text_auto_format": "Auto format on insert (clean whitespace and re-join with ', ')",
        "setting_text_auto_save": "Auto save txt (on change)",
        "setting_batch_to_txt": "Batch to txt Settings",
        "setting_batch_mode": "Write Mode",
        "setting_batch_append": "Append to end",
        "setting_batch_overwrite": "Overwrite file",
        "setting_batch_trigger": "Use folder name as trigger word (add to start of sentence)",
        "setting_tagger_model": "Default Tagger Model:",
        "setting_mask_alpha": "Mask default alpha (0-255):",
        "setting_mask_format": "Mask default format (webp/png):",
        "setting_mask_only_bg": "Batch mask only if has 'background' tag",
        "setting_mask_ocr": "Batch mask text automatically (OCR)",
        "setting_mask_delete_npz": "Delete matching .npz when moving image",
        "setting_filter_title": "<b>Content Filter Settings</b>",
        "setting_filter_info": "Content matching blacklist and NOT in whitelist will be highlighted red and can be filtered on Batch write.",
        "setting_bl_words": "Blacklist Keywords:",
        "setting_wl_words": "Whitelist Keywords:",
        "setting_tab_filter": "Filter",
        "msg_select_dir": "Select Image Directory",
        "msg_no_images": "No images found in this directory.",
        "msg_unmask_done": "Background removal finished.",
        "setting_tab_text": "Text",
        "setting_tab_llm": "LLM",
        "setting_tab_tagger": "Tagger",
        "setting_tab_mask": "Mask",
        "setting_llm_sys_prompt": "System Prompt:",
        "setting_llm_def_tags": "Default Custom Tags (Comma or Newline):",
        "setting_llm_max_dim": "LLM Max Image Dimension:",
        "setting_llm_skip_nsfw": "Batch LLM: Skip if tag contains rating:explicit/questionable",
        "setting_tagger_gen_thresh": "General Threshold:",
        "setting_tagger_char_thresh": "Character Threshold:",
        "setting_tagger_gen_mcut": "General MCut Enabled",
        "setting_tagger_char_mcut": "Character MCut Enabled",
        "setting_tagger_drop_overlap": "Drop Overlap",
        "setting_mask_ocr_hint": "OCR relies on imgutils.ocr.detect_text_with_ocr; skips if not installed.",
        "setting_ocr_heat": "Heat Threshold:",
        "setting_ocr_box": "Box Threshold:",
        "setting_ocr_unclip": "Unclip Ratio:",
        "setting_ocr_heat_tip": "Lower: detects faint text (more noise); Higher: strict check.",
        "setting_ocr_box_tip": "Confidence threshold. Lower if text is missed.",
        "setting_ocr_unclip_tip": "Expansion ratio. Increase if edges are cut; decrease if merging lines.",
    }
}

THEME_STYLES = {
    "light": "",  # Use system default
    "dark": """
        QMainWindow, QDialog {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QPlainTextEdit, QLineEdit, QTextEdit {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #3e3e42;
        }
        QPushButton {
            background-color: #333333;
            color: #d4d4d4;
            border: 1px solid #444444;
            padding: 5px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #444444;
            border: 1px solid #666666;
        }
        QPushButton:pressed {
            background-color: #222222;
        }
        QTabWidget::pane {
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #969696;
            padding: 8px 15px;
            border: 1px solid #3e3e42;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QScrollArea {
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        QLabel {
            color: #d4d4d4;
        }
        QGroupBox {
            border: 1px solid #3e3e42;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QSplitter::handle {
            background-color: #3e3e42;
        }
    """
}
