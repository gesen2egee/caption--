# -*- coding: utf-8 -*-
"""
設定對話框 (SettingsDialog)
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout, 
    QComboBox, QRadioButton, QHBoxLayout, QLineEdit, QSpinBox, 
    QCheckBox, QLabel, QPlainTextEdit, QGroupBox, QPushButton, 
    QFrame, QDoubleSpinBox
)
from lib.locales import load_locale, tr as locale_tr
from lib.core.settings import (
    DEFAULT_APP_SETTINGS, DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_CUSTOM_TAGS, _coerce_float, _coerce_int
)


class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = dict(cfg or {})
        self.setWindowTitle(self.tr("btn_settings"))
        self.setMinimumWidth(640)

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs, 1)
        
        # 初始化 UI
        self._init_ui()

    def tr(self, key: str) -> str:
        lang = self.cfg.get("ui_language", "zh_tw")
        load_locale(lang)
        return locale_tr(key)

    def _init_ui(self):
        # ---- UI ----
        tab_ui = QWidget()
        ui_layout = QVBoxLayout(tab_ui)
        ui_form = QFormLayout()
        
        self.cb_lang = QComboBox()
        self.cb_lang.addItem(self.tr("setting_lang_zh"), "zh_tw")
        self.cb_lang.addItem(self.tr("setting_lang_en"), "en")
        idx_lang = self.cb_lang.findData(self.cfg.get("ui_language", "zh_tw"))
        self.cb_lang.setCurrentIndex(idx_lang if idx_lang >= 0 else 0)
        
        ui_form.addRow(self.tr("setting_ui_lang"), self.cb_lang)
        
        self.rb_light = QRadioButton(self.tr("setting_theme_light"))
        self.rb_dark = QRadioButton(self.tr("setting_theme_dark"))
        if self.cfg.get("ui_theme", "light") == "dark":
            self.rb_dark.setChecked(True)
        else:
            self.rb_light.setChecked(True)
        
        ui_theme_lay = QHBoxLayout()
        ui_theme_lay.addWidget(self.rb_light)
        ui_theme_lay.addWidget(self.rb_dark)
        ui_form.addRow(self.tr("setting_ui_theme"), ui_theme_lay)
        
        ui_layout.addLayout(ui_form)
        ui_layout.addStretch(1)
        self.tabs.addTab(tab_ui, self.tr("setting_tab_ui"))

        # ---- LLM ----
        tab_llm = QWidget()
        llm_layout = QVBoxLayout(tab_llm)
        form = QFormLayout()

        self.ed_base_url = QLineEdit(str(self.cfg.get("llm_base_url", "")))
        self.ed_api_key = QLineEdit(str(self.cfg.get("llm_api_key", "")))
        self.ed_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_model = QLineEdit(str(self.cfg.get("llm_model", "")))

        form.addRow("LLM Base URL:", self.ed_base_url)
        form.addRow("API Key:", self.ed_api_key)
        # Note: Previous code had duplicate API Key row, removing one.
        form.addRow("Model:", self.ed_model)
        
        self.spin_llm_dim = QSpinBox()
        self.spin_llm_dim.setRange(256, 4096)
        self.spin_llm_dim.setSingleStep(128)
        self.spin_llm_dim.setValue(int(self.cfg.get("llm_max_image_dimension", 1024)))
        self.spin_llm_dim.setToolTip("傳給 LLM 的圖片最大邊長。\n調大：細節更多但 API 費用較高、速度較慢\n調小：處理更快且省費用，但可能遺漏細節")
        form.addRow(self.tr("setting_llm_max_dim"), self.spin_llm_dim)
        
        self.chk_llm_skip_nsfw = QCheckBox(self.tr("setting_llm_skip_nsfw"))
        self.chk_llm_skip_nsfw.setChecked(bool(self.cfg.get("llm_skip_nsfw_on_batch", False)))
        self.chk_llm_skip_nsfw.setToolTip("勾選後，批量 LLM 會自動跳過含 explicit/questionable 標籤的圖片")
        form.addRow("", self.chk_llm_skip_nsfw)
        
        self.chk_llm_use_gray_mask = QCheckBox(self.tr("setting_llm_use_gray_mask"))
        self.chk_llm_use_gray_mask.setChecked(bool(self.cfg.get("llm_use_gray_mask", True)))
        self.chk_llm_use_gray_mask.setToolTip("勾選後，去背後的透明區域會填灰色再傳給 LLM，\n讓 AI 專注描述主體而不是背景")
        form.addRow("", self.chk_llm_use_gray_mask)

        llm_layout.addLayout(form)

        llm_layout.addWidget(QLabel(self.tr("setting_llm_sys_prompt")))
        self.ed_system_prompt = QPlainTextEdit()
        self.ed_system_prompt.setPlainText(str(self.cfg.get("llm_system_prompt", DEFAULT_SYSTEM_PROMPT)))
        self.ed_system_prompt.setMinimumHeight(90)
        llm_layout.addWidget(self.ed_system_prompt)

        llm_layout.addWidget(QLabel(self.tr("setting_llm_def_prompt")))
        self.ed_user_template = QPlainTextEdit()
        self.ed_user_template.setPlainText(str(self.cfg.get("llm_user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE)))
        self.ed_user_template.setMinimumHeight(200)
        llm_layout.addWidget(self.ed_user_template, 1)


        llm_layout.addWidget(QLabel(self.tr("setting_llm_cust_prompt")))
        self.ed_custom_template = QPlainTextEdit()
        self.ed_custom_template.setPlainText(str(self.cfg.get("llm_custom_prompt_template", DEFAULT_CUSTOM_PROMPT_TEMPLATE)))
        self.ed_custom_template.setMinimumHeight(200)
        llm_layout.addWidget(self.ed_custom_template, 1)

        llm_layout.addWidget(QLabel(self.tr("setting_llm_def_tags")))
        self.ed_default_custom_tags = QPlainTextEdit()
        tags = self.cfg.get("default_custom_tags", list(DEFAULT_CUSTOM_TAGS))
        if isinstance(tags, list):
            self.ed_default_custom_tags.setPlainText("\n".join([str(t) for t in tags]))
        else:
            self.ed_default_custom_tags.setPlainText(str(tags))
        self.ed_default_custom_tags.setMinimumHeight(80)
        llm_layout.addWidget(self.ed_default_custom_tags)

        self.tabs.addTab(tab_llm, self.tr("setting_tab_llm"))

        # ---- Tagger ----
        tab_tagger = QWidget()
        tagger_layout = QVBoxLayout(tab_tagger)
        form2 = QFormLayout()
        self.ed_tagger_model = QLineEdit(str(self.cfg.get("tagger_model", "EVA02_Large")))
        self.ed_tagger_model.setToolTip("WD14 標籤模型名稱。常用: EVA02_Large (最準)、SwinV2 (較快)")
        
        self.ed_general_threshold = QLineEdit(str(self.cfg.get("general_threshold", 0.2)))
        self.ed_general_threshold.setToolTip("一般標籤的信心閾值 (0.0~1.0)\n調低：標籤更多但可能有誤判\n調高：標籤更精準但可能遺漏")
        
        self.chk_general_mcut = QCheckBox(self.tr("setting_tagger_gen_mcut"))
        self.chk_general_mcut.setChecked(bool(self.cfg.get("general_mcut_enabled", False)))
        self.chk_general_mcut.setToolTip("啟用 MCut 演算法自動決定閾值，會覆蓋上方的手動閾值設定")

        self.ed_character_threshold = QLineEdit(str(self.cfg.get("character_threshold", 0.85)))
        self.ed_character_threshold.setToolTip("角色/特徵標籤的信心閾值 (0.0~1.0)\n建議設較高 (0.8+) 以避免誤判角色")
        
        self.chk_character_mcut = QCheckBox(self.tr("setting_tagger_char_mcut"))
        self.chk_character_mcut.setChecked(bool(self.cfg.get("character_mcut_enabled", True)))
        self.chk_character_mcut.setToolTip("啟用 MCut 演算法自動決定閾值，會覆蓋上方的手動閾值設定")

        self.chk_drop_overlap = QCheckBox(self.tr("setting_tagger_drop_overlap"))
        self.chk_drop_overlap.setChecked(bool(self.cfg.get("drop_overlap", True)))
        self.chk_drop_overlap.setToolTip("移除重疊標籤，例如同時有 'long hair' 和 'hair' 時只保留更具體的")

        form2.addRow(self.tr("setting_tagger_model"), self.ed_tagger_model)
        form2.addRow(self.tr("setting_tagger_gen_thresh"), self.ed_general_threshold)
        form2.addRow("", self.chk_general_mcut)
        form2.addRow(self.tr("setting_tagger_char_thresh"), self.ed_character_threshold)
        form2.addRow("", self.chk_character_mcut)
        form2.addRow("", self.chk_drop_overlap)

        tagger_layout.addLayout(form2)
        tagger_layout.addStretch(1)
        self.tabs.addTab(tab_tagger, self.tr("setting_tab_tagger"))

        # ---- Text ----
        tab_text = QWidget()
        text_layout = QVBoxLayout(tab_text)
        self.chk_force_lower = QCheckBox(self.tr("setting_text_force_lower"))
        self.chk_force_lower.setChecked(bool(self.cfg.get("english_force_lowercase", True)))
        self.chk_force_lower.setToolTip("勾選後，所有英文標籤和句子會自動轉為小寫，\n符合 Stable Diffusion 訓練資料的常見格式")
        text_layout.addWidget(self.chk_force_lower)

        self.chk_auto_remove_empty = QCheckBox(self.tr("setting_text_auto_remove_empty"))
        self.chk_auto_remove_empty.setChecked(bool(self.cfg.get("text_auto_remove_empty_lines", True)))
        self.chk_auto_remove_empty.setToolTip("自動移除文字檔中的空白行，保持內容整潔")
        text_layout.addWidget(self.chk_auto_remove_empty)

        self.chk_auto_format = QCheckBox(self.tr("setting_text_auto_format"))
        self.chk_auto_format.setChecked(bool(self.cfg.get("text_auto_format", True)))
        self.chk_auto_format.setToolTip("自動整理標籤格式：移除多餘空格、統一用 ', ' 分隔")
        text_layout.addWidget(self.chk_auto_format)

        self.chk_auto_save = QCheckBox(self.tr("setting_text_auto_save"))
        self.chk_auto_save.setChecked(bool(self.cfg.get("text_auto_save", True)))
        self.chk_auto_save.setToolTip("編輯內容時自動儲存到 .txt 檔案，無需手動按儲存")
        text_layout.addWidget(self.chk_auto_save)

        # Batch to txt options
        text_layout.addWidget(self.make_hline())
        text_layout.addWidget(QLabel(f"<b>{self.tr('setting_batch_to_txt')}</b>"))
        
        mode_grp = QGroupBox(self.tr("setting_batch_mode"))
        mode_grp.setToolTip("決定批量處理時如何寫入 .txt 檔案")
        mode_lay = QHBoxLayout()
        self.rb_batch_append = QRadioButton(self.tr("setting_batch_append"))
        self.rb_batch_append.setToolTip("將新內容附加到現有文字的後面 (推薦)")
        self.rb_batch_overwrite = QRadioButton(self.tr("setting_batch_overwrite"))
        self.rb_batch_overwrite.setToolTip("完全覆蓋原有文字，請謹慎使用")
        if self.cfg.get("batch_to_txt_mode", "append") == "overwrite":
            self.rb_batch_overwrite.setChecked(True)
        else:
            self.rb_batch_append.setChecked(True)
        mode_lay.addWidget(self.rb_batch_append)
        mode_lay.addWidget(self.rb_batch_overwrite)
        mode_grp.setLayout(mode_lay)
        text_layout.addWidget(mode_grp)
        
        self.chk_folder_trigger = QCheckBox(self.tr("setting_batch_trigger"))
        self.chk_folder_trigger.setChecked(bool(self.cfg.get("batch_to_txt_folder_trigger", False)))
        self.chk_folder_trigger.setToolTip("勾選後，會把資料夾名稱當作觸發詞加到句子最前面\n例如資料夾 '1girl_miku' 會在開頭加上 'miku'")
        text_layout.addWidget(self.chk_folder_trigger)

        text_layout.addStretch(1)
        self.tabs.addTab(tab_text, self.tr("setting_tab_text"))

        # ---- Mask ----
        tab_mask = QWidget()
        mask_layout = QVBoxLayout(tab_mask)
        form3 = QFormLayout()

        self.ed_mask_alpha = QLineEdit(str(self.cfg.get("mask_default_alpha", 0)))
        self.ed_mask_alpha.setToolTip("去除部分的殘留透明度 (1-254)\n調低：去得更乾淨 (接近全透明)\n調高：保留更多半透明效果")
        self.ed_mask_format = QLineEdit(str(self.cfg.get("mask_default_format", "webp")))
        self.ed_mask_format.setToolTip("輸出格式：webp (檔案小) 或 png (相容性好)")
        form3.addRow(self.tr("setting_mask_alpha"), self.ed_mask_alpha)
        
        # New Settings
        self.spin_mask_padding = QSpinBox()
        self.spin_mask_padding.setRange(0, 50)
        self.spin_mask_padding.setValue(int(self.cfg.get("mask_padding", 3)))
        self.spin_mask_padding.setToolTip("主體邊緣內縮的像素數\n調大：邊緣更乾淨，但可能切到主體\n調小：保留更多邊緣細節")
        form3.addRow("Mask Padding (內縮像素):", self.spin_mask_padding)

        self.spin_mask_blur = QSpinBox()
        self.spin_mask_blur.setRange(0, 50)
        self.spin_mask_blur.setValue(int(self.cfg.get("mask_blur_radius", 10)))
        self.spin_mask_blur.setToolTip("邊緣模糊半徑 (高斯模糊)\n調大：邊緣更柔和自然\n調小：邊緣更銳利")
        form3.addRow("Mask Blur (模糊半徑):", self.spin_mask_blur)

        form3.addRow(self.tr("setting_mask_format"), self.ed_mask_format)

        self.chk_mask_bg_only = QCheckBox(self.tr("setting_mask_only_bg"))
        self.chk_mask_bg_only.setChecked(bool(self.cfg.get("mask_batch_only_if_has_background_tag", False)))
        self.chk_mask_bg_only.setToolTip("勾選後，批量去背只處理標籤含 'background' 的圖片\n避免誤處理不需要去背的圖")
        form3.addRow("", self.chk_mask_bg_only)

        self.chk_mask_ocr = QCheckBox(self.tr("setting_mask_ocr"))
        self.chk_mask_ocr.setChecked(bool(self.cfg.get("mask_batch_detect_text_enabled", True)))
        self.chk_mask_ocr.setToolTip("啟用 OCR 自動偵測並遮蔽圖片中的文字區域")
        form3.addRow("", self.chk_mask_ocr)

        # OCR Advanced
        self.spin_ocr_heat = QDoubleSpinBox()
        self.spin_ocr_heat.setRange(0.01, 1.0)
        self.spin_ocr_heat.setSingleStep(0.05)
        self.spin_ocr_heat.setValue(float(self.cfg.get("mask_ocr_heat_threshold", 0.2)))
        self.spin_ocr_heat.setToolTip(self.tr("setting_ocr_heat_tip"))
        form3.addRow(self.tr("setting_ocr_heat"), self.spin_ocr_heat)

        self.spin_ocr_box = QDoubleSpinBox()
        self.spin_ocr_box.setRange(0.01, 1.0)
        self.spin_ocr_box.setSingleStep(0.05)
        self.spin_ocr_box.setValue(float(self.cfg.get("mask_ocr_box_threshold", 0.6)))
        self.spin_ocr_box.setToolTip(self.tr("setting_ocr_box_tip"))
        form3.addRow(self.tr("setting_ocr_box"), self.spin_ocr_box)

        self.spin_ocr_unclip = QDoubleSpinBox()
        self.spin_ocr_unclip.setRange(1.0, 5.0)
        self.spin_ocr_unclip.setSingleStep(0.1)
        self.spin_ocr_unclip.setValue(float(self.cfg.get("mask_ocr_unclip_ratio", 2.3)))
        self.spin_ocr_unclip.setToolTip(self.tr("setting_ocr_unclip_tip"))
        form3.addRow(self.tr("setting_ocr_unclip"), self.spin_ocr_unclip)

        self.spin_mask_text_alpha = QSpinBox()
        self.spin_mask_text_alpha.setRange(0, 255)
        self.spin_mask_text_alpha.setValue(int(self.cfg.get("mask_text_alpha", 10)))
        self.spin_mask_text_alpha.setToolTip("僅針對『去文字』功能使用的 Alpha 透明度 (預設 10)\n設定越低，遮得越透明/乾淨")
        form3.addRow("Text Mask Alpha (去文字遮罩值):", self.spin_mask_text_alpha)

        self.chk_mask_del_npz = QCheckBox(self.tr("setting_mask_delete_npz"))
        self.chk_mask_del_npz.setChecked(bool(self.cfg.get("mask_delete_npz_on_move", True)))
        self.chk_mask_del_npz.setToolTip("移動原圖時自動刪除對應的 .npz 快取檔案 (SD 訓練用)")
        form3.addRow("", self.chk_mask_del_npz)

        self.chk_mask_reverse = QCheckBox("反轉遮罩 (Reverse)")
        self.chk_mask_reverse.setChecked(bool(self.cfg.get("mask_reverse", False)))
        self.chk_mask_reverse.setToolTip("將主體與背景反轉 (去主體留背景)")
        form3.addRow("", self.chk_mask_reverse)

        self.chk_save_map = QCheckBox("保存黑白 Mask 到 .\mask\ ")
        self.chk_save_map.setChecked(bool(self.cfg.get("mask_save_map_file", False)))
        form3.addRow("", self.chk_save_map)

        self.chk_only_map = QCheckBox("不修改原圖，僅輸出黑白 Mask (動態顯示)")
        self.chk_only_map.setChecked(bool(self.cfg.get("mask_only_output_map", False)))
        self.chk_only_map.setToolTip("維持原圖檔不變，但在軟體顯示時會套用 mask/ 內的黑白圖進行去背預覽")
        form3.addRow("", self.chk_only_map)

        self.ed_remover_mode = QLineEdit(str(self.cfg.get("mask_remover_mode", "base-nightly")))
        self.ed_remover_mode.setToolTip("Remover 模式：base, base-nightly, fast")
        form3.addRow("Remover Mode:", self.ed_remover_mode)

        self.chk_mask_batch_skip = QCheckBox("批量處理時跳過已去背過的圖片")
        self.chk_mask_batch_skip.setChecked(bool(self.cfg.get("mask_batch_skip_once_processed", True)))
        form3.addRow("", self.chk_mask_batch_skip)

        mask_layout.addLayout(form3)

        # Batch Ratio Limits
        ratio_box = QGroupBox("Batch Mask 主體佔比限制")
        ratio_box.setToolTip("根據去背後主體佔畫面的比例來決定是否套用去背")
        ratio_lay = QFormLayout()
        
        self.spin_mask_min_ratio = QDoubleSpinBox()
        self.spin_mask_min_ratio.setRange(0.0, 1.0)
        self.spin_mask_min_ratio.setSingleStep(0.05)
        self.spin_mask_min_ratio.setValue(float(self.cfg.get("mask_batch_min_foreground_ratio", 0.1)))
        self.spin_mask_min_ratio.setToolTip("主體佔比下限。若主體太小 (佔比低於此值)，可能是誤判，跳過不處理")
        ratio_lay.addRow("Min Ratio (主體過小跳過):", self.spin_mask_min_ratio)

        self.spin_mask_max_ratio = QDoubleSpinBox()
        self.spin_mask_max_ratio.setRange(0.0, 1.0)
        self.spin_mask_max_ratio.setSingleStep(0.05)
        self.spin_mask_max_ratio.setValue(float(self.cfg.get("mask_batch_max_foreground_ratio", 0.8)))
        self.spin_mask_max_ratio.setToolTip("主體佔比上限。若主體佔滿畫面 (無背景可去)，跳過不處理")
        ratio_lay.addRow("Max Ratio (主體過大跳過):", self.spin_mask_max_ratio)
        
        self.chk_skip_scenery = QCheckBox("跳過場景圖 (含 indoors/outdoors 標籤)")
        self.chk_skip_scenery.setChecked(bool(self.cfg.get("mask_batch_skip_if_scenery_tag", True)))
        self.chk_skip_scenery.setToolTip("勾選後，若標籤含 indoors 或 outdoors (場景圖)，則跳過去背")
        ratio_lay.addRow("", self.chk_skip_scenery)

        ratio_box.setLayout(ratio_lay)
        mask_layout.addWidget(ratio_box)

        mask_layout.addStretch(1)
        self.tabs.addTab(tab_mask, self.tr("setting_tab_mask"))

        # ---- Tags Filter (Character Tags) ----
        tab_filter = QWidget()
        filter_layout = QVBoxLayout(tab_filter)
        filter_layout.addWidget(QLabel(self.tr("setting_filter_title")))
        filter_layout.addWidget(QLabel(self.tr("setting_filter_info")))
        
        bl_label = QLabel(self.tr("setting_bl_words"))
        bl_label.setToolTip("包含這些關鍵字的標籤會被標記為『特徵標籤』(紅框)，\n批量寫入 txt 時可選擇自動刪除")
        filter_layout.addWidget(bl_label)
        self.ed_bl_words = QPlainTextEdit()
        self.ed_bl_words.setPlainText(", ".join(self.cfg.get("char_tag_blacklist_words", [])))
        self.ed_bl_words.setMinimumHeight(120)
        self.ed_bl_words.setToolTip("例如: hair, eyes, skin 等通用外觀描述\n這些標籤適合用於 LoRA 訓練時過濾")
        filter_layout.addWidget(self.ed_bl_words)

        wl_label = QLabel(self.tr("setting_wl_words"))
        wl_label.setToolTip("包含這些關鍵字的標籤即使符合黑名單也不會被標記")
        filter_layout.addWidget(wl_label)
        self.ed_wl_words = QPlainTextEdit()
        self.ed_wl_words.setPlainText(", ".join(self.cfg.get("char_tag_whitelist_words", [])))
        self.ed_wl_words.setMinimumHeight(80)
        self.ed_wl_words.setToolTip("例如: holding hair, background 等動作或情境描述\n這些標籤不是角色固有特徵，應該保留")
        filter_layout.addWidget(self.ed_wl_words)
        
        filter_layout.addStretch(1)
        
        self.tabs.addTab(tab_filter, self.tr("setting_tab_filter"))

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QPushButton(self.tr("setting_save"))
        self.btn_cancel = QPushButton(self.tr("setting_cancel"))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        self.layout.addLayout(btns)

    def _parse_tags(self, s: str):
        raw = (s or "").strip()
        if not raw:
            return []
        if "\n" in raw:
            parts = [x.strip() for x in raw.splitlines() if x.strip()]
        else:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
        return [p.replace("_", " ").strip() for p in parts if p.strip()]

    def make_hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def get_cfg(self) -> dict:
        cfg = dict(self.cfg)

        cfg["llm_base_url"] = self.ed_base_url.text().strip() or DEFAULT_APP_SETTINGS["llm_base_url"]
        cfg["llm_api_key"] = self.ed_api_key.text().strip()
        cfg["llm_model"] = self.ed_model.text().strip() or DEFAULT_APP_SETTINGS["llm_model"]
        cfg["llm_system_prompt"] = self.ed_system_prompt.toPlainText()
        cfg["llm_user_prompt_template"] = self.ed_user_template.toPlainText()
        cfg["llm_custom_prompt_template"] = self.ed_custom_template.toPlainText()
        cfg["llm_max_image_dimension"] = self.spin_llm_dim.value()
        cfg["llm_skip_nsfw_on_batch"] = self.chk_llm_skip_nsfw.isChecked()
        cfg["llm_use_gray_mask"] = self.chk_llm_use_gray_mask.isChecked()
        cfg["default_custom_tags"] = self._parse_tags(self.ed_default_custom_tags.toPlainText())

        cfg["tagger_model"] = self.ed_tagger_model.text().strip() or DEFAULT_APP_SETTINGS["tagger_model"]
        cfg["general_threshold"] = _coerce_float(self.ed_general_threshold.text(), DEFAULT_APP_SETTINGS["general_threshold"])
        cfg["general_mcut_enabled"] = self.chk_general_mcut.isChecked()
        cfg["character_threshold"] = _coerce_float(self.ed_character_threshold.text(), DEFAULT_APP_SETTINGS["character_threshold"])
        cfg["character_mcut_enabled"] = self.chk_character_mcut.isChecked()
        cfg["drop_overlap"] = self.chk_drop_overlap.isChecked()

        cfg["english_force_lowercase"] = self.chk_force_lower.isChecked()
        cfg["text_auto_remove_empty_lines"] = self.chk_auto_remove_empty.isChecked()
        cfg["text_auto_format"] = self.chk_auto_format.isChecked()
        cfg["text_auto_save"] = self.chk_auto_save.isChecked()
        cfg["batch_to_txt_mode"] = "overwrite" if self.rb_batch_overwrite.isChecked() else "append"
        cfg["batch_to_txt_folder_trigger"] = self.chk_folder_trigger.isChecked()

        a = _coerce_int(self.ed_mask_alpha.text(), DEFAULT_APP_SETTINGS["mask_default_alpha"])
        # Rule: 1-254 (USER request: "1-254 才對 (以防RGB丟失)")
        a = max(1, min(254, a))
        
        fmt = (self.ed_mask_format.text().strip().lower() or DEFAULT_APP_SETTINGS["mask_default_format"]).strip(".")
        if fmt not in ("webp", "png"):
            fmt = DEFAULT_APP_SETTINGS["mask_default_format"]

        cfg["mask_default_alpha"] = a
        cfg["mask_default_format"] = fmt
        cfg["mask_padding"] = self.spin_mask_padding.value()
        cfg["mask_blur_radius"] = self.spin_mask_blur.value()
        cfg["mask_batch_only_if_has_background_tag"] = self.chk_mask_bg_only.isChecked()
        cfg["mask_batch_detect_text_enabled"] = self.chk_mask_ocr.isChecked()
        cfg["mask_delete_npz_on_move"] = self.chk_mask_del_npz.isChecked()
        cfg["mask_ocr_heat_threshold"] = float(f"{self.spin_ocr_heat.value():.2f}")
        cfg["mask_ocr_box_threshold"] = float(f"{self.spin_ocr_box.value():.2f}")
        cfg["mask_ocr_unclip_ratio"] = float(f"{self.spin_ocr_unclip.value():.2f}")
        cfg["mask_batch_min_foreground_ratio"] = float(f"{self.spin_mask_min_ratio.value():.2f}")
        cfg["mask_batch_max_foreground_ratio"] = float(f"{self.spin_mask_max_ratio.value():.2f}")
        cfg["mask_batch_skip_if_scenery_tag"] = self.chk_skip_scenery.isChecked()
        cfg["mask_reverse"] = self.chk_mask_reverse.isChecked()
        cfg["mask_save_map_file"] = self.chk_save_map.isChecked()
        cfg["mask_only_output_map"] = self.chk_only_map.isChecked()
        cfg["mask_remover_mode"] = self.ed_remover_mode.text().strip()
        cfg["mask_batch_skip_once_processed"] = self.chk_mask_batch_skip.isChecked()
        cfg["mask_text_alpha"] = self.spin_mask_text_alpha.value()

        # Tags Filter
        cfg["char_tag_blacklist_words"] = self._parse_tags(self.ed_bl_words.toPlainText())
        cfg["char_tag_whitelist_words"] = self._parse_tags(self.ed_wl_words.toPlainText())

        cfg["ui_language"] = self.cb_lang.currentData()
        cfg["ui_theme"] = "dark" if self.rb_dark.isChecked() else "light"

        return cfg
