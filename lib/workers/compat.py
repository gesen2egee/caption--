# -*- coding: utf-8 -*-
"""
Workers 兼容層

提供舊式 QThread Workers 到新 Task 架構的橋接。
這允許漸進式遷移，不需要一次性修改所有 UI 程式碼。
"""
from typing import Callable, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from lib.core.dataclasses import ImageData, Settings
from lib.core.settings import load_app_settings, DEFAULT_APP_SETTINGS


def create_image_data(image_path: str) -> ImageData:
    """從圖片路徑建立 ImageData"""
    from lib.utils.sidecar import load_image_sidecar
    
    sidecar = load_image_sidecar(image_path)
    return ImageData(
        path=image_path,
        tagger_tags=sidecar.get("tagger_tags"),
        nl_pages=sidecar.get("nl_pages", []),
        masked_background=sidecar.get("masked_background", False),
        masked_text=sidecar.get("masked_text", False),
        raw_image_rel_path=sidecar.get("raw_image_rel_path") or sidecar.get("raw_backup_path"),
    )


def create_settings_from_dict(cfg: dict) -> Settings:
    """從設定字典建立 Settings dataclass"""
    defaults = DEFAULT_APP_SETTINGS
    
    return Settings(
        # LLM
        llm_base_url=cfg.get("llm_base_url", defaults["llm_base_url"]),
        llm_api_key=cfg.get("llm_api_key", defaults["llm_api_key"]),
        llm_model=cfg.get("llm_model", defaults["llm_model"]),
        llm_system_prompt=cfg.get("llm_system_prompt", defaults["llm_system_prompt"]),
        llm_user_prompt_template=cfg.get("llm_user_prompt_template", defaults["llm_user_prompt_template"]),
        llm_custom_prompt_template=cfg.get("llm_custom_prompt_template", defaults["llm_custom_prompt_template"]),
        llm_max_image_dimension=cfg.get("llm_max_image_dimension", defaults["llm_max_image_dimension"]),
        llm_skip_nsfw_on_batch=cfg.get("llm_skip_nsfw_on_batch", defaults["llm_skip_nsfw_on_batch"]),
        llm_use_gray_mask=cfg.get("llm_use_gray_mask", defaults["llm_use_gray_mask"]),
        
        # Tagger
        tagger_model=cfg.get("tagger_model", defaults["tagger_model"]),
        general_threshold=cfg.get("general_threshold", defaults["general_threshold"]),
        general_mcut_enabled=cfg.get("general_mcut_enabled", defaults["general_mcut_enabled"]),
        character_threshold=cfg.get("character_threshold", defaults["character_threshold"]),
        character_mcut_enabled=cfg.get("character_mcut_enabled", defaults["character_mcut_enabled"]),
        drop_overlap=cfg.get("drop_overlap", defaults["drop_overlap"]),
        
        # Text
        english_force_lowercase=cfg.get("english_force_lowercase", defaults["english_force_lowercase"]),
        text_auto_remove_empty_lines=cfg.get("text_auto_remove_empty_lines", defaults["text_auto_remove_empty_lines"]),
        text_auto_format=cfg.get("text_auto_format", defaults["text_auto_format"]),
        text_auto_save=cfg.get("text_auto_save", defaults["text_auto_save"]),
        batch_to_txt_mode=cfg.get("batch_to_txt_mode", defaults["batch_to_txt_mode"]),
        batch_to_txt_folder_trigger=cfg.get("batch_to_txt_folder_trigger", defaults["batch_to_txt_folder_trigger"]),
        
        # Mask
        mask_remover_mode=cfg.get("mask_remover_mode", defaults["mask_remover_mode"]),
        mask_default_alpha=cfg.get("mask_default_alpha", defaults["mask_default_alpha"]),
        mask_default_format=cfg.get("mask_default_format", defaults["mask_default_format"]),
        mask_reverse=cfg.get("mask_reverse", defaults["mask_reverse"]),
        mask_save_map_file=cfg.get("mask_save_map_file", defaults["mask_save_map_file"]),
        mask_only_output_map=cfg.get("mask_only_output_map", defaults["mask_only_output_map"]),
        mask_batch_only_if_has_background_tag=cfg.get("mask_batch_only_if_has_background_tag", defaults["mask_batch_only_if_has_background_tag"]),
        mask_batch_detect_text_enabled=cfg.get("mask_batch_detect_text_enabled", defaults["mask_batch_detect_text_enabled"]),
        mask_delete_npz_on_move=cfg.get("mask_delete_npz_on_move", defaults["mask_delete_npz_on_move"]),
        mask_padding=cfg.get("mask_padding", defaults["mask_padding"]),
        mask_blur_radius=cfg.get("mask_blur_radius", defaults["mask_blur_radius"]),
        mask_batch_skip_once_processed=cfg.get("mask_batch_skip_once_processed", defaults["mask_batch_skip_once_processed"]),
        mask_batch_min_foreground_ratio=cfg.get("mask_batch_min_foreground_ratio", defaults["mask_batch_min_foreground_ratio"]),
        mask_batch_max_foreground_ratio=cfg.get("mask_batch_max_foreground_ratio", defaults["mask_batch_max_foreground_ratio"]),
        mask_batch_skip_if_scenery_tag=cfg.get("mask_batch_skip_if_scenery_tag", defaults["mask_batch_skip_if_scenery_tag"]),
        
        # OCR
        mask_ocr_max_candidates=cfg.get("mask_ocr_max_candidates", defaults["mask_ocr_max_candidates"]),
        mask_ocr_heat_threshold=cfg.get("mask_ocr_heat_threshold", defaults["mask_ocr_heat_threshold"]),
        mask_ocr_box_threshold=cfg.get("mask_ocr_box_threshold", defaults["mask_ocr_box_threshold"]),
        mask_ocr_unclip_ratio=cfg.get("mask_ocr_unclip_ratio", defaults["mask_ocr_unclip_ratio"]),
        mask_text_alpha=cfg.get("mask_text_alpha", defaults["mask_text_alpha"]),
        
        # Filter
        char_tag_blacklist_words=cfg.get("char_tag_blacklist_words", defaults["char_tag_blacklist_words"]),
        char_tag_whitelist_words=cfg.get("char_tag_whitelist_words", defaults["char_tag_whitelist_words"]),
        default_custom_tags=cfg.get("default_custom_tags", defaults["default_custom_tags"]),
        
        # UI
        ui_language=cfg.get("ui_language", defaults["ui_language"]),
        ui_theme=cfg.get("ui_theme", defaults["ui_theme"]),
        last_open_dir=cfg.get("last_open_dir", defaults["last_open_dir"]),
    )


# ============================================================
# 兼容層 Workers (舊式 API，內部使用新架構)
# ============================================================

class TaggerWorkerCompat(QThread):
    """
    Tagger Worker 兼容層
    
    保持舊的 signal 介面，內部使用新的 Worker 架構。
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, image_path: str, cfg: dict):
        super().__init__()
        self.image_path = image_path
        self.cfg = dict(cfg or {})
    
    def run(self):
        try:
            from lib.workers.tagger_imgutils_tagging_local import TaggerImgutilsTaggingLocalWorker
            from lib.workers.base import WorkerInput
            
            # 建立 Worker
            worker = TaggerImgutilsTaggingLocalWorker({
                "model_name": self.cfg.get("tagger_model", "EVA02_Large"),
                "general_threshold": self.cfg.get("general_threshold", 0.2),
                "general_mcut_enabled": self.cfg.get("general_mcut_enabled", False),
                "character_threshold": self.cfg.get("character_threshold", 0.85),
                "character_mcut_enabled": self.cfg.get("character_mcut_enabled", True),
                "drop_overlap": self.cfg.get("drop_overlap", True),
            })
            
            # 準備輸入
            image_data = create_image_data(self.image_path)
            input_data = WorkerInput(image=image_data)
            
            # 執行
            output = worker.process(input_data)
            
            if output.success:
                # 組合成舊格式的字串
                result_data = output.result_data or {}
                rating = result_data.get("rating", {})
                features = result_data.get("features", {})
                chars = result_data.get("chars", {})
                
                rating_tag = f"rating:{max(rating, key=rating.get)}" if rating else ""
                tags_list = [rating_tag] if rating_tag else []
                tags_list += list(chars.keys()) + list(features.keys())
                tags_str = ", ".join(tags_list)
                
                self.finished.emit(tags_str)
            else:
                self.error.emit(output.error or "未知錯誤")
                
        except Exception as e:
            self.error.emit(str(e))


class BatchTaggerWorkerCompat(QThread):
    """
    批量 Tagger Worker 兼容層
    """
    progress = pyqtSignal(int, int, str)
    per_image = pyqtSignal(str, str)
    done = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[str], cfg: dict):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def run(self):
        try:
            from lib.workers.tagger_imgutils_tagging_local import TaggerImgutilsTaggingLocalWorker
            from lib.workers.base import WorkerInput
            
            worker = TaggerImgutilsTaggingLocalWorker({
                "model_name": self.cfg.get("tagger_model", "EVA02_Large"),
                "general_threshold": self.cfg.get("general_threshold", 0.2),
                "general_mcut_enabled": self.cfg.get("general_mcut_enabled", False),
                "character_threshold": self.cfg.get("character_threshold", 0.85),
                "character_mcut_enabled": self.cfg.get("character_mcut_enabled", True),
                "drop_overlap": self.cfg.get("drop_overlap", True),
            })
            
            total = len(self.image_paths)
            for i, image_path in enumerate(self.image_paths):
                if self._stop:
                    break
                
                self.progress.emit(i + 1, total, image_path)
                
                try:
                    image_data = create_image_data(image_path)
                    input_data = WorkerInput(image=image_data)
                    output = worker.process(input_data)
                    
                    if output.success:
                        self.per_image.emit(image_path, output.result_text or "")
                except Exception as e:
                    self.per_image.emit(image_path, f"[錯誤] {e}")
                    
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()


class BatchUnmaskWorkerCompat(QThread):
    """
    批量去背 Worker 兼容層
    """
    progress = pyqtSignal(int, int, str)
    per_image = pyqtSignal(str, str)
    done = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[str], cfg: dict = None, 
                 background_tag_checker: Callable = None, is_batch: bool = True):
        super().__init__()
        self.image_paths = list(image_paths)
        self.cfg = dict(cfg or {})
        self.background_tag_checker = background_tag_checker
        self.is_batch = is_batch
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def run(self):
        try:
            from lib.workers.mask_transparent_background_local import MaskTransparentBackgroundLocalWorker
            from lib.workers.base import WorkerInput
            from lib.utils.sidecar import load_image_sidecar, save_image_sidecar
            from lib.utils.file_ops import backup_original_image
            
            worker = MaskTransparentBackgroundLocalWorker({
                "mode": self.cfg.get("mask_remover_mode", "base-nightly"),
                "default_alpha": self.cfg.get("mask_default_alpha", 64),
                "default_format": self.cfg.get("mask_default_format", "webp"),
                "padding": self.cfg.get("mask_padding", 1),
                "blur_radius": self.cfg.get("mask_blur_radius", 3),
                "min_foreground_ratio": self.cfg.get("mask_batch_min_foreground_ratio", 0.3),
                "max_foreground_ratio": self.cfg.get("mask_batch_max_foreground_ratio", 0.8),
            })
            
            total = len(self.image_paths)
            for i, image_path in enumerate(self.image_paths):
                if self._stop:
                    break
                
                self.progress.emit(i + 1, total, image_path)
                
                try:
                    # 檢查是否需要處理
                    if self.is_batch:
                        sidecar = load_image_sidecar(image_path)
                        if self.cfg.get("mask_batch_skip_once_processed") and sidecar.get("masked_background"):
                            self.per_image.emit(image_path, "[跳過] 已處理")
                            continue
                        
                        if self.cfg.get("mask_batch_only_if_has_background_tag"):
                            if self.background_tag_checker and not self.background_tag_checker(image_path):
                                self.per_image.emit(image_path, "[跳過] 無 background 標籤")
                                continue
                    
                    # 備份原圖
                    backup_original_image(image_path)
                    
                    # 執行去背
                    image_data = create_image_data(image_path)
                    input_data = WorkerInput(image=image_data)
                    output = worker.process(input_data)
                    
                    if output.success and not output.skipped:
                        # 更新 sidecar
                        sidecar = load_image_sidecar(image_path)
                        sidecar["masked_background"] = True
                        save_image_sidecar(output.result_path or image_path, sidecar)
                        self.per_image.emit(image_path, output.result_path or "完成")
                    elif output.skipped:
                        self.per_image.emit(image_path, f"[跳過] {output.skip_reason or ''}")
                    else:
                        self.per_image.emit(image_path, f"[錯誤] {output.error or ''}")
                        
                except Exception as e:
                    self.per_image.emit(image_path, f"[錯誤] {e}")
                    
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()


class BatchRestoreWorkerCompat(QThread):
    """
    批量還原 Worker 兼容層
    """
    progress = pyqtSignal(int, int, str)
    per_image = pyqtSignal(str, str)
    done = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.image_paths = list(image_paths)
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def run(self):
        try:
            from lib.workers.image_restore_raw import ImageRestoreRawWorker
            from lib.workers.base import WorkerInput
            
            worker = ImageRestoreRawWorker()
            
            total = len(self.image_paths)
            for i, image_path in enumerate(self.image_paths):
                if self._stop:
                    break
                
                self.progress.emit(i + 1, total, image_path)
                
                try:
                    image_data = create_image_data(image_path)
                    input_data = WorkerInput(image=image_data)
                    output = worker.process(input_data)
                    
                    if output.success and not output.skipped:
                        self.per_image.emit(image_path, "已還原")
                    elif output.skipped:
                        self.per_image.emit(image_path, f"[跳過] {output.skip_reason or ''}")
                    else:
                        self.per_image.emit(image_path, f"[錯誤] {output.error or ''}")
                        
                except Exception as e:
                    self.per_image.emit(image_path, f"[錯誤] {e}")
                    
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()

