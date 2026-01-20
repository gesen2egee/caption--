# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING
import os
from PyQt6.QtWidgets import QMessageBox, QDialog

from lib.pipeline.manager import create_image_data_list
from lib.utils.batch_writer import write_batch_result
from lib.utils.sidecar import load_image_sidecar
from lib.utils.memory_utils import unload_all_models

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class BatchMixin:
    """
    Mixin class handling all batch operations for MainWindow.
    """

    def on_pipeline_done(self, name, results):
        self.statusBar().showMessage(f"Task '{name}' completed.", 5000)
        self.progress_bar.setVisible(False)
        self.btn_cancel_batch.setVisible(False)
        self.set_batch_ui_enabled(True)
        
        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(True)
        
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText(self.tr("btn_run_llm"))
        
        unload_all_models()

    def set_batch_ui_enabled(self, enabled):
        self.btn_batch_tagger.setEnabled(enabled)
        self.btn_batch_tagger_to_txt.setEnabled(enabled)
        self.btn_batch_llm.setEnabled(enabled)
        self.btn_batch_llm_to_txt.setEnabled(enabled)

    def show_progress(self, current, total, name):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{name} ({current}/{total})")
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(True)
            self.btn_cancel_batch.setEnabled(True)

    def hide_progress(self):
        self.progress_bar.setVisible(False)
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")

    def on_batch_done(self, msg="Batch Process Completed"):
        self.hide_progress()
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(False)
            self.btn_cancel_batch.setEnabled(False)
        QMessageBox.information(self, "Batch", msg)
        unload_all_models()
        
    def on_batch_error(self, err):
        self.set_batch_ui_enabled(True)
        self.btn_auto_tag.setEnabled(True)
        self.btn_run_llm.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage(f"Batch Error: {err}", 8000)

    def cancel_batch(self):
        self.statusBar().showMessage("正在中止...", 2000)
        if self.pipeline_manager.is_running():
            self.pipeline_manager.stop()
        for attr in ['batch_mask_text_thread']:
             thread = getattr(self, attr, None)
             if thread is not None and thread.isRunning():
                 thread.stop()

    def run_batch_tagger(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
            
        self.btn_batch_tagger.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)
        self.btn_batch_tagger_to_txt.setEnabled(False)
        self._is_batch_to_txt = False

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_tagger(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_tagger_to_txt(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
        
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_tagger_to_txt.setEnabled(False)
        self.btn_batch_tagger.setEnabled(False)

        files_to_process = []
        already_done_count = 0

        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                tags_str = sidecar.get("tagger_tags", "")

                if tags_str:
                    self.write_batch_result_to_txt(img_path, tags_str, is_tagger=True)
                    already_done_count += 1
                else:
                    files_to_process.append(img_path)

            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 Tagger 結果至 txt", 5000)

            if not files_to_process:
                self.set_batch_ui_enabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch Tagger to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 Tagger...", 5000)

            images = create_image_data_list(files_to_process)
            self.pipeline_manager.run_batch_tagger(images)

        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_llm(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
            
        user_prompt = self.prompt_edit.toPlainText()
        if "{角色名}" in user_prompt:
             reply = QMessageBox.question(self, "Warning", "Prompt 包含未替換的 '{角色名}'。\n這可能會導致生成結果不正確。\n請手動輸入角色名或調整提示。\n\n確定要繼續嗎？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 return

        self.btn_batch_llm.setEnabled(False)
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.btn_run_llm.setEnabled(False)
        self._is_batch_to_txt = False
        
        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_llm(
                 images,
                 prompt_mode=self.current_prompt_mode,
                 user_prompt=user_prompt
             )
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_llm_to_txt(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return
            
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.btn_batch_llm.setEnabled(False)

        files_to_process = []
        already_done_count = 0
        
        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                nl = sidecar.get("nl_pages", [])
                content = nl[-1] if nl and isinstance(nl, list) else ""
                
                if content:
                    self.write_batch_result_to_txt(img_path, content, is_tagger=False)
                    already_done_count += 1
                else:
                    files_to_process.append(img_path)
            
            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 LLM 結果至 txt", 5000)

            if not files_to_process:
                self.set_batch_ui_enabled(True)
                self.btn_run_llm.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch LLM to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 LLM...", 5000)
            
            user_prompt = self.prompt_edit.toPlainText()
            
            images = create_image_data_list(files_to_process)
            self.pipeline_manager.run_batch_llm(
                 images,
                 prompt_mode=self.current_prompt_mode,
                 user_prompt=user_prompt
            )

        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_unmask_background(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
             QMessageBox.warning(self, "Warning", "已有任務正在執行中")
             return

        only_bg = bool(self.settings.get("mask_batch_only_if_has_background_tag", False))
        if only_bg:
            targets = [p for p in self.image_files if self._image_has_background_tag(p)]
            if not targets:
                QMessageBox.information(self, "Batch Unmask", "找不到含有 'background' 標籤的圖片")
                return
        else:
            targets = self.image_files

        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(False)
            
        try:
             images = create_image_data_list(targets)
             self.pipeline_manager.run_batch_unmask(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_mask_text(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_mask_text(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_restore(self):
        if not self.image_files:
            QMessageBox.information(self, "Info", "No images loaded.")
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, "Warning", "已有任務正在執行中")
            return

        reply = QMessageBox.question(
            self, "Batch Restore",
            "是否確定還原所有圖片的原檔 (若存在)？\n這將會覆蓋/刪除目前的去背版本。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_batch_restore(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def _image_has_background_tag(self, image_path: str) -> bool:
        try:
            sidecar = load_image_sidecar(image_path)
            tags_all = (sidecar.get("tagger_tags", "") + " " + sidecar.get("tags_context", "")).lower()
            if "background" in tags_all:
                return True
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                return "background" in content
        except Exception:
            pass
        return False

    def prompt_delete_chars(self) -> bool:
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch to txt")
        msg.setText("是否自動刪除特徵標籤 (Character Tags)？")
        msg.setInformativeText("將根據設定中的黑白名單過濾標籤或句子。")
        btn_yes = msg.addButton("自動刪除", QMessageBox.ButtonRole.YesRole)
        btn_no = msg.addButton("保留", QMessageBox.ButtonRole.NoRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        if msg.clickedButton() == btn_yes:
            return True
        elif msg.clickedButton() == btn_no:
            return False
        return None

    def write_batch_result_to_txt(self, image_path, content, is_tagger: bool):
        delete_chars = getattr(self, "_batch_delete_chars", False)
        final = write_batch_result(image_path, content, is_tagger, self.settings, delete_chars)
        
        if final and image_path == self.current_image_path:
             self.txt_edit.setPlainText(final)
