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
        self.statusBar().showMessage(self.tr("status_task_completed").replace("{name}", name), 5000)
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

    def on_batch_done(self, msg=None):
        if msg is None:
            msg = self.tr("msg_batch_completed")
        self.hide_progress()
        if hasattr(self, "btn_cancel_batch"):
            self.btn_cancel_batch.setVisible(False)
            self.btn_cancel_batch.setEnabled(False)
        QMessageBox.information(self, self.tr("title_batch"), msg)
        unload_all_models()
        
    def on_batch_error(self, err):
        self.set_batch_ui_enabled(True)
        self.btn_auto_tag.setEnabled(True)
        self.btn_run_llm.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage(self.tr("status_batch_error").replace("{err}", str(err)), 8000)

    def cancel_batch(self):
        self.statusBar().showMessage(self.tr("status_aborting"), 2000)
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
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return
            
        self.btn_batch_tagger.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)
        self.btn_batch_tagger_to_txt.setEnabled(False)
        self._is_batch_to_txt = False

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_tagger(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_tagger_to_txt(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
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
                self.statusBar().showMessage(self.tr("msg_restore_sidecar_tagger").replace("{count}", str(already_done_count)), 5000)

            if not files_to_process:
                self.set_batch_ui_enabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, self.tr("title_batch_tagger_txt"), self.tr("msg_batch_done_fmt").replace("{count}", str(already_done_count)))
                return

            self.statusBar().showMessage(self.tr("status_remaining_tagger").replace("{count}", str(len(files_to_process))), 5000)

            images = create_image_data_list(files_to_process)
            self.pipeline_manager.run_tagger(images)

        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_llm(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return
            
        user_prompt = self.prompt_edit.toPlainText()
        if "{角色名}" in user_prompt:
             reply = QMessageBox.question(self, self.tr("title_warning"), self.tr("msg_prompt_char_name_warn"), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 return

        self.btn_batch_llm.setEnabled(False)
        self.btn_batch_llm_to_txt.setEnabled(False)
        self.btn_run_llm.setEnabled(False)
        self._is_batch_to_txt = False
        
        try:
             images = create_image_data_list(self.image_files)
             user_prompt = self.prompt_edit.toPlainText()
             self.pipeline_manager.run_llm(
                 images,
                 user_prompt=user_prompt
             )
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_llm_to_txt(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
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
                self.statusBar().showMessage(self.tr("msg_restore_sidecar_llm").replace("{count}", str(already_done_count)), 5000)

            if not files_to_process:
                self.set_batch_ui_enabled(True)
                self.btn_run_llm.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, self.tr("title_batch_llm_txt"), self.tr("msg_batch_done_fmt").replace("{count}", str(already_done_count)))
                return

            self.statusBar().showMessage(self.tr("status_remaining_llm").replace("{count}", str(len(files_to_process))), 5000)
            
            images = create_image_data_list(files_to_process)
            user_prompt = self.prompt_edit.toPlainText()
            self.pipeline_manager.run_llm(
                 images,
                 user_prompt=user_prompt
            )

        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_unmask_background(self):
        if not self.image_files:
            return
        if self.pipeline_manager.is_running():
             QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
             return

        only_bg = bool(self.settings.get("mask_batch_only_if_has_background_tag", False))
        if only_bg:
            targets = [p for p in self.image_files if self._image_has_background_tag(p)]
            if not targets:
                QMessageBox.information(self, self.tr("title_batch_unmask"), self.tr("msg_no_bg_tag_found"))
                return
        else:
            targets = self.image_files

        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(False)
            
        try:
             images = create_image_data_list(targets)
             self.pipeline_manager.run_unmask(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_mask_text(self):
        if not self.image_files:
            QMessageBox.information(self, self.tr("title_info"), self.tr("msg_no_images"))
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_mask_text(images)
        except Exception as e:
             self.on_pipeline_error(str(e))

    def run_batch_restore(self):
        if not self.image_files:
            QMessageBox.information(self, self.tr("title_info"), self.tr("msg_no_images"))
            return

        if self.pipeline_manager.is_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return

        reply = QMessageBox.question(
            self, self.tr("title_batch_restore"),
            self.tr("msg_batch_restore_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
             images = create_image_data_list(self.image_files)
             self.pipeline_manager.run_restore(images)
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
        msg.setWindowTitle(self.tr("title_batch_to_txt"))
        msg.setText(self.tr("msg_batch_delete_char_tags"))
        msg.setInformativeText(self.tr("msg_batch_delete_info"))
        btn_yes = msg.addButton(self.tr("btn_auto_delete"), QMessageBox.ButtonRole.YesRole)
        btn_no = msg.addButton(self.tr("btn_keep"), QMessageBox.ButtonRole.NoRole)
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
