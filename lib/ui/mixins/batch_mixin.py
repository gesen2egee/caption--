# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Any, Dict, Optional

import lib.runtime.batch_service as batch_service
from lib.utils.sidecar import load_image_sidecar
from lib.utils.memory_utils import unload_all_models

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class BatchMixin:
    """
    Mixin class handling all batch operations for MainWindow.
    """

    def _batch_result(self, **kwargs) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "started": False,
            "task_started": False,
        }
        payload.update(kwargs)
        return payload

    def _batch_task_running_result(self, interactive: bool = False) -> Dict[str, Any]:
        if interactive:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
        return self._batch_result(reason="task_running")

    def _resolve_batch_delete_chars(
        self,
        delete_chars: Optional[bool],
        *,
        allow_dialog: bool,
    ) -> Optional[bool]:
        if delete_chars is not None:
            return bool(delete_chars)
        if allow_dialog:
            return self.prompt_delete_chars()
        return None

    def _set_batch_txt_mode(self, save_to_txt: bool, delete_chars: bool = False) -> None:
        self._is_batch_to_txt = bool(save_to_txt)
        self._batch_delete_chars = bool(delete_chars)

    def _clear_batch_txt_mode(self) -> None:
        self._is_batch_to_txt = False
        self._batch_delete_chars = False

    def on_pipeline_done(self, name, results):
        # 分析結果
        total = len(results)
        skipped = sum(1 for r in results if r.skipped)
        errors = sum(1 for r in results if not r.success)
        
        # 如果是單張處理
        if total == 1:
            res = results[0]
            if res.skipped:
                reason = res.skip_reason or self.tr("msg_task_skipped")
                self.statusBar().showMessage(f"{name}: {reason}", 5000)
            elif not res.success:
                self.statusBar().showMessage(f"{name}: 失敗 - {res.error}", 5000)
            else:
                # 成功情況，檢查特殊狀態
                is_mask_text_no_box = "mask_text" in name.lower() and res.result_data and res.result_data.get("box_count", 0) == 0
                
                if is_mask_text_no_box:
                    self.statusBar().showMessage(self.tr("msg_no_text_detected"), 5000)
                else:
                    self.statusBar().showMessage(self.tr("status_task_completed").replace("{name}", name), 5000)
        else:
            # 批量處理
            msg = self.tr("status_task_completed").replace("{name}", name)
            if skipped > 0 or errors > 0:
                msg += f" (Success: {total-skipped-errors}, Skip: {skipped}, Err: {errors})"
            self.statusBar().showMessage(msg, 5000)
            
        self.progress_bar.setVisible(False)
        self.btn_cancel_batch.setVisible(False)
        self.set_batch_ui_enabled(True)
        
        if hasattr(self, 'action_batch_unmask'):
            self.action_batch_unmask.setEnabled(True)
        
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText(self.tr("btn_run_llm"))
        if hasattr(self, "btn_run_imgproc"):
            self.btn_run_imgproc.setEnabled(True)
            self.btn_run_imgproc.setText(self.tr("btn_run_imgproc"))
        
        unload_all_models()

    def set_batch_ui_enabled(self, enabled):
        self.btn_batch_tagger.setEnabled(enabled)
        if hasattr(self, 'chk_tags_save_txt'):
             self.chk_tags_save_txt.setEnabled(enabled)
        self.btn_batch_llm.setEnabled(enabled)
        if hasattr(self, 'chk_llm_save_txt'):
             self.chk_llm_save_txt.setEnabled(enabled)
        if hasattr(self, "btn_batch_imgproc"):
            self.btn_batch_imgproc.setEnabled(enabled)

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
        from PyQt6.QtWidgets import QMessageBox

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
        if hasattr(self, "btn_run_imgproc"):
            self.btn_run_imgproc.setEnabled(True)
        self._is_batch_to_txt = False
        self.hide_progress()
        self.statusBar().showMessage(self.tr("status_batch_error").replace("{err}", str(err)), 8000)

    def cancel_batch(self):
        self.statusBar().showMessage(self.tr("status_aborting"), 2000)
        self.btn_cancel_batch.setEnabled(False)
        self.stop_current_task()
        for attr in ['batch_mask_text_thread']:
             thread = getattr(self, attr, None)
             if thread is not None and thread.isRunning():
                 thread.stop()

    def run_batch_tagger(self):
        try:
            self.execute_command("batch.run_tagger", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_llm(self):
        try:
            self.execute_command("batch.run_llm", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_image_processing(self):
        try:
            self.execute_command("batch.run_image_process")
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_unmask_background(self):
        try:
            self.execute_command("batch.run_unmask", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_mask_text(self):
        try:
            self.execute_command("batch.run_mask_text", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_batch_restore(self):
        try:
            self.execute_command("batch.run_restore", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def _image_has_background_tag(self, image_path: str) -> bool:
        return bool(batch_service.filter_background_targets([image_path]))

    def prompt_delete_chars(self) -> bool:
        from PyQt6.QtWidgets import QMessageBox

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
        from lib.utils.batch_writer import write_batch_result
        delete_chars = getattr(self, "_batch_delete_chars", False)
        final = write_batch_result(image_path, content, is_tagger, self.settings, delete_chars)
        
        if final and image_path == self._runtime_current_image_path() and hasattr(self, "txt_edit") and self.txt_edit is not None:
             self.txt_edit.setPlainText(final)

    def command_run_batch_tagger(
        self,
        save_to_txt: Optional[bool] = None,
        delete_chars: Optional[bool] = None,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        image_paths = self._runtime_loaded_image_paths()
        if not image_paths:
            return self._batch_result(reason="no_images", image_count=0)
        if self.is_task_running():
            return self._batch_task_running_result(interactive=allow_dialog)

        resolved_save_to_txt = bool(save_to_txt) if save_to_txt is not None else self._runtime_tagger_save_to_txt()

        self.btn_batch_tagger.setEnabled(False)
        self.btn_auto_tag.setEnabled(False)
        if hasattr(self, "chk_tags_save_txt"):
            self.chk_tags_save_txt.setEnabled(False)

        files_to_process = list(image_paths)
        restored_files = []
        resolved_delete_chars = False

        if resolved_save_to_txt:
            delete_choice = self._resolve_batch_delete_chars(delete_chars, allow_dialog=allow_dialog)
            if delete_choice is None:
                self.set_batch_ui_enabled(True)
                self.btn_auto_tag.setEnabled(True)
                return self._batch_result(
                    reason="delete_chars_required" if not allow_dialog else "cancelled",
                    save_to_txt=True,
                )

            resolved_delete_chars = bool(delete_choice)
            self._set_batch_txt_mode(True, resolved_delete_chars)
            restore_result = batch_service.restore_batch_tagger_to_txt(
                image_paths,
                self.settings,
                delete_chars=resolved_delete_chars,
                write_callback=lambda img_path, _final: self.write_batch_result_to_txt(
                    img_path,
                    load_image_sidecar(img_path).get("tagger_tags", ""),
                    is_tagger=True,
                ),
            )
            files_to_process = list(restore_result["files_to_process"])
            restored_files = list(restore_result["restored_files"])

            if restored_files:
                self.statusBar().showMessage(
                    self.tr("msg_restore_sidecar_tagger").replace("{count}", str(len(restored_files))),
                    5000,
                )

            if not files_to_process:
                self.set_batch_ui_enabled(True)
                self.btn_auto_tag.setEnabled(True)
                self._clear_batch_txt_mode()
                if allow_dialog:
                    from PyQt6.QtWidgets import QMessageBox

                    QMessageBox.information(
                        self,
                        self.tr("title_batch_tagger_txt"),
                        self.tr("msg_batch_done_fmt").replace("{count}", str(len(restored_files))),
                    )
                return self._batch_result(
                    reason="completed_from_sidecar",
                    completed=True,
                    save_to_txt=True,
                    delete_chars=resolved_delete_chars,
                    restored_from_sidecar_count=len(restored_files),
                    restored_from_sidecar_files=restored_files,
                    files_to_process_count=0,
                    files_to_process=[],
                )

            self.statusBar().showMessage(
                self.tr("status_remaining_tagger").replace("{count}", str(len(files_to_process))),
                5000,
            )
        else:
            self._clear_batch_txt_mode()

        try:
            self.execute_command("task.run_tagger", files_to_process)
        except Exception:
            self._clear_batch_txt_mode()
            self.set_batch_ui_enabled(True)
            self.btn_auto_tag.setEnabled(True)
            raise

        return self._batch_result(
            started=True,
            task_started=True,
            mode="tagger",
            save_to_txt=resolved_save_to_txt,
            delete_chars=resolved_delete_chars if resolved_save_to_txt else False,
            target_count=len(image_paths),
            files_to_process_count=len(files_to_process),
            files_to_process=list(files_to_process),
            restored_from_sidecar_count=len(restored_files),
            restored_from_sidecar_files=restored_files,
        )

    def command_run_batch_llm(
        self,
        user_prompt: Optional[str] = None,
        save_to_txt: Optional[bool] = None,
        delete_chars: Optional[bool] = None,
        confirm_character_prompt: bool = False,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        image_paths = self._runtime_loaded_image_paths()
        if not image_paths:
            return self._batch_result(reason="no_images", image_count=0)
        if self.is_task_running():
            return self._batch_task_running_result(interactive=allow_dialog)

        resolved_save_to_txt = bool(save_to_txt) if save_to_txt is not None else self._runtime_llm_save_to_txt()
        resolved_user_prompt = (
            str(user_prompt)
            if user_prompt is not None
            else self._runtime_prompt_text()
        )

        if "{角色名}" in resolved_user_prompt and not confirm_character_prompt:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                reply = QMessageBox.question(
                    self,
                    self.tr("title_warning"),
                    self.tr("msg_prompt_char_name_warn"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return self._batch_result(reason="cancelled", save_to_txt=resolved_save_to_txt)
                confirm_character_prompt = True
            else:
                return self._batch_result(
                    reason="confirm_character_prompt_required",
                    save_to_txt=resolved_save_to_txt,
                    prompt_contains_character_name=True,
                )

        self.btn_batch_llm.setEnabled(False)
        self.btn_run_llm.setEnabled(False)
        if hasattr(self, "chk_llm_save_txt"):
            self.chk_llm_save_txt.setEnabled(False)

        files_to_process = list(image_paths)
        restored_files = []
        resolved_delete_chars = False

        if resolved_save_to_txt:
            delete_choice = self._resolve_batch_delete_chars(delete_chars, allow_dialog=allow_dialog)
            if delete_choice is None:
                self.set_batch_ui_enabled(True)
                self.btn_run_llm.setEnabled(True)
                return self._batch_result(
                    reason="delete_chars_required" if not allow_dialog else "cancelled",
                    save_to_txt=True,
                    prompt_contains_character_name="{角色名}" in resolved_user_prompt,
                )

            resolved_delete_chars = bool(delete_choice)
            self._set_batch_txt_mode(True, resolved_delete_chars)
            restore_result = batch_service.restore_batch_llm_to_txt(
                image_paths,
                self.settings,
                delete_chars=resolved_delete_chars,
                write_callback=lambda img_path, content: self.write_batch_result_to_txt(
                    img_path,
                    content,
                    is_tagger=False,
                ),
            )
            files_to_process = list(restore_result["files_to_process"])
            restored_files = list(restore_result["restored_files"])

            if restored_files:
                self.statusBar().showMessage(
                    self.tr("msg_restore_sidecar_llm").replace("{count}", str(len(restored_files))),
                    5000,
                )

            if not files_to_process:
                self.set_batch_ui_enabled(True)
                self.btn_run_llm.setEnabled(True)
                self._clear_batch_txt_mode()
                if allow_dialog:
                    from PyQt6.QtWidgets import QMessageBox

                    QMessageBox.information(
                        self,
                        self.tr("title_batch_llm_txt"),
                        self.tr("msg_batch_done_fmt").replace("{count}", str(len(restored_files))),
                    )
                return self._batch_result(
                    reason="completed_from_sidecar",
                    completed=True,
                    save_to_txt=True,
                    delete_chars=resolved_delete_chars,
                    user_prompt=resolved_user_prompt,
                    restored_from_sidecar_count=len(restored_files),
                    restored_from_sidecar_files=restored_files,
                    files_to_process_count=0,
                    files_to_process=[],
                )

            self.statusBar().showMessage(
                self.tr("status_remaining_llm").replace("{count}", str(len(files_to_process))),
                5000,
            )
        else:
            self._clear_batch_txt_mode()

        try:
            self.execute_command("task.run_llm", files_to_process, user_prompt=resolved_user_prompt)
        except Exception:
            self._clear_batch_txt_mode()
            self.set_batch_ui_enabled(True)
            self.btn_run_llm.setEnabled(True)
            raise

        return self._batch_result(
            started=True,
            task_started=True,
            mode="llm",
            user_prompt=resolved_user_prompt,
            save_to_txt=resolved_save_to_txt,
            delete_chars=resolved_delete_chars if resolved_save_to_txt else False,
            prompt_contains_character_name="{角色名}" in resolved_user_prompt,
            target_count=len(image_paths),
            files_to_process_count=len(files_to_process),
            files_to_process=list(files_to_process),
            restored_from_sidecar_count=len(restored_files),
            restored_from_sidecar_files=restored_files,
        )

    def command_run_batch_image_process(self, edit_prompt: Optional[str] = None) -> Dict[str, Any]:
        image_paths = self._runtime_loaded_image_paths()
        if not image_paths:
            return self._batch_result(reason="no_images", image_count=0)
        if self.is_task_running():
            return self._batch_task_running_result()

        prompt = str(edit_prompt or "").strip()
        if not prompt:
            prompt = self._runtime_image_process_prompt_text().strip()
        if not prompt:
            prompt = self.settings.get("image_process_prompt_template", "幫我移除圖中所有的文字、文字氣泡、文字框")

        if hasattr(self, "btn_batch_imgproc"):
            self.btn_batch_imgproc.setEnabled(False)
        if hasattr(self, "btn_run_imgproc"):
            self.btn_run_imgproc.setEnabled(False)

        try:
            self.execute_command("task.run_image_process", image_paths, edit_prompt=prompt)
        except Exception:
            self.set_batch_ui_enabled(True)
            if hasattr(self, "btn_run_imgproc"):
                self.btn_run_imgproc.setEnabled(True)
            raise

        return self._batch_result(
            started=True,
            task_started=True,
            mode="image_process",
            edit_prompt=prompt,
            target_count=len(image_paths),
            files_to_process_count=len(image_paths),
            files_to_process=list(image_paths),
        )

    def command_run_batch_unmask(
        self,
        only_bg: Optional[bool] = None,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        image_paths = self._runtime_loaded_image_paths()
        if not image_paths:
            return self._batch_result(reason="no_images", image_count=0)
        if self.is_task_running():
            return self._batch_task_running_result(interactive=allow_dialog)

        resolved_only_bg = bool(only_bg) if only_bg is not None else bool(
            self.settings.get("mask_batch_only_if_has_background_tag", False)
        )
        targets = batch_service.filter_background_targets(image_paths) if resolved_only_bg else list(image_paths)
        if not targets:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.information(self, self.tr("title_batch_unmask"), self.tr("msg_no_bg_tag_found"))
            return self._batch_result(reason="no_bg_tag_found", only_bg=True, target_count=0, files_to_process=[])

        if hasattr(self, "action_batch_unmask"):
            self.action_batch_unmask.setEnabled(False)

        try:
            self.execute_command("task.run_unmask", targets)
        except Exception:
            if hasattr(self, "action_batch_unmask"):
                self.action_batch_unmask.setEnabled(True)
            raise

        return self._batch_result(
            started=True,
            task_started=True,
            mode="unmask",
            only_bg=resolved_only_bg,
            target_count=len(targets),
            files_to_process_count=len(targets),
            files_to_process=list(targets),
        )

    def command_run_batch_mask_text(self, allow_dialog: bool = False) -> Dict[str, Any]:
        image_paths = self._runtime_loaded_image_paths()
        if not image_paths:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.information(self, self.tr("title_info"), self.tr("msg_no_images"))
            return self._batch_result(reason="no_images", image_count=0)
        if self.is_task_running():
            return self._batch_task_running_result(interactive=allow_dialog)

        self.execute_command("task.run_mask_text", image_paths)
        return self._batch_result(
            started=True,
            task_started=True,
            mode="mask_text",
            target_count=len(image_paths),
            files_to_process_count=len(image_paths),
            files_to_process=list(image_paths),
        )

    def command_run_batch_restore(
        self,
        confirm: bool = False,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        image_paths = self._runtime_loaded_image_paths()
        if not image_paths:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.information(self, self.tr("title_info"), self.tr("msg_no_images"))
            return self._batch_result(reason="no_images", image_count=0)
        if self.is_task_running():
            return self._batch_task_running_result(interactive=allow_dialog)

        if not confirm:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                reply = QMessageBox.question(
                    self,
                    self.tr("title_batch_restore"),
                    self.tr("msg_batch_restore_confirm"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return self._batch_result(reason="cancelled")
                confirm = True
            else:
                return self._batch_result(reason="confirm_required", target_count=len(image_paths))

        self.execute_command("task.run_restore", image_paths)
        return self._batch_result(
            started=True,
            task_started=True,
            mode="restore",
            confirm=bool(confirm),
            target_count=len(image_paths),
            files_to_process_count=len(image_paths),
            files_to_process=list(image_paths),
        )
