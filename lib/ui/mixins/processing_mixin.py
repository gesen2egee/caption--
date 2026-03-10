from typing import TYPE_CHECKING, Any, Dict, Optional
import os

import lib.runtime.processing_service as processing_service

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class ProcessingMixin:
    """
    Mixin handling single-image processing actions:
    Tagger, LLM, Image Process, Unmask, Mask Text, Restore, Stroke Eraser.
    """

    def _processing_result(self, **kwargs) -> Dict[str, Any]:
        return processing_service.build_processing_result(**kwargs)

    def _qimage_to_png_bytes(self, qimg: Any) -> bytes:
        from PyQt6.QtCore import QBuffer, QByteArray, QIODevice

        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not qimg.save(buffer, "PNG"):
            raise ValueError("failed to encode stroke mask png")
        return bytes(ba.data())

    def _load_stroke_mask_image(
        self,
        mask_png_base64: Optional[str] = None,
        mask_png_bytes: Optional[bytes] = None,
        mask_path: Optional[str] = None,
    ) -> tuple[Any, str, str]:
        return processing_service.load_stroke_mask(
            mask_png_base64=mask_png_base64,
            mask_png_bytes=mask_png_bytes,
            mask_path=mask_path,
        )

    def _apply_stroke_eraser_mask(
        self,
        image_path: str,
        mask_image: Any,
        mask_source: str,
        mask_path: str = "",
    ) -> Dict[str, Any]:
        old_path = image_path
        result = processing_service.apply_stroke_eraser(
            old_path,
            mask_image,
            shrink=self.settings.get("mask_text_shrink_size", 1),
            blur=self.settings.get("mask_text_blur_radius", 3),
            min_alpha=self.settings.get("mask_text_min_alpha", 0),
            unique_path_resolver=self._unique_path,
        )
        new_path = str(result.get("output_image_path", "") or "")
        if not new_path:
            return self._processing_result(
                reason="no_output",
                mode="stroke_eraser",
                image_path=old_path,
                mask_source=mask_source,
            )

        self._replace_image_path_in_list(old_path, new_path)
        self.load_image()
        if hasattr(self, "_sync_runtime_selection_state"):
            self._sync_runtime_selection_state()
        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()
        self.statusBar().showMessage(self.tr("status_stroke_done"), 5000)

        payload = self._processing_result(
            started=True,
            completed=True,
            mode="stroke_eraser",
            image_path=new_path,
            mask_source=mask_source,
            mask_path=mask_path,
        )
        payload.update(result)
        event_bus_getter = getattr(self, "_get_runtime_event_bus", None)
        if callable(event_bus_getter):
            event_bus_getter().emit("image.stroke_erased", payload)
        return payload

    def command_run_tagger_current(self, allow_dialog: bool = False) -> Dict[str, Any]:
        action = processing_service.plan_current_tagger_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
        )
        result = dict(action["result"])
        if not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    self.tr("title_warning"),
                    self.tr("msg_no_image_selected" if result.get("reason") == "no_current_image" else "msg_task_running"),
                )
            return result

        current_image_path = str(result.get("image_path", "") or "")

        self.btn_auto_tag.setEnabled(False)
        self.btn_auto_tag.setText(self.tr("btn_txt_tagging"))
        self.statusBar().showMessage(f"{self.tr('status_tagging')} {os.path.basename(current_image_path)}...")

        self.execute_command("task.run_tagger", [current_image_path])
        return result

    def command_run_llm_current(
        self,
        user_prompt: Optional[str] = None,
        confirm_missing_tags: bool = False,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        action = processing_service.plan_current_llm_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
            user_prompt=user_prompt,
            runtime_prompt=self._runtime_prompt_text(),
            confirm_missing_tags=confirm_missing_tags,
        )
        result = dict(action["result"])
        if not action["ready"] and action.get("requires_confirmation") == "confirm_missing_tags":
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                reply = QMessageBox.question(
                    self,
                    self.tr("title_warning"),
                    self.tr("msg_confirm_prompt_tags"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return self._processing_result(
                        reason="cancelled",
                        image_path=result.get("image_path"),
                        prompt_contains_tags_placeholder=True,
                    )
                action = processing_service.plan_current_llm_action(
                    self._runtime_current_image_path(),
                    task_running=self.is_task_running(),
                    user_prompt=user_prompt,
                    runtime_prompt=self._runtime_prompt_text(),
                    confirm_missing_tags=True,
                )
                result = dict(action["result"])
            else:
                return result
        elif not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    self.tr("title_warning"),
                    self.tr("msg_no_image_selected" if result.get("reason") == "no_current_image" else "msg_task_running"),
                )
            return result

        current_image_path = str(result.get("image_path", "") or "")
        resolved_user_prompt = str(result.get("user_prompt", "") or "")

        self.btn_run_llm.setEnabled(False)
        self.btn_run_llm.setText(self.tr("btn_txt_running_llm"))
        self.execute_command("task.run_llm", [current_image_path], user_prompt=resolved_user_prompt)
        return result

    def command_run_image_process_current(
        self,
        edit_prompt: Optional[str] = None,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        action = processing_service.plan_current_image_process_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
            explicit_prompt=edit_prompt,
            runtime_prompt=self._runtime_image_process_prompt_text(),
            default_prompt=self.settings.get("image_process_prompt_template", "幫我移除圖中所有的文字、文字氣泡、文字框"),
        )
        result = dict(action["result"])
        if not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected" if result.get("reason") == "no_current_image" else "msg_task_running"))
            return result

        current_image_path = str(result.get("image_path", "") or "")
        prompt = str(action.get("resolved_prompt", "") or result.get("edit_prompt", "") or "")

        self.btn_run_imgproc.setEnabled(False)
        self.btn_run_imgproc.setText(self.tr("btn_txt_image_processing"))
        self.statusBar().showMessage(self.tr("status_image_processing"), 2000)
        self.execute_command("task.run_image_process", [current_image_path], edit_prompt=prompt)
        return result

    def command_run_unmask_current(self, allow_dialog: bool = False) -> Dict[str, Any]:
        action = processing_service.plan_current_unmask_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
        )
        result = dict(action["result"])
        if not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected" if result.get("reason") == "no_current_image" else "msg_task_running"))
            return result

        current_image_path = str(result.get("image_path", "") or "")

        self.execute_command("task.run_unmask", [current_image_path])
        self.statusBar().showMessage(self.tr("status_unmasking"), 2000)
        return result

    def command_run_mask_text_current(self, allow_dialog: bool = False) -> Dict[str, Any]:
        action = processing_service.plan_current_mask_text_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
            ocr_enabled=bool(self.settings.get("mask_batch_detect_text_enabled", True)),
        )
        result = dict(action["result"])
        if not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                if result.get("reason") == "ocr_disabled":
                    QMessageBox.information(self, self.tr("title_info"), self.tr("msg_ocr_disabled"))
                elif result.get("reason") == "no_current_image":
                    QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected"))
                elif result.get("reason") == "task_running":
                    QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return result

        current_image_path = str(result.get("image_path", "") or "")

        self.execute_command("task.run_mask_text", [current_image_path])
        self.statusBar().showMessage(self.tr("status_masking_text"), 2000)
        return result

    def command_run_restore_current(self, allow_dialog: bool = False) -> Dict[str, Any]:
        action = processing_service.plan_current_restore_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
        )
        result = dict(action["result"])
        if not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                if result.get("reason") == "no_current_image":
                    QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected"))
                elif result.get("reason") == "no_backup":
                    QMessageBox.information(self, self.tr("title_restore"), self.tr("msg_restore_no_backup"))
                else:
                    QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return result

        current_image_path = str(result.get("image_path", "") or "")

        self.execute_command("task.run_restore", [current_image_path])
        self.statusBar().showMessage(self.tr("status_restoring"), 2000)
        return result

    def command_run_stroke_eraser_current(
        self,
        mask_png_base64: Optional[str] = None,
        mask_path: Optional[str] = None,
        allow_dialog: bool = False,
    ) -> Dict[str, Any]:
        action = processing_service.plan_current_stroke_eraser_action(
            self._runtime_current_image_path(),
            task_running=self.is_task_running(),
        )
        result = dict(action["result"])
        if not action["ready"]:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected" if result.get("reason") == "no_current_image" else "msg_task_running"))
            return result

        current_image_path = str(result.get("image_path", "") or "")

        try:
            mask_image, mask_source, resolved_mask_path = self._load_stroke_mask_image(
                mask_png_base64=mask_png_base64,
                mask_path=mask_path,
            )
        except Exception as exc:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, self.tr("title_stroke_eraser"), f"{self.tr('msg_stroke_load_err')}{exc}")
            return self._processing_result(
                reason="invalid_mask",
                mode="stroke_eraser",
                image_path=current_image_path,
                error=str(exc),
            )

        try:
            return self._apply_stroke_eraser_mask(
                current_image_path,
                mask_image,
                mask_source=mask_source,
                mask_path=resolved_mask_path,
            )
        except Exception as exc:
            if allow_dialog:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, self.tr("title_stroke_eraser"), f"失敗: {exc}")
            return self._processing_result(
                reason="failed",
                mode="stroke_eraser",
                image_path=current_image_path,
                error=str(exc),
                mask_source=mask_source,
                mask_path=resolved_mask_path,
            )

    def run_tagger(self):
        try:
            self.execute_command("action.run_tagger_current", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_llm_generation(self):
        try:
            self.execute_command("action.run_llm_current", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_image_processing(self):
        try:
            self.execute_command("action.run_image_process_current", allow_dialog=True)
        except Exception as e:
            self.on_pipeline_error(str(e))

    def unmask_current_image(self):
        try:
            self.execute_command("action.run_unmask_current", allow_dialog=True)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, self.tr("title_error"), f"{self.tr('msg_unmask_failed')}{e}")

    def mask_text_current_image(self):
        try:
             self.execute_command("action.run_mask_text_current", allow_dialog=True)
        except Exception as e:
             from PyQt6.QtWidgets import QMessageBox

             QMessageBox.warning(self, self.tr("title_error"), f"{self.tr('msg_failed')}{e}")

    def restore_current_image(self):
        """還原當前圖片為原始備份 (從 raw_image 資料夾)"""
        try:
            self.execute_command("action.run_restore_current", allow_dialog=True)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, self.tr("title_error"), f"{self.tr('msg_restore_failed')}{e}")

    def open_stroke_eraser(self):
        from PyQt6.QtWidgets import QDialog, QMessageBox

        from lib.ui.components.stroke import StrokeEraseDialog

        current_image_path = self._runtime_current_image_path()
        if not current_image_path:
            return
        try:
            dlg = StrokeEraseDialog(current_image_path, self)
        except Exception as e:
            QMessageBox.warning(self, self.tr("title_stroke_eraser"), f"{self.tr('msg_stroke_load_err')}{e}")
            return

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        mask_qimg, _w = dlg.get_result()
        try:
            mask_image, _mask_source, _resolved_mask_path = self._load_stroke_mask_image(
                mask_png_bytes=self._qimage_to_png_bytes(mask_qimg),
            )
            self._apply_stroke_eraser_mask(
                current_image_path,
                mask_image,
                mask_source="dialog",
            )
        except Exception as e:
            QMessageBox.warning(self, self.tr("title_stroke_eraser"), f"失敗: {e}")
