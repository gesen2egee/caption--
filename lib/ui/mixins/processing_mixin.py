from typing import TYPE_CHECKING
import os
import shutil
from io import BytesIO

from PyQt6.QtWidgets import QMessageBox, QDialog
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt, QBuffer, QIODevice, QByteArray

from PIL import Image, ImageChops

from lib.utils.file_ops import create_image_data_from_path, has_raw_backup, backup_raw_image
from lib.utils.tag_context import build_llm_tags_context_for_image
from lib.ui.components.stroke import StrokeEraseDialog
from lib.pipeline.tasks import TaggerTask, LLMTask, UnmaskTask, MaskTextTask, RestoreTask

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class ProcessingMixin:
    """
    Mixin handling single-image processing actions:
    Tagger, LLM, Unmask, Mask Text, Restore, Stroke Eraser.
    """

    def run_tagger(self):
        if not self.current_image_path:
            return
        
        if self.is_task_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return
            
        self.btn_auto_tag.setEnabled(False)
        self.btn_auto_tag.setText(self.tr("btn_txt_tagging"))
        self.statusBar().showMessage(f"{self.tr('status_tagging')} {os.path.basename(self.current_image_path)}...")
        
        try:
            image_data = create_image_data_from_path(self.current_image_path)
            self.run_task(TaggerTask, [image_data])
        except Exception as e:
            self.on_pipeline_error(str(e))

    def run_llm_generation(self):
        if not self.current_image_path:
            return
            
        if self.is_task_running():
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
            return

        tags_text = build_llm_tags_context_for_image(self.current_image_path)
        user_prompt = self.prompt_edit.toPlainText()

        if "{tags}" in user_prompt and not tags_text.strip():
            reply = QMessageBox.question(
                self, self.tr("title_warning"), 
                self.tr("msg_confirm_prompt_tags"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.btn_run_llm.setEnabled(False)
        self.btn_run_llm.setText(self.tr("btn_txt_running_llm"))
        
        try:
            image_data = create_image_data_from_path(self.current_image_path)
            self.run_task(LLMTask, [image_data], extra={"user_prompt": user_prompt})
        except Exception as e:
            self.on_pipeline_error(str(e))

    def unmask_current_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected"))
            return
            
        if self.is_task_running():
             QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
             return

        try:
            self.run_task(UnmaskTask, [create_image_data_from_path(self.current_image_path)])
            self.statusBar().showMessage(self.tr("status_unmasking"), 2000)
        except Exception as e:
            QMessageBox.warning(self, self.tr("title_error"), f"{self.tr('msg_unmask_failed')}{e}")

    def mask_text_current_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_no_image_selected"))
            return
            
        if not self.settings.get("mask_batch_detect_text_enabled", True):
             QMessageBox.information(self, self.tr("title_info"), self.tr("msg_ocr_disabled"))
             return
             
        if self.is_task_running():
             QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
             return

        try:
             self.run_task(MaskTextTask, [create_image_data_from_path(self.current_image_path)])
             self.statusBar().showMessage(self.tr("status_masking_text"), 2000)
        except Exception as e:
             QMessageBox.warning(self, self.tr("title_error"), f"{self.tr('msg_failed')}{e}")

    def restore_current_image(self):
        """還原當前圖片為原始備份 (從 raw_image 資料夾)"""
        if not self.current_image_path:
            return
        
        if not has_raw_backup(self.current_image_path):
            QMessageBox.information(self, self.tr("title_restore"), self.tr("msg_restore_no_backup"))
            return
            
        if self.is_task_running():
             QMessageBox.warning(self, self.tr("title_warning"), self.tr("msg_task_running"))
             return

        try:
            self.run_task(RestoreTask, [create_image_data_from_path(self.current_image_path)])
            self.statusBar().showMessage(self.tr("status_restoring"), 2000)
        except Exception as e:
            QMessageBox.warning(self, self.tr("title_error"), f"{self.tr('msg_restore_failed')}{e}")

    @staticmethod
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

    def stroke_erase_to_webp(self, image_path: str, mask_qimg: QImage) -> str:
        if not image_path:
            return ""

        from lib.utils.sidecar import load_image_sidecar, save_image_sidecar

        src_dir = os.path.dirname(image_path)
        unmask_dir = os.path.join(src_dir, "unmask")
        os.makedirs(unmask_dir, exist_ok=True)

        ext = os.path.splitext(image_path)[1].lower()
        base_no_ext = os.path.splitext(image_path)[0]

        target_file = base_no_ext + ".webp"
        # Change: Allow overwrite of existing webp
        # if os.path.exists(target_file) and os.path.abspath(target_file) != os.path.abspath(image_path):
        #    target_file = self._unique_path(target_file)

        moved_original = ""
        if ext == ".webp":
            # If editing webp, move original to backup and overwrite current
            moved_original = self._unique_path(os.path.join(unmask_dir, os.path.basename(image_path)))
            shutil.copy2(image_path, moved_original) # Use copy then delete/overwrite to be safe
            # Actually shutil.move is fine if we are about to overwrite 'image_path' which is 'target_file'
            # But wait, we save to 'target_file' later.
            # If we move 'image_path' away, 'target_file' (same path) is free.
            shutil.move(image_path, moved_original)
            
            src_for_processing = moved_original
            target_file = image_path
        else:
            src_for_processing = image_path
            # Check if target_file exists, maybe backup it too?
            # If overwriting an existing webp (not the source image), maybe we should backup that webp?
            if os.path.exists(target_file):
                bkp_existing = self._unique_path(os.path.join(unmask_dir, os.path.basename(target_file)))
                shutil.move(target_file, bkp_existing)

        from PIL import ImageChops
        from lib.utils.image_processing import process_mask_channel

        with Image.open(src_for_processing) as img:
            img_rgba = img.convert("RGBA")
            mask_pil = self._qimage_to_pil_l(mask_qimg)
            # resize to original size
            mask_pil = mask_pil.resize(img_rgba.size, Image.Resampling.NEAREST)

            # mask_pil: 255=Painted(Remove), 0=Keep
            # Create Keep Mask: 0=Remove, 255=Keep
            keep = Image.eval(mask_pil, lambda v: 0 if v > 0 else 255)
            
            # Apply Advanced Processing to the "Keep Mask" using Text Settings
            # User request: "手繪 ... (用文字的設定值)"
            keep_processed = process_mask_channel(
                keep, 
                shrink=self.settings.get("mask_text_shrink_size", 1),
                blur=self.settings.get("mask_text_blur_radius", 3),
                min_alpha=self.settings.get("mask_text_min_alpha", 0)
            )

            # Combine with Original Alpha
            alpha = img_rgba.getchannel("A")
            new_alpha = ImageChops.multiply(alpha, keep_processed)
            
            # "alpha=0 像素填補白色" logic (Based on ORIGINAL Alpha)
            # Logic: If Original Alpha is 0, set RGB to White.
            # Otherwise, keep Original RGB (Don't affect RGB of erased stroked area).
            # And use New Alpha.
            
            datas = img_rgba.getdata() # (r, g, b, old_a)
            new_alpha_data = new_alpha.getdata()
            
            new_combined_data = []
            # Optimization: Use zip for iteration
            for item, new_a in zip(datas, new_alpha_data):
                # item: (r, g, b, old_a)
                if item[3] == 0:
                    # Original was transparent -> Fill White
                    new_combined_data.append((255, 255, 255, new_a))
                else:
                    # Original had content -> Keep RGB, use New Alpha
                    new_combined_data.append((item[0], item[1], item[2], new_a))
            
            img_rgba.putdata(new_combined_data)
            
            img_rgba.save(target_file, "WEBP")

        if ext != ".webp":
            moved_original = self._unique_path(os.path.join(unmask_dir, os.path.basename(image_path)))
            shutil.move(image_path, moved_original)
            
            # Handle Sidecar: Copy to new webp, and move old json to backup
            old_json = image_path + ".json"
            if os.path.exists(old_json):
                 sc = load_image_sidecar(image_path)
                 save_image_sidecar(target_file, sc)
                 
                 backup_json = moved_original + ".json"
                 shutil.move(old_json, backup_json)

        return target_file

    def open_stroke_eraser(self):
        if not self.current_image_path:
            return
        try:
            dlg = StrokeEraseDialog(self.current_image_path, self)
        except Exception as e:
            QMessageBox.warning(self, self.tr("title_stroke_eraser"), f"{self.tr('msg_stroke_load_err')}{e}")
            return

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        mask_qimg, _w = dlg.get_result()
        try:
            # Backup Raw Image if needed
            backup_raw_image(self.current_image_path)
            
            old_path = self.current_image_path
            new_path = self.stroke_erase_to_webp(old_path, mask_qimg)
            if not new_path:
                return
            self._replace_image_path_in_list(old_path, new_path)
            self.load_image()
            self.statusBar().showMessage(self.tr("status_stroke_done"), 5000)
        except Exception as e:
            QMessageBox.warning(self, self.tr("title_stroke_eraser"), f"失敗: {e}")
