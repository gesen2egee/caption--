from typing import TYPE_CHECKING, Optional
import os
import json
import re
from pathlib import Path

import lib.runtime.selection_service as selection_service
from lib.core.settings import save_app_settings
from lib.core.dataclasses import Settings
from lib.utils.file_ops import load_image_sidecar, save_image_sidecar
from lib.utils.boorutag import parse_boorutag_meta
from lib.utils.parsing import extract_bracket_content, smart_parse_tags

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class NavigationMixin:
    """
    Mixin handling file list navigation, loading, image display, and tag/NL data management.
    Expected to be mixed into MainWindow.
    """

    def _unique_path(self, path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        for i in range(1, 9999):
            p2 = f"{base}_{i}{ext}"
            if not os.path.exists(p2):
                return p2
        return path

    def _replace_image_path_in_list(self, old_path: str, new_path: str):
        if not old_path or not new_path or os.path.abspath(old_path) == os.path.abspath(new_path):
            return
        
        # Update current path first if match
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        if current_image_path and os.path.abspath(current_image_path) == os.path.abspath(old_path):
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(current_image_path=new_path)
            else:
                self.current_image_path = new_path

        abs_old = os.path.abspath(old_path)
        next_image_files = list(self.image_files)
        for i, p in enumerate(next_image_files):
            if os.path.abspath(p) == abs_old:
                next_image_files[i] = new_path
                break
        if hasattr(self, "_set_runtime_selection_values"):
            self._set_runtime_selection_values(loaded_image_paths=next_image_files)
        else:
            self.image_files = next_image_files
    
    def _tagger_has_background(self, image_path: str) -> bool:
        """檢查 tagger_tags 是否含有 background"""
        try:
            sidecar = load_image_sidecar(image_path)
            raw = sidecar.get("tagger_tags", "")
            if not raw:
                return False
            return re.search(r"background", raw, re.IGNORECASE) is not None
        except Exception:
            return False

    def _selection_filter_query(self) -> str:
        if hasattr(self, "filter_input") and self.filter_input is not None:
            return self.filter_input.text().strip()
        getter = getattr(self, "_runtime_selection_value", None)
        if callable(getter):
            return str(getter("filter_query", "") or "").strip()
        return ""

    def _selection_filter_tags_enabled(self) -> bool:
        if hasattr(self, "chk_filter_tags") and self.chk_filter_tags is not None:
            return bool(self.chk_filter_tags.isChecked())
        getter = getattr(self, "_runtime_selection_value", None)
        if callable(getter):
            return bool(getter("filter_tags", True))
        return True

    def _selection_filter_text_enabled(self) -> bool:
        if hasattr(self, "chk_filter_text") and self.chk_filter_text is not None:
            return bool(self.chk_filter_text.isChecked())
        getter = getattr(self, "_runtime_selection_value", None)
        if callable(getter):
            return bool(getter("filter_text", False))
        return False

    def _ensure_selection_lists(self) -> None:
        if not getattr(self, "image_files", None):
            getter = getattr(self, "_runtime_loaded_image_paths", None)
            if callable(getter):
                loaded = list(getter())
                if hasattr(self, "_set_runtime_selection_values"):
                    self._set_runtime_selection_values(loaded_image_paths=loaded)
                else:
                    self.image_files = loaded
        if not getattr(self, "all_image_files", None):
            getter = getattr(self, "_runtime_all_image_paths", None)
            if callable(getter):
                all_paths = list(getter())
                if hasattr(self, "_set_runtime_selection_values"):
                    self._set_runtime_selection_values(all_image_paths=all_paths)
                else:
                    self.all_image_files = all_paths
        if getattr(self, "filter_active", False) and not getattr(self, "filtered_image_files", None):
            getter = getattr(self, "_runtime_filtered_image_paths", None)
            if callable(getter):
                filtered = list(getter())
                if hasattr(self, "_set_runtime_selection_values"):
                    self._set_runtime_selection_values(filtered_image_paths=filtered)
                else:
                    self.filtered_image_files = filtered

    def refresh_file_list(self, current_path=None):
        if not self.root_dir_path or not os.path.exists(self.root_dir_path):
            return
        
        if not current_path:
            current_path = self.current_image_path
        discovered_files = selection_service.discover_image_files(self.root_dir_path)
        if hasattr(self, "_set_runtime_selection_values"):
            self._set_runtime_selection_values(loaded_image_paths=discovered_files)
        else:
            self.image_files = discovered_files

        if not self.image_files:
            if hasattr(self, "image_label") and self.image_label is not None:
                self.image_label.clear()
            if hasattr(self, "txt_edit") and self.txt_edit is not None:
                self.txt_edit.clear()
            if hasattr(self, "img_file_label") and self.img_file_label is not None:
                self.img_file_label.setText(self.tr("label_no_image"))
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(current_index=-1, current_image_path="", loaded_image_paths=[])
            else:
                self.current_index = -1
                self.current_image_path = None
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()
            if hasattr(self, "_sync_runtime_content_state"):
                self._sync_runtime_content_state()
            if hasattr(self, "_sync_runtime_tags_state"):
                self._sync_runtime_tags_state()
            return

        if current_path and current_path in self.image_files:
            next_index = self.image_files.index(current_path)
        else:
            next_index = self.current_index
            if next_index >= len(self.image_files):
                next_index = len(self.image_files) - 1
            if next_index < 0:
                next_index = 0
        if hasattr(self, "_set_runtime_selection_values"):
            self._set_runtime_selection_values(current_index=next_index)
        else:
            self.current_index = next_index
        
        self.load_image()
        if hasattr(self, "_sync_runtime_selection_state"):
            self._sync_runtime_selection_state()
        self.statusBar().showMessage(self.tr("msg_refreshed").replace("{count}", str(len(self.image_files))), 3000)

    def open_directory(self):
        from PyQt6.QtWidgets import QFileDialog

        default_dir = self.settings.get("last_open_dir", "")
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("msg_select_dir"), default_dir)
        if dir_path:
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(
                    root_dir_path=dir_path,
                    filter_active=False,
                    all_image_paths=[],
                    filtered_image_paths=[],
                    filter_query="",
                )
            else:
                self.root_dir_path = dir_path
            if hasattr(self, "_runtime_settings_dict") and hasattr(self, "_replace_runtime_settings_dict"):
                new_cfg = self._runtime_settings_dict()
                new_cfg["last_open_dir"] = dir_path
                self._replace_runtime_settings_dict(new_cfg, persist=True)
            else:
                self.settings["last_open_dir"] = dir_path
                save_app_settings(self.settings)

            self.filter_active = False
            self.filter_input.clear()
            self.all_image_files = []
            self.filtered_image_files = []

            self.refresh_file_list()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()

    def load_image(self):
        from PyQt6.QtGui import QPixmap

        if 0 <= self.current_index < len(self.image_files):
            next_image_path = self.image_files[self.current_index]
            next_folder_path = str(Path(next_image_path).parent)
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(
                    current_image_path=next_image_path,
                    current_folder_path=next_folder_path,
                    current_index=self.current_index,
                )
            else:
                self.current_image_path = next_image_path
                self.current_folder_path = next_folder_path

            # Update info bar
            total_count = len(self.filtered_image_files) if self.filter_active else len(self.image_files)
            current_num = self.filtered_image_files.index(self.current_image_path) + 1 if self.filter_active and self.current_image_path in self.filtered_image_files else self.current_index + 1
            
            if hasattr(self, "index_input") and self.index_input is not None:
                self.index_input.blockSignals(True)
                self.index_input.setText(str(current_num))
                self.index_input.blockSignals(False)
            
            if hasattr(self, "total_info_label") and self.total_info_label is not None:
                if self.filter_active:
                    self.total_info_label.setText(f"<span style='color:red;'> / {total_count}</span>")
                else:
                    self.total_info_label.setText(f" / {total_count}")
            
            if hasattr(self, "img_file_label") and self.img_file_label is not None:
                self.img_file_label.setText(f" : {os.path.basename(self.current_image_path)}")

            self.current_pixmap = QPixmap(self.current_image_path)
            if not self.current_pixmap.isNull() and hasattr(self, "image_label") and self.image_label is not None:
                self.update_image_display()
            elif hasattr(self, "image_label") and self.image_label is not None:
                self.image_label.clear()

            content = selection_service.load_text_content(self.current_image_path)
            if hasattr(self, "txt_edit") and self.txt_edit is not None:
                self.txt_edit.blockSignals(True)
                self.txt_edit.setPlainText(content)
                self.txt_edit.blockSignals(False)
            elif hasattr(self, "_update_runtime_state_section"):
                self._update_runtime_state_section("content", {"txt_content": content})
            elif hasattr(self, "_get_runtime_state_store"):
                self._get_runtime_state_store().update_section("content", {"txt_content": content})

            self.top_tags = self.build_top_tags_for_current_image()
            self.custom_tags = self.load_folder_custom_tags(self.current_folder_path)
            self.tagger_tags = self.load_tagger_tags_for_current_image()

            self.nl_pages = self.load_nl_pages_for_image(self.current_image_path)
            if self.nl_pages:
                self.nl_page_index = len(self.nl_pages) - 1
                self.nl_latest = self.nl_pages[self.nl_page_index]
            else:
                self.nl_page_index = 0
                self.nl_latest = ""

            self.refresh_tags_tab()
            self.refresh_nl_tab()
            self.update_nl_page_controls()

            self.on_text_changed()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()
            if hasattr(self, "_sync_runtime_tags_state"):
                self._sync_runtime_tags_state()
            if hasattr(self, "_sync_runtime_content_state"):
                self._sync_runtime_content_state()

    def _get_image_content_for_filter(self, image_path: str) -> str:
        return selection_service.build_filter_content(
            image_path,
            include_tags=self._selection_filter_tags_enabled(),
            include_text=self._selection_filter_text_enabled(),
        )

    def apply_filter(self):
        query = self._selection_filter_query()
        if not query:
            self.clear_filter()
            return
        self._ensure_selection_lists()
        if not self.image_files and not self.all_image_files:
            return
        if not self.all_image_files:
            self.all_image_files = list(self.image_files)
        
        matched = selection_service.filter_image_paths(
            self.all_image_files,
            query,
            include_tags=self._selection_filter_tags_enabled(),
            include_text=self._selection_filter_text_enabled(),
        )
        
        if not matched:
            self.statusBar().showMessage(self.tr("msg_filter_empty"), 3000)
            return
        
        if hasattr(self, "_set_runtime_selection_values"):
            self._set_runtime_selection_values(
                filtered_image_paths=matched,
                loaded_image_paths=matched,
                all_image_paths=self.all_image_files,
                filter_active=True,
                current_index=0,
                filter_query=query,
            )
        else:
            self.filtered_image_files = matched
            self.image_files = matched
            self.filter_active = True
            self.current_index = 0
        self.load_image()
        if hasattr(self, "_sync_runtime_selection_state"):
            self._sync_runtime_selection_state()
        self.statusBar().showMessage(self.tr("msg_filter_result").replace("{count}", str(len(matched))), 3000)

    def clear_filter(self):
        if hasattr(self, "filter_input") and self.filter_input is not None:
            self.filter_input.clear()
        if self.all_image_files:
            current_path = getattr(self, "current_image_path", None) or ""
            restored_paths = list(self.all_image_files)
            
            if current_path and current_path in restored_paths:
                next_index = restored_paths.index(current_path)
            else:
                next_index = 0
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(
                    loaded_image_paths=restored_paths,
                    all_image_paths=[],
                    filtered_image_paths=[],
                    filter_active=False,
                    current_index=next_index,
                    filter_query="",
                )
            else:
                self.image_files = restored_paths
                self.all_image_files = []
                self.filtered_image_files = []
                self.filter_active = False
                self.current_index = next_index
            
            if self.image_files:
                self.load_image()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()
            self.statusBar().showMessage(self.tr("msg_filter_cleared"), 2000)

    def next_image(self):
        self._ensure_selection_lists()
        if self.current_index < len(self.image_files) - 1:
            next_index = self.current_index + 1
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(current_index=next_index)
            else:
                self.current_index = next_index
            self.load_image()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()

    def prev_image(self):
        self._ensure_selection_lists()
        if self.current_index > 0:
            next_index = self.current_index - 1
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(current_index=next_index)
            else:
                self.current_index = next_index
            self.load_image()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()

    def first_image(self):
        self._ensure_selection_lists()
        if self.image_files:
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(current_index=0)
            else:
                self.current_index = 0
            self.load_image()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()

    def last_image(self):
        self._ensure_selection_lists()
        if self.image_files:
            next_index = len(self.image_files) - 1
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(current_index=next_index)
            else:
                self.current_index = next_index
            self.load_image()
            if hasattr(self, "_sync_runtime_selection_state"):
                self._sync_runtime_selection_state()

    def jump_to_index(self, index: Optional[int] = None):
        self._ensure_selection_lists()
        try:
            if index is None:
                if hasattr(self, "index_input") and self.index_input is not None:
                    val = int(self.index_input.text())
                else:
                    getter = getattr(self, "_runtime_controls_value", None)
                    val = int(getter("current_index", 1)) if callable(getter) else 1
            else:
                val = int(index)
            target_idx = val - 1
            
            if self.filter_active:
                if 0 <= target_idx < len(self.filtered_image_files):
                    target_path = self.filtered_image_files[target_idx]
                    next_index = self.image_files.index(target_path)
                    if hasattr(self, "_set_runtime_selection_values"):
                        self._set_runtime_selection_values(current_index=next_index)
                    else:
                        self.current_index = next_index
                    self.load_image()
                else:
                    self.load_image()
            else:
                if 0 <= target_idx < len(self.image_files):
                    if hasattr(self, "_set_runtime_selection_values"):
                        self._set_runtime_selection_values(current_index=target_idx)
                    else:
                        self.current_index = target_idx
                    self.load_image()
                else:
                    self.load_image()
        except Exception:
            self.load_image()
        if hasattr(self, "_sync_runtime_selection_state"):
            self._sync_runtime_selection_state()

    def update_image_display(self):
        from PyQt6.QtCore import Qt

        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return
        scaled = self._get_processed_pixmap().scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def _get_processed_pixmap(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap

        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return QPixmap()

        mode = self.current_view_mode
        if self.temp_view_mode is not None:
            mode = self.temp_view_mode

        # 0=Original (with mask overlay), 1=RGB, 2=Alpha
        if mode == 0:
            sidecar = load_image_sidecar(self.current_image_path)
            rel_mask = sidecar.get("mask_map_rel_path", "")
            if rel_mask:
                mask_abs = os.path.normpath(os.path.join(os.path.dirname(self.current_image_path), rel_mask))
                if os.path.exists(mask_abs):
                    img_q = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                    mask_q = QImage(mask_abs).convertToFormat(QImage.Format.Format_Alpha8)
                    if img_q.size() == mask_q.size():
                        img_q.setAlphaChannel(mask_q)
                        return QPixmap.fromImage(img_q)
            return self.current_pixmap
        
        img = self.current_pixmap.toImage()
        if mode == 1: 
            img = img.convertToFormat(QImage.Format.Format_RGB888)
            return QPixmap.fromImage(img)
        elif mode == 2: 
            if img.hasAlphaChannel():
                alpha_img = img.convertToFormat(QImage.Format.Format_Alpha8)
                ptr = alpha_img.constBits()
                ptr.setsize(alpha_img.sizeInBytes())
                gray_img = QImage(ptr, alpha_img.width(), alpha_img.height(), alpha_img.bytesPerLine(), QImage.Format.Format_Grayscale8)
                return QPixmap.fromImage(gray_img.copy())
            else:
                white = QPixmap(img.size())
                white.fill(Qt.GlobalColor.white)
                return white
        return self.current_pixmap

    def on_view_mode_changed(self, index):
        self.current_view_mode = index
        self.update_image_display()
        if hasattr(self, "_sync_runtime_ui_state"):
            self._sync_runtime_ui_state()

    def show_image_context_menu(self, pos):
        from PyQt6.QtGui import QAction, QDesktopServices
        from PyQt6.QtWidgets import QApplication, QMenu

        if not self.current_image_path:
            return
        menu = QMenu(self)
        
        action_copy_img = QAction(self.tr("ctx_copy_image"), self)
        action_copy_img.triggered.connect(self._ctx_copy_image)
        menu.addAction(action_copy_img)
        
        action_copy_path = QAction(self.tr("ctx_copy_path"), self)
        action_copy_path.triggered.connect(self._ctx_copy_path)
        menu.addAction(action_copy_path)
        
        menu.addSeparator()
        
        action_open_dir = QAction(self.tr("ctx_open_folder"), self)
        action_open_dir.triggered.connect(self._ctx_open_folder)
        menu.addAction(action_open_dir)
        
        menu.exec(self.image_label.mapToGlobal(pos))

    def _ctx_copy_image(self):
        from PyQt6.QtWidgets import QApplication

        if hasattr(self, 'current_pixmap') and not self.current_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.current_pixmap)
            self.statusBar().showMessage(self.tr("msg_copied_image"), 2000)

    def _ctx_copy_path(self):
        from PyQt6.QtWidgets import QApplication

        if self.current_image_path:
            QApplication.clipboard().setText(os.path.abspath(self.current_image_path))
            self.statusBar().showMessage(self.tr("msg_copied_path"), 2000)
    
    def _ctx_open_folder(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        if self.current_image_path:
            folder = os.path.dirname(self.current_image_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def delete_current_image(self, require_confirmation: bool = True):
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        if not current_image_path:
            return {
                "deleted": False,
                "reason": "no_current_image",
            }

        if require_confirmation:
            from PyQt6.QtWidgets import QMessageBox

            reply = QMessageBox.question(
                self,
                self.tr("title_confirm"),
                self.tr("msg_delete_confirm"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return {
                    "deleted": False,
                    "reason": "cancelled",
                    "source_image": current_image_path,
                }

        delete_result = selection_service.delete_image_bundle(current_image_path)
        moved_files = list(delete_result.get("moved_files", []) or [])
        move_errors = list(delete_result.get("move_errors", []) or [])
        no_used_dir = str(delete_result.get("destination_dir", "") or "")

        abs_current = os.path.abspath(current_image_path)
        next_image_files = [path for path in self.image_files if os.path.abspath(path) != abs_current]
        next_filtered_files = [
            path for path in self.filtered_image_files if os.path.abspath(path) != abs_current
        ]
        next_all_files = [
            path for path in self.all_image_files if os.path.abspath(path) != abs_current
        ]
        next_index = self.current_index
        if next_index >= len(next_image_files):
            next_index -= 1
        if hasattr(self, "_set_runtime_selection_values"):
            self._set_runtime_selection_values(
                loaded_image_paths=next_image_files,
                filtered_image_paths=next_filtered_files,
                all_image_paths=next_all_files,
                current_index=next_index,
            )
        else:
            self.image_files = next_image_files
            self.filtered_image_files = next_filtered_files
            self.all_image_files = next_all_files
            self.current_index = next_index

        if self.image_files:
            self.load_image()
        else:
            if hasattr(self, "_set_runtime_selection_values"):
                self._set_runtime_selection_values(
                    current_index=-1,
                    current_image_path="",
                    current_folder_path="",
                    loaded_image_paths=[],
                    filtered_image_paths=[],
                    all_image_paths=[],
                )
            else:
                self.current_index = -1
                self.current_image_path = ""
                self.current_folder_path = ""
            self.top_tags = []
            self.custom_tags = []
            self.tagger_tags = []
            self.nl_pages = []
            self.nl_page_index = 0
            self.nl_latest = ""
            if hasattr(self, "image_label") and self.image_label is not None:
                self.image_label.clear()
            if hasattr(self, "txt_edit") and self.txt_edit is not None:
                self.txt_edit.clear()
            if hasattr(self, "img_file_label") and self.img_file_label is not None:
                self.img_file_label.setText(self.tr("label_no_image"))
            if hasattr(self, "index_input") and self.index_input is not None:
                self.index_input.blockSignals(True)
                self.index_input.setText("0")
                self.index_input.blockSignals(False)
            if hasattr(self, "total_info_label") and self.total_info_label is not None:
                self.total_info_label.setText(" / 0")
            self.refresh_tags_tab()
            self.refresh_nl_tab()
            self.update_nl_page_controls()

        if hasattr(self, "_sync_runtime_selection_state"):
            self._sync_runtime_selection_state()
        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()
        if hasattr(self, "_sync_runtime_tags_state"):
            self._sync_runtime_tags_state()
        if hasattr(self, "_sync_runtime_ui_state"):
            self._sync_runtime_ui_state()
        if hasattr(self, "_sync_runtime_controls_state"):
            self._sync_runtime_controls_state()

        return {
            "deleted": bool(delete_result.get("deleted", True)),
            "source_image": str(delete_result.get("source_image", current_image_path) or current_image_path),
            "destination_dir": no_used_dir,
            "moved_file_count": len(moved_files),
            "moved_files": moved_files,
            "move_error_count": len(move_errors),
            "move_errors": move_errors,
            "remaining_images": len(self.image_files),
            "current_image_path": str(self.current_image_path or ""),
        }

    # ==========================
    # DATA LOADING
    # ==========================
    def build_top_tags_for_current_image(self):
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        if not current_image_path:
            return []
        hints = []
        tags_from_meta = []
        meta_path = str(current_image_path) + ".boorutag"
        if os.path.isfile(meta_path):
            tags_meta, hint_info = parse_boorutag_meta(meta_path)
            tags_from_meta.extend(tags_meta)
            hints.extend(hint_info)

        parent = Path(current_image_path).parent.name
        if "_" in parent:
            folder_hint = parent.split("_", 1)[1]
            if "{" not in folder_hint:
                folder_hint = f"{{{folder_hint}}}"
            hints.append(folder_hint)

        initial_keywords = []
        for h in hints:
            initial_keywords.extend(extract_bracket_content(h))

        combined = initial_keywords + tags_from_meta
        seen = set()
        final_list = [x for x in combined if not (x in seen or seen.add(x))]
        final_list = [str(t).replace("_", " ").strip() for t in final_list if str(t).strip()]
        if self.english_force_lowercase:
            final_list = [t.lower() for t in final_list]
        return final_list

    def folder_custom_tags_path(self, folder_path):
        return os.path.join(folder_path, ".custom_tags.json")

    def load_folder_custom_tags(self, folder_path):
        p = self.folder_custom_tags_path(folder_path)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tags = data.get("custom_tags", [])
                tags = [str(t).strip() for t in tags if str(t).strip()]
                if not tags:
                    tags = list(self.default_custom_tags_global)
                return tags
            except Exception:
                return list(self.default_custom_tags_global)
        else:
            tags = list(self.default_custom_tags_global)
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"custom_tags": tags}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return tags

    def save_folder_custom_tags(self, folder_path, tags):
        p = self.folder_custom_tags_path(folder_path)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"custom_tags": tags}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_custom_tag(self, tag: str):
        if not self.current_folder_path:
            return False
        tag = str(tag).strip()
        if not tag:
            return False
        tag = tag.replace("_", " ").strip()
        if self.english_force_lowercase:
            tag = tag.lower()

        tags = list(self.custom_tags)
        if tag not in tags:
            tags.append(tag)
            self.custom_tags = tags
            self.save_folder_custom_tags(self.current_folder_path, tags)
            self.refresh_tags_tab()
            self.on_text_changed()
            if hasattr(self, "_sync_runtime_content_state"):
                self._sync_runtime_content_state()
            if hasattr(self, "_sync_runtime_tags_state"):
                self._sync_runtime_tags_state()
            return True
        return False

    def add_custom_tag_dialog(self):
        from PyQt6.QtWidgets import QInputDialog

        if not self.current_folder_path:
            return
        tag, ok = QInputDialog.getText(self, self.tr("dialog_add_tag_title"), self.tr("dialog_add_tag_label"))
        if not ok:
            return
        self.add_custom_tag(tag)

    def load_tagger_tags_for_current_image(self):
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        if not current_image_path:
            return []
        sidecar = load_image_sidecar(current_image_path)
        raw = sidecar.get("tagger_tags", "")
        if not raw:
            return []
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        parts = [t.replace("_", " ").strip() for t in parts]
        if self.english_force_lowercase:
            parts = [t.lower() for t in parts]
        return parts

    def save_tagger_tags_for_image(self, image_path, raw_tags_str):
        sidecar = load_image_sidecar(image_path)
        sidecar["tagger_tags"] = raw_tags_str
        save_image_sidecar(image_path, sidecar)

    def load_nl_pages_for_image(self, image_path):
        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if isinstance(pages, list):
            return [p for p in pages if p and str(p).strip()]
        return []

    def load_nl_for_current_image(self):
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        if not current_image_path:
            return ""
        pages = self.load_nl_pages_for_image(current_image_path)
        return pages[-1] if pages else ""

    def save_nl_for_image(self, image_path, content):
        if not content:
            return
        content = str(content).strip()
        if not content:
            return

        sidecar = load_image_sidecar(image_path)
        pages = sidecar.get("nl_pages", [])
        if not isinstance(pages, list):
            pages = []
        pages.append(content)
        sidecar["nl_pages"] = pages
        save_image_sidecar(image_path, sidecar)

    def refresh_tags_tab(self):
        active_text = self._runtime_txt_content() if hasattr(self, "_runtime_txt_content") else (self.txt_edit.toPlainText() if hasattr(self, "txt_edit") and self.txt_edit is not None else "")

        if hasattr(self, "flow_top") and self.flow_top is not None:
            self.flow_top.render_tags_flow(
                smart_parse_tags(", ".join(self.top_tags)),
                active_text,
                self.settings
            )
        if hasattr(self, "flow_custom") and self.flow_custom is not None:
            self.flow_custom.render_tags_flow(
                smart_parse_tags(", ".join(self.custom_tags)),
                active_text,
                self.settings
            )
        if hasattr(self, "flow_tagger") and self.flow_tagger is not None:
            self.flow_tagger.render_tags_flow(
                smart_parse_tags(", ".join(self.tagger_tags)),
                active_text,
                self.settings
            )
        if hasattr(self, "_sync_runtime_tags_state"):
            self._sync_runtime_tags_state()

    def refresh_nl_tab(self):
        active_text = self._runtime_txt_content() if hasattr(self, "_runtime_txt_content") else (self.txt_edit.toPlainText() if hasattr(self, "txt_edit") and self.txt_edit is not None else "")
        if hasattr(self, "flow_nl") and self.flow_nl is not None:
            self.flow_nl.render_tags_flow(
                smart_parse_tags(self.nl_latest),
                active_text,
                self.settings
            )
        if hasattr(self, "_sync_runtime_tags_state"):
            self._sync_runtime_tags_state()

    def set_current_nl_page(self, idx: int):
        if not self.nl_pages:
            self.nl_page_index = 0
            self.nl_latest = ""
            self.refresh_nl_tab()
            self.update_nl_page_controls()
            if hasattr(self, "_sync_runtime_ui_state"):
                self._sync_runtime_ui_state()
            if hasattr(self, "_sync_runtime_content_state"):
                self._sync_runtime_content_state()
            return

        idx = max(0, min(int(idx), len(self.nl_pages) - 1))
        self.nl_page_index = idx
        self.nl_latest = self.nl_pages[self.nl_page_index]

        self.refresh_nl_tab()
        self.update_nl_page_controls()
        self.on_text_changed()
        if hasattr(self, "_sync_runtime_ui_state"):
            self._sync_runtime_ui_state()
        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()

    def update_nl_page_controls(self):
        total = len(self.nl_pages)
        if total <= 0:
            if hasattr(self, "nl_page_label") and self.nl_page_label is not None:
                self.nl_page_label.setText(f"{self.tr('label_page')} 0/0")
            if hasattr(self, "btn_prev_nl") and self.btn_prev_nl is not None:
                self.btn_prev_nl.setEnabled(False)
            if hasattr(self, "btn_next_nl") and self.btn_next_nl is not None:
                self.btn_next_nl.setEnabled(False)
        else:
            self.nl_page_index = max(0, min(self.nl_page_index, total - 1))
            if hasattr(self, "nl_page_label") and self.nl_page_label is not None:
                txt = self.tr("label_page_fmt").replace("{current}", str(self.nl_page_index + 1)).replace("{total}", str(total))
                self.nl_page_label.setText(txt)
            if hasattr(self, "btn_prev_nl") and self.btn_prev_nl is not None:
                self.btn_prev_nl.setEnabled(self.nl_page_index > 0)
            if hasattr(self, "btn_next_nl") and self.btn_next_nl is not None:
                self.btn_next_nl.setEnabled(self.nl_page_index < total - 1)

        self.update_nl_result_height()
        if hasattr(self, "_sync_runtime_ui_state"):
            self._sync_runtime_ui_state()

    def prev_nl_page(self):
        if self.nl_pages and self.nl_page_index > 0:
            self.set_current_nl_page(self.nl_page_index - 1)

    def next_nl_page(self):
        if self.nl_pages and self.nl_page_index < len(self.nl_pages) - 1:
            self.set_current_nl_page(self.nl_page_index + 1)

    def update_nl_result_height(self):
        try:
            lines = [l for l in (self.nl_latest or "").splitlines() if l.strip()]
            n = len(lines)
            if n >= 16:
                self.flow_nl.setMinimumHeight(760)
                self.prompt_edit.setMaximumHeight(220)
            elif n >= 10:
                self.flow_nl.setMinimumHeight(660)
                self.prompt_edit.setMaximumHeight(280)
            else:
                self.flow_nl.setMinimumHeight(520)
                self.prompt_edit.setMaximumHeight(9999)
        except Exception:
            pass
