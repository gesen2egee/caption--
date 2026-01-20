from typing import TYPE_CHECKING
import os
import shutil
import json
import re
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication, QMenu, QInputDialog, QFrame
from PyQt6.QtCore import Qt, QUrl, QTimer, QPoint, QBuffer, QIODevice, QByteArray
from PyQt6.QtGui import QPixmap, QImage, QDesktopServices, QAction, QBrush, QPalette
from natsort import natsorted

from lib.core.settings import save_app_settings
from lib.core.dataclasses import Settings
from lib.utils.file_ops import load_image_sidecar, save_image_sidecar
from lib.utils.query_filter import DanbooruQueryFilter
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
        if self.current_image_path and os.path.abspath(self.current_image_path) == os.path.abspath(old_path):
            self.current_image_path = new_path

        found = False
        abs_old = os.path.abspath(old_path)
        for i, p in enumerate(self.image_files):
            if os.path.abspath(p) == abs_old:
                self.image_files[i] = new_path
                found = True
                break
    
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

    def refresh_file_list(self, current_path=None):
        if not self.root_dir_path or not os.path.exists(self.root_dir_path):
            return
        
        dir_path = self.root_dir_path
        if not current_path:
            current_path = self.current_image_path
        
        self.image_files = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        ignore_dirs = {"no_used", "unmask"}

        try:
            for entry in os.scandir(dir_path):
                if entry.is_file() and entry.name.lower().endswith(valid_exts):
                    if any(part.lower() in ignore_dirs for part in Path(entry.path).parts):
                        continue
                    self.image_files.append(entry.path)
        except Exception:
            pass

        try:
            for entry in os.scandir(dir_path):
                if entry.is_dir():
                    if entry.name.lower() in ignore_dirs:
                        continue
                    try:
                        for sub in os.scandir(entry.path):
                            if sub.is_file() and sub.name.lower().endswith(valid_exts):
                                if any(part.lower() in ignore_dirs for part in Path(sub.path).parts):
                                    continue
                                self.image_files.append(sub.path)
                    except Exception:
                        pass
        except Exception:
            pass

        self.image_files = natsorted(self.image_files)

        if not self.image_files:
            self.image_label.clear()
            self.txt_edit.clear()
            self.img_file_label.setText(self.tr("label_no_image"))
            self.current_index = -1
            self.current_image_path = None
            return

        if current_path and current_path in self.image_files:
            self.current_index = self.image_files.index(current_path)
        else:
            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1
            if self.current_index < 0:
                self.current_index = 0
        
        self.load_image()
        self.statusBar().showMessage(self.tr("msg_refreshed").replace("{count}", str(len(self.image_files))), 3000)

    def open_directory(self):
        default_dir = self.settings.get("last_open_dir", "")
        dir_path = QFileDialog.getExistingDirectory(self, self.tr("msg_select_dir"), default_dir)
        if dir_path:
            self.root_dir_path = dir_path
            self.settings["last_open_dir"] = dir_path
            save_app_settings(self.settings)

            self.filter_active = False
            self.filter_input.clear()
            self.all_image_files = []
            self.filtered_image_files = []

            self.refresh_file_list()

    def load_image(self):
        if 0 <= self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
            self.current_folder_path = str(Path(self.current_image_path).parent)

            # Update info bar
            total_count = len(self.filtered_image_files) if self.filter_active else len(self.image_files)
            current_num = self.filtered_image_files.index(self.current_image_path) + 1 if self.filter_active and self.current_image_path in self.filtered_image_files else self.current_index + 1
            
            self.index_input.blockSignals(True)
            self.index_input.setText(str(current_num))
            self.index_input.blockSignals(False)
            
            if self.filter_active:
                self.total_info_label.setText(f"<span style='color:red;'> / {total_count}</span>")
            else:
                self.total_info_label.setText(f" / {total_count}")
            
            self.img_file_label.setText(f" : {os.path.basename(self.current_image_path)}")

            self.current_pixmap = QPixmap(self.current_image_path)
            if not self.current_pixmap.isNull():
                self.update_image_display()
            else:
                self.image_label.clear()

            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            content = ""
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            self.txt_edit.blockSignals(True)
            self.txt_edit.setPlainText(content)
            self.txt_edit.blockSignals(False)

            self.update_txt_token_count()

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

    def _get_image_content_for_filter(self, image_path: str) -> str:
        content_parts = []
        if self.chk_filter_tags.isChecked():
            sidecar = load_image_sidecar(image_path)
            tags = sidecar.get("tagger_tags", "")
            content_parts.append(tags)
        
        if self.chk_filter_text.isChecked():
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content_parts.append(f.read())
                except Exception:
                    pass
        return " ".join(content_parts)

    def apply_filter(self):
        query = self.filter_input.text().strip()
        if not query:
            self.clear_filter()
            return
        if not self.image_files and not self.all_image_files:
            return
        if not self.all_image_files:
            self.all_image_files = list(self.image_files)
        
        qf = DanbooruQueryFilter(query)
        matched = []
        for img_path in self.all_image_files:
            content = self._get_image_content_for_filter(img_path)
            if qf.matches(content):
                matched.append(img_path)
        
        matched = qf.sort_images(matched)
        
        if not matched:
            self.statusBar().showMessage(self.tr("msg_filter_empty"), 3000)
            return
        
        self.filtered_image_files = matched
        self.image_files = matched
        self.filter_active = True
        self.current_index = 0
        self.load_image()
        self.statusBar().showMessage(self.tr("msg_filter_result").replace("{count}", str(len(matched))), 3000)

    def clear_filter(self):
        self.filter_input.clear()
        if self.all_image_files:
            current_path = self.current_image_path
            self.image_files = list(self.all_image_files)
            self.all_image_files = []
            self.filtered_image_files = []
            self.filter_active = False
            
            if current_path and current_path in self.image_files:
                self.current_index = self.image_files.index(current_path)
            else:
                self.current_index = 0
            
            if self.image_files:
                self.load_image()
            self.statusBar().showMessage(self.tr("msg_filter_cleared"), 2000)

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def first_image(self):
        if self.image_files:
            self.current_index = 0
            self.load_image()

    def last_image(self):
        if self.image_files:
            self.current_index = len(self.image_files) - 1
            self.load_image()

    def jump_to_index(self):
        try:
            val = int(self.index_input.text())
            target_idx = val - 1
            
            if self.filter_active:
                if 0 <= target_idx < len(self.filtered_image_files):
                    target_path = self.filtered_image_files[target_idx]
                    self.current_index = self.image_files.index(target_path)
                    self.load_image()
                else:
                    self.load_image()
            else:
                if 0 <= target_idx < len(self.image_files):
                    self.current_index = target_idx
                    self.load_image()
                else:
                    self.load_image()
        except Exception:
            self.load_image()

    def update_image_display(self):
        if not hasattr(self, 'current_pixmap') or self.current_pixmap.isNull():
            return
        scaled = self._get_processed_pixmap().scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def _get_processed_pixmap(self) -> QPixmap:
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

    def show_image_context_menu(self, pos: QPoint):
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
        if hasattr(self, 'current_pixmap') and not self.current_pixmap.isNull():
            QApplication.clipboard().setPixmap(self.current_pixmap)
            self.statusBar().showMessage(self.tr("msg_copied_image"), 2000)

    def _ctx_copy_path(self):
        if self.current_image_path:
            QApplication.clipboard().setText(os.path.abspath(self.current_image_path))
            self.statusBar().showMessage(self.tr("msg_copied_path"), 2000)
    
    def _ctx_open_folder(self):
        if self.current_image_path:
            folder = os.path.dirname(self.current_image_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def delete_current_image(self):
        if not self.current_image_path:
            return
        reply = QMessageBox.question(
            self, self.tr("title_confirm"), self.tr("msg_delete_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            src_dir = os.path.dirname(self.current_image_path)
            no_used_dir = os.path.join(src_dir, "no_used")
            if not os.path.exists(no_used_dir):
                os.makedirs(no_used_dir)

            files_to_move = [self.current_image_path]
            for ext in [".txt", ".npz", ".boorutag", ".pool.json", ".json"]:
                p = os.path.splitext(self.current_image_path)[0] + ext
                if os.path.exists(p):
                    files_to_move.append(p)

            for f_path in files_to_move:
                try:
                    shutil.move(f_path, os.path.join(no_used_dir, os.path.basename(f_path)))
                except Exception:
                    pass

            self.image_files.pop(self.current_index)
            if self.current_index >= len(self.image_files):
                self.current_index -= 1
            if self.image_files:
                self.load_image()
            else:
                self.image_label.clear()
                self.txt_edit.clear()

    # ==========================
    # DATA LOADING
    # ==========================
    def build_top_tags_for_current_image(self):
        hints = []
        tags_from_meta = []
        meta_path = str(self.current_image_path) + ".boorutag"
        if os.path.isfile(meta_path):
            tags_meta, hint_info = parse_boorutag_meta(meta_path)
            tags_from_meta.extend(tags_meta)
            hints.extend(hint_info)

        parent = Path(self.current_image_path).parent.name
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

    def add_custom_tag_dialog(self):
        if not self.current_folder_path:
            return
        tag, ok = QInputDialog.getText(self, self.tr("dialog_add_tag_title"), self.tr("dialog_add_tag_label"))
        if not ok:
            return
        tag = str(tag).strip()
        if not tag:
            return
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

    def load_tagger_tags_for_current_image(self):
        sidecar = load_image_sidecar(self.current_image_path)
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
        pages = self.load_nl_pages_for_image(self.current_image_path)
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
        active_text = self.txt_edit.toPlainText()

        self.flow_top.render_tags_flow(
            smart_parse_tags(", ".join(self.top_tags)),
            active_text,
            self.settings
        )
        self.flow_custom.render_tags_flow(
            smart_parse_tags(", ".join(self.custom_tags)),
            active_text,
            self.settings
        )
        self.flow_tagger.render_tags_flow(
            smart_parse_tags(", ".join(self.tagger_tags)),
            active_text,
            self.settings
        )

    def refresh_nl_tab(self):
        active_text = self.txt_edit.toPlainText()
        self.flow_nl.render_tags_flow(
            smart_parse_tags(self.nl_latest),
            active_text,
            self.settings
        )

    def set_current_nl_page(self, idx: int):
        if not self.nl_pages:
            self.nl_page_index = 0
            self.nl_latest = ""
            self.refresh_nl_tab()
            self.update_nl_page_controls()
            return

        idx = max(0, min(int(idx), len(self.nl_pages) - 1))
        self.nl_page_index = idx
        self.nl_latest = self.nl_pages[self.nl_page_index]

        self.refresh_nl_tab()
        self.update_nl_page_controls()
        self.on_text_changed()

    def update_nl_page_controls(self):
        total = len(self.nl_pages)
        if total <= 0:
            if hasattr(self, "nl_page_label"):
                self.nl_page_label.setText(f"{self.tr('label_page')} 0/0")
            if hasattr(self, "btn_prev_nl"):
                self.btn_prev_nl.setEnabled(False)
            if hasattr(self, "btn_next_nl"):
                self.btn_next_nl.setEnabled(False)
        else:
            self.nl_page_index = max(0, min(self.nl_page_index, total - 1))
            if hasattr(self, "nl_page_label"):
                txt = self.tr("label_page_fmt").replace("{current}", str(self.nl_page_index + 1)).replace("{total}", str(total))
                self.nl_page_label.setText(txt)
            if hasattr(self, "btn_prev_nl"):
                self.btn_prev_nl.setEnabled(self.nl_page_index > 0)
            if hasattr(self, "btn_next_nl"):
                self.btn_next_nl.setEnabled(self.nl_page_index < total - 1)

        self.update_nl_result_height()

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
