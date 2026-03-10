from typing import TYPE_CHECKING
import os
import re

import lib.runtime.editor_service as editor_service
from lib.core.settings import DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CUSTOM_PROMPT_TEMPLATE, DEFAULT_APP_SETTINGS
from lib.utils.parsing import cleanup_csv_like_text

AutoTokenizer = None
CLIPTokenizer = None
TRANSFORMERS_AVAILABLE = None

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class EditorMixin:
    """
    Mixin handling text editor operations, token counting, prompt management, and Find/Replace.
    """

    def _ensure_transformers(self) -> bool:
        global AutoTokenizer, CLIPTokenizer, TRANSFORMERS_AVAILABLE
        if TRANSFORMERS_AVAILABLE is False:
            return False
        if TRANSFORMERS_AVAILABLE is True:
            return True
        try:
            from transformers import AutoTokenizer as _AutoTokenizer, CLIPTokenizer as _CLIPTokenizer
            AutoTokenizer = _AutoTokenizer
            CLIPTokenizer = _CLIPTokenizer
            TRANSFORMERS_AVAILABLE = True
            return True
        except Exception:
            TRANSFORMERS_AVAILABLE = False
            return False

    def on_text_changed(self):
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        if not current_image_path:
            return
        
        if hasattr(self, "txt_edit") and self.txt_edit is not None:
            content = self.txt_edit.toPlainText()
        elif hasattr(self, "_runtime_txt_content"):
            content = self._runtime_txt_content()
        else:
            content = ""
        original_content = content
        
        # 自動移除空行
        if self.settings.get("text_auto_remove_empty_lines", True):
            lines = content.split("\n")
            lines = [line for line in lines if line.strip()]
            content = "\n".join(lines)
        
        # 自動格式化 (用 , 分割，去除空白，用 ', ' 重組)
        if self.settings.get("text_auto_format", True):
            # 如果內容看起來是 CSV 格式
            if "," in content and "\n" not in content.strip():
                parts = [p.strip() for p in content.split(",") if p.strip()]
                content = ", ".join(parts)
        
        # 如果內容有變動，更新編輯框
        if content != original_content and hasattr(self, "txt_edit") and self.txt_edit is not None:
            cursor_pos = self.txt_edit.textCursor().position()
            self.txt_edit.blockSignals(True)
            self.txt_edit.setPlainText(content)
            self.txt_edit.blockSignals(False)
            # 嘗試恢復游標位置
            cursor = self.txt_edit.textCursor()
            cursor.setPosition(min(cursor_pos, len(content)))
            self.txt_edit.setTextCursor(cursor)
        
        # 自動儲存 txt
        if self.settings.get("text_auto_save", True):
            txt_path = os.path.splitext(current_image_path)[0] + ".txt"
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception:
                pass

        if hasattr(self, "flow_top") and self.flow_top is not None:
            self.flow_top.sync_state(content)
        if hasattr(self, "flow_custom") and self.flow_custom is not None:
            self.flow_custom.sync_state(content)
        if hasattr(self, "flow_tagger") and self.flow_tagger is not None:
            self.flow_tagger.sync_state(content)
        if hasattr(self, "flow_nl") and self.flow_nl is not None:
            self.flow_nl.sync_state(content)

        self.update_txt_token_count()
        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()

    def _get_clip_tokenizer(self):
        if not self._ensure_transformers() or CLIPTokenizer is None:
            return None
        if self._clip_tokenizer is None:
            try:
                self._clip_tokenizer = CLIPTokenizer("openai/clip-vit-large-patch14")
            except Exception:
                self._clip_tokenizer = None
        return self._clip_tokenizer

    def _get_tokenizer(self):
        """
        Lazy load tokenizer to avoid startup lag.
        Uses the standard SD 1.5 CLIP model (openai/clip-vit-large-patch14).
        """
        if not self._ensure_transformers():
            return None

        if getattr(self, "_tokenizer_failed", False):
            return None

        if self._hf_tokenizer is None:
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    local_files_only=bool(self.settings.get("tokenizer_local_only", True)),
                )
            except Exception as e:
                self._hf_tokenizer = None
                if not bool(self.settings.get("tokenizer_retry_on_failure", False)):
                    self._tokenizer_failed = True
                if not getattr(self, "_tokenizer_warned", False):
                    self._tokenizer_warned = True
                    print(f"Tokenizer unavailable, fallback to regex token count: {e}")

        return self._hf_tokenizer

    def update_txt_token_count(self):
            if hasattr(self, "txt_edit") and self.txt_edit is not None:
                content = self.txt_edit.toPlainText()
            elif hasattr(self, "_runtime_txt_content"):
                content = self._runtime_txt_content()
            else:
                content = ""
            tokenizer = None
            if getattr(self, "_app_startup_complete", True):
                tokenizer = self._get_tokenizer()

            count = 0

            try:
                if tokenizer:
                    # 使用 CLIP Tokenizer 精確計算
                    tokens = tokenizer.encode(content, add_special_tokens=False)
                    count = len(tokens)
                else:
                    # 降級使用 Regex 估算
                    if content.strip():
                        tokens = re.findall(r'\w+|[^\w\s]', content)
                        count = len(tokens)
                
                # 設定顏色：超過 225 才變紅，否則全黑
                text_color = "red" if count > 225 else "black"
                if hasattr(self, "txt_token_label") and self.txt_token_label is not None:
                    self.txt_token_label.setStyleSheet(f"color: {text_color}")
                
                # 設定文字：只顯示 "Tokens: 數字"
                if hasattr(self, "txt_token_label") and self.txt_token_label is not None:
                    self.txt_token_label.setText(f"{self.tr('label_tokens')}{count}")
                
            except Exception as e:
                print(f"Token count error: {e}")
                if hasattr(self, "txt_token_label") and self.txt_token_label is not None:
                    self.txt_token_label.setText(self.tr("label_tokens_err"))

    def on_tag_button_toggled(self, tag, checked):
        if not self.current_image_path:
            return

        tag = str(tag).strip()
        if not tag:
            return

        if checked:
            self.insert_token_at_cursor(tag)
        else:
            self.remove_token_everywhere(tag)

        self.on_text_changed()

    def insert_token_at_cursor(self, token: str):
        from PyQt6.QtGui import QTextCursor

        token = token.strip()
        if not token:
            return

        edit = self.txt_edit
        text = edit.toPlainText()
        cursor = edit.textCursor()
        
        # (2) 如果沒有游標 (游標在開頭且沒焦點) 則附加在 text 尾
        # 在 PyQt 中，hasFocus() 可以在點擊按鈕前判斷是否有交互
        if cursor.position() == 0 and len(text) > 0 and not edit.hasFocus():
            cursor.movePosition(QTextCursor.MoveOperation.End)
            edit.setTextCursor(cursor)

        # (1) 優先插入在游標位置
        pos = cursor.position()
        before = text[:pos]
        after = text[pos:]

        # 前後加 ", " 然後格式化
        new_text = before + ", " + token + ", " + after
        final = cleanup_csv_like_text(new_text, self.english_force_lowercase)

        edit.blockSignals(True)
        edit.setPlainText(final)
        edit.blockSignals(False)
        
        # 格式化後嘗試把游標移到插入的 token 之後
        new_cursor = edit.textCursor()
        # 簡單搜尋 token 出現的位置 (從之前位置附近開始找)
        search_start = max(0, pos - 5)
        new_pos = final.find(token, search_start)
        if new_pos != -1:
            new_cursor.setPosition(new_pos + len(token))
        else:
            new_cursor.movePosition(QTextCursor.MoveOperation.End)
        
        edit.setTextCursor(new_cursor)
        edit.ensureCursorVisible()
        # 不需要強行 setFocus，保留按鈕焦點可能更方便連續按

    def remove_token_everywhere(self, token: str):
        token = token.strip()
        if not token:
            return
        text = self.txt_edit.toPlainText()

        new_text = text.replace(token, "")
        new_text = cleanup_csv_like_text(new_text)

        self.txt_edit.blockSignals(True)
        self.txt_edit.setPlainText(new_text)
        self.txt_edit.blockSignals(False)

        self.update_txt_token_count()

    def reset_prompt(self):
        self.prompt_edit.setPlainText(DEFAULT_USER_PROMPT_TEMPLATE)

    def use_default_prompt(self):
        """Switch prompt editor to Default Prompt template."""
        self.current_prompt_mode = "default"
        try:
            self.prompt_edit.setPlainText(self.default_user_prompt_template)
        except Exception:
            pass
        if hasattr(self, "_sync_runtime_ui_state"):
            self._sync_runtime_ui_state()
        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()

    def use_custom_prompt(self):
        """Switch prompt editor to Custom Prompt template."""
        self.current_prompt_mode = "custom"
        try:
            self.prompt_edit.setPlainText(self.custom_prompt_template)
        except Exception:
            pass
        if hasattr(self, "_sync_runtime_ui_state"):
            self._sync_runtime_ui_state()
        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()

    def run_find_replace(
        self,
        find_text: str,
        replace_text: str = "",
        *,
        scope_all: bool = False,
        case_sensitive: bool = False,
        regex: bool = False,
    ):
        find_text = str(find_text or "")
        replace_text = str(replace_text or "")
        if not find_text:
            raise ValueError("find_text is required")

        if scope_all:
            if hasattr(self, "_runtime_loaded_image_paths"):
                target_files = list(self._runtime_loaded_image_paths())
            else:
                target_files = list(self.image_files)
        else:
            current_runtime_image = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
            target_files = [current_runtime_image]
        current_image_path = self._runtime_current_image_path() if hasattr(self, "_runtime_current_image_path") else str(getattr(self, "current_image_path", "") or "")
        result = editor_service.run_find_replace_on_images(
            target_files,
            find_text=find_text,
            replace_text=replace_text,
            case_sensitive=case_sensitive,
            regex=regex,
        )

        current_reloaded = False
        changed_files = list(result.get("changed_files", []) or [])
        if current_image_path and (not scope_all or current_image_path in changed_files or current_image_path in target_files):
            self.load_image()
            current_reloaded = True
            try:
                from PyQt6.QtGui import QTextCursor

                self.txt_edit.moveCursor(QTextCursor.MoveOperation.End)
                self.txt_edit.setFocus()
                self.txt_edit.ensureCursorVisible()
            except Exception:
                pass

        if hasattr(self, "_sync_runtime_content_state"):
            self._sync_runtime_content_state()

        result.update(
            {
            "scope_all": bool(scope_all),
            "current_reloaded": current_reloaded,
            }
        )
        return result

    def open_find_replace(self):
        from PyQt6.QtWidgets import QDialog, QMessageBox

        from lib.ui.dialogs.find_replace import AdvancedFindReplaceDialog

        dlg = AdvancedFindReplaceDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            settings = dlg.get_settings()
            try:
                result = self.run_find_replace(
                    settings["find"],
                    settings["replace"],
                    scope_all=bool(settings["scope_all"]),
                    case_sensitive=bool(settings["case_sensitive"]),
                    regex=bool(settings["regex"]),
                )
            except ValueError:
                return
            QMessageBox.information(
                self,
                self.tr("title_info"),
                self.tr("msg_replace_result").replace("{count}", str(result["replacement_count"])),
            )
