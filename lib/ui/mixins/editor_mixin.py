from typing import TYPE_CHECKING
import os
import re
from PyQt6.QtWidgets import QMessageBox, QDialog
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import Qt

from lib.core.settings import DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CUSTOM_PROMPT_TEMPLATE, DEFAULT_APP_SETTINGS
from lib.utils.parsing import cleanup_csv_like_text
from lib.ui.dialogs.find_replace import AdvancedFindReplaceDialog

try:
    from transformers import AutoTokenizer, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPTokenizer = None

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class EditorMixin:
    """
    Mixin handling text editor operations, token counting, prompt management, and Find/Replace.
    """

    def on_text_changed(self):
        if not self.current_image_path:
            return
        
        content = self.txt_edit.toPlainText()
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
        if content != original_content:
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
            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception:
                pass

        self.flow_top.sync_state(content)
        self.flow_custom.sync_state(content)
        self.flow_tagger.sync_state(content)
        self.flow_nl.sync_state(content)

        self.update_txt_token_count()

    def _get_clip_tokenizer(self):
        if CLIPTokenizer is None:
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
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        if self._hf_tokenizer is None:
            try:
                # 這裡會下載約 1MB 的 tokenizer 設定檔 (只會下載一次)
                # 這是 Stable Diffusion 1.x / 2.x 最常用的 Text Encoder
                self._hf_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            except Exception as e:
                print(f"Failed to load CLIP tokenizer: {e}")
                self._hf_tokenizer = None
                
        return self._hf_tokenizer

    def update_txt_token_count(self):
            content = self.txt_edit.toPlainText()
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
                self.txt_token_label.setStyleSheet(f"color: {text_color}")
                
                # 設定文字：只顯示 "Tokens: 數字"
                self.txt_token_label.setText(f"{self.tr('label_tokens')}{count}")
                
            except Exception as e:
                print(f"Token count error: {e}")
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

    def use_custom_prompt(self):
        """Switch prompt editor to Custom Prompt template."""
        self.current_prompt_mode = "custom"
        try:
            self.prompt_edit.setPlainText(self.custom_prompt_template)
        except Exception:
            pass

    def open_find_replace(self):
        dlg = AdvancedFindReplaceDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            settings = dlg.get_settings()
            find_str = settings['find']
            rep_str = settings['replace']
            if not find_str:
                return
            target_files = self.image_files if settings['scope_all'] else [self.current_image_path]
            count = 0
            for img_path in target_files:
                if not img_path:
                    continue
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    flags = 0 if settings['case_sensitive'] else re.IGNORECASE
                    try:
                        # 1. 先執行取代
                        if settings['regex']:
                            new_content, n = re.subn(find_str, rep_str, content, flags=flags)
                            count += n
                        else:
                            if not settings['case_sensitive']:
                                pattern = re.compile(re.escape(find_str), re.IGNORECASE)
                                new_content, n = pattern.subn(rep_str, content)
                                count += n
                            else:
                                n = content.count(find_str)
                                if n > 0:
                                    new_content = content.replace(find_str, rep_str)
                                    count += n
                        
                        # 2. 如果有變動，執行自動格式化 (Format Refresh)
                        if new_content != content:
                            # === 修改重點開始：格式重整 ===
                            # 用逗號分割 -> 去除前後空白 -> 過濾空字串 -> 用 ", " 接回
                            parts = [p.strip() for p in new_content.split(",") if p.strip()]
                            new_content = ", ".join(parts)
                            # === 修改重點結束 ===

                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)

                    except Exception as e:
                        print(f"Replace error in {img_path}: {e}")

            self.load_image() # 重新載入當前圖片以顯示結果
            
            # 嘗試將焦點放回編輯框並捲動到底部 (非必要，但體驗較好)
            try:
                self.txt_edit.moveCursor(QTextCursor.MoveOperation.End)
                self.txt_edit.setFocus()
                self.txt_edit.ensureCursorVisible()
            except Exception:
                pass
                
            QMessageBox.information(self, "Result", f"Replaced {count} occurrences and reformatted.")
