"""
文本編輯 Mixin

負責處理：
- 文本變更處理
- Token 計數
- 插入/移除 Token
- 查找替換

依賴的屬性：
- self.txt_edit - 文本編輯器
- self.current_image_path: str - 當前圖片路徑
- self.settings: dict - 設定
- self.english_force_lowercase: bool - 強制小寫
- self._hf_tokenizer - HuggingFace tokenizer
- self.txt_token_label - Token 標籤
"""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QTextCursor
import os
import re


class TextEditMixin:
    """文本編輯 Mixin"""
    
    def on_text_changed(self):
        """文本變更處理"""
        if not self.current_image_path:
            return
        
        content = self.txt_edit.toPlainText()
        original_content = content
        
        # 自動格式化
        if self.settings.get("text_auto_format", True):
            if "," in content and "\n" not in content.strip():
                parts = [p.strip() for p in content.split(",") if p.strip()]
                content = ", ".join(parts)
        
        # 更新編輯框
        if content != original_content:
            cursor_pos = self.txt_edit.textCursor().position()
            self.txt_edit.blockSignals(True)
            self.txt_edit.setPlainText(content)
            self.txt_edit.blockSignals(False)
            cursor = self.txt_edit.textCursor()
            cursor.setPosition(min(cursor_pos, len(content)))
            self.txt_edit.setTextCursor(cursor)
        
        # 自動儲存
        if self.settings.get("text_auto_save", True):
            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except:
                pass
        
        # 同步標籤狀態
        if hasattr(self, 'flow_top'):
            self.flow_top.sync_state(content)
        if hasattr(self, 'flow_custom'):
            self.flow_custom.sync_state(content)
        if hasattr(self, 'flow_tagger'):
            self.flow_tagger.sync_state(content)
        if hasattr(self, 'flow_nl'):
            self.flow_nl.sync_state(content)
        
        self.update_txt_token_count()

    def update_txt_token_count(self):
        """更新 Token 計數"""
        content = self.txt_edit.toPlainText()
        tokenizer = self._get_tokenizer()
        count = 0
        
        try:
            if tokenizer:
                tokens = tokenizer.encode(content, add_special_tokens=False)
                count = len(tokens)
            else:
                if content.strip():
                    tokens = re.findall(r'\w+|[^\w\s]', content)
                    count = len(tokens)
            
            text_color = "red" if count > 225 else "black"
            self.txt_token_label.setStyleSheet(f"color: {text_color}")
            self.txt_token_label.setText(f"{self.tr('label_tokens')}{count}")
        except Exception as e:
            print(f"Token count error: {e}")
            self.txt_token_label.setText(self.tr("label_tokens_err"))

    def _get_tokenizer(self):
        """取得 tokenizer"""
        try:
            from transformers import AutoTokenizer
            if self._hf_tokenizer is None:
                self._hf_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            return self._hf_tokenizer
        except:
            return None

    def insert_token_at_cursor(self, token: str):
        """在游標位置插入 token"""
        token = token.strip()
        if not token:
            return
        
        edit = self.txt_edit
        text = edit.toPlainText()
        cursor = edit.textCursor()
        
        if cursor.position() == 0 and len(text) > 0 and not edit.hasFocus():
            cursor.movePosition(QTextCursor.MoveOperation.End)
            edit.setTextCursor(cursor)
        
        pos = cursor.position()
        before = text[:pos]
        after = text[pos:]
        
        new_text = before + ", " + token + ", " + after
        from lib.utils import cleanup_csv_like_text
        final = cleanup_csv_like_text(new_text, self.english_force_lowercase)
        
        edit.blockSignals(True)
        edit.setPlainText(final)
        edit.blockSignals(False)
        
        new_cursor = edit.textCursor()
        search_start = max(0, pos - 5)
        new_pos = final.find(token, search_start)
        if new_pos != -1:
            new_cursor.setPosition(new_pos + len(token))
        else:
            new_cursor.movePosition(QTextCursor.MoveOperation.End)
        
        edit.setTextCursor(new_cursor)
        edit.ensureCursorVisible()

    def remove_token_everywhere(self, token: str):
        """移除所有 token"""
        token = token.strip()
        if not token:
            return
        
        text = self.txt_edit.toPlainText()
        new_text = text.replace(token, "")
        from lib.utils import cleanup_csv_like_text
        new_text = cleanup_csv_like_text(new_text)
        
        self.txt_edit.blockSignals(True)
        self.txt_edit.setPlainText(new_text)
        self.txt_edit.blockSignals(False)
        
        self.update_txt_token_count()

    def _do_find_replace(self, opts):
        """執行查找替換"""
        find_str = opts['find']
        rep_str = opts['replace']
        if not find_str:
            return
        
        target_files = self.image_files if opts['scope_all'] else [self.current_image_path]
        count = 0
        
        for img_path in target_files:
            if not img_path:
                continue
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                flags = 0 if opts['case_sensitive'] else re.IGNORECASE
                
                try:
                    if opts['regex']:
                        new_content, n = re.subn(find_str, rep_str, content, flags=flags)
                        count += n
                    else:
                        if not opts['case_sensitive']:
                            pattern = re.compile(re.escape(find_str), re.IGNORECASE)
                            new_content, n = pattern.subn(rep_str, content)
                            count += n
                        else:
                            n = content.count(find_str)
                            if n > 0:
                                new_content = content.replace(find_str, rep_str)
                                count += n
                    
                    if new_content != content:
                        parts = [p.strip() for p in new_content.split(",") if p.strip()]
                        new_content = ", ".join(parts)
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                except Exception as e:
                    print(f"Replace error in {img_path}: {e}")
        
        self.load_image()
        try:
            self.txt_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.txt_edit.setFocus()
            self.txt_edit.ensureCursorVisible()
        except:
            pass
        
        QMessageBox.information(self, "Result", f"Replaced {count} occurrences and reformatted.")
