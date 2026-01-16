"""
文本編輯 Mixin

負責處理：
- 文本變更監聽
- Token 計數
- 插入/移除 Token
- 查找替換
- 撤銷/重做綁定

依賴的屬性：
- self.txt_edit: QTextEdit
- self.txt_token_label: QLabel
- self.image_files, self.current_image_path
- self.settings
"""

from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QDialog
import re
from lib.ui.dialogs import AdvancedFindReplaceDialog
import os

try:
    from clip_anytokenizer import CLIPTokenizer
except ImportError:
    CLIPTokenizer = None

class TextEditMixin:
    """文本編輯 Mixin"""
    
    def on_text_changed(self):
        """文本變更事件"""
        content = self.txt_edit.toPlainText()
        
        # Update token count
        count = self.calculate_token_count(content)
        if hasattr(self, 'txt_token_label'):
            self.txt_token_label.setText(f"{self.tr('label_tokens')}{count}")
            if count > 75:
                self.txt_token_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.txt_token_label.setStyleSheet("")
        
        # Save to file
        if self.current_image_path:
            txt_path = os.path.splitext(self.current_image_path)[0] + ".txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"Failed to save txt: {e}")
        
        # Refresh Highlight
        if hasattr(self, 'refresh_tags_tab'):
            # Avoid infinite loop or lag? Typically ok.
            # Ideally we only refresh highlighting, not reload everything.
            # But refresh_tags_tab uses active_text to highlight buttons.
            self.refresh_tags_tab()

    def calculate_token_count(self, text):
        """計算 Token 數量"""
        tokenizer = self._get_clip_tokenizer()
        if tokenizer:
            try:
                # CLIP tokenizer usage
                # batch_decode/encode not strictly standard everywhere, but assuming clip_anytokenizer
                # Just length of tokens
                tokens = tokenizer.encode(text) 
                return len(tokens)
            except:
                pass
        
        # Fallback regex
        # Just comma split count? Or word count?
        parts = [p for p in re.split(r'[, ]+', text) if p.strip()]
        return len(parts)

    def _get_clip_tokenizer(self):
        if CLIPTokenizer is None:
            return None
        if not hasattr(self, '_clip_tokenizer') or self._clip_tokenizer is None:
            try:
                self._clip_tokenizer = CLIPTokenizer("openai/clip-vit-large-patch14")
            except Exception:
                self._clip_tokenizer = None
        return self._clip_tokenizer

    def insert_token_at_cursor(self, token):
        """在游標處插入 Token"""
        cursor = self.txt_edit.textCursor()
        text = self.txt_edit.toPlainText()
        
        # If token already exists?
        # Simple check:
        # If we just want to ensure it's there. 
        # But this function is typically called by 'Toggle On'.
        
        if not text.strip():
            self.txt_edit.insertPlainText(token)
            return

        # Smart comma insertion
        # If previous char is not space/comma, add comma
        cursor.movePosition(QTextCursor.MoveOperation.End) # Append to end usually?
        # Wait, tag toggle usually appends or inserts?
        # Legacy behavior: append to end.
        
        if len(text) > 0 and not text.endswith(", ") and not text.endswith(","):
             self.txt_edit.moveCursor(QTextCursor.MoveOperation.End)
             self.txt_edit.insertPlainText(", " + token)
        else:
             self.txt_edit.moveCursor(QTextCursor.MoveOperation.End)
             self.txt_edit.insertPlainText(token)
             
        # Scroll to bottom
        sb = self.txt_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def remove_token_everywhere(self, token):
        """移除所有匹配的 Token"""
        text = self.txt_edit.toPlainText()
        
        # Regex replace
        # "tag", "tag, ", ", tag"
        # Need robust removal
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        
        # This is tricky with commas.
        # Simple approach: split by commas, filter, join
        parts = [p.strip() for p in text.split(',')]
        new_parts = [p for p in parts if p.lower() != token.lower()]
        new_text = ", ".join(new_parts)
        
        if new_text != text:
            self.txt_edit.blockSignals(True)
            self.txt_edit.setPlainText(new_text)
            self.txt_edit.blockSignals(False)
            self.on_text_changed() # Manual trigger

    def open_find_replace(self):
        """打開搜尋取代對話框"""
        dlg = AdvancedFindReplaceDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            settings = dlg.get_settings()
            find_str = settings['find']
            rep_str = settings['replace']
            if not find_str:
                return
            target_files = self.image_files if settings['scope_all'] else [self.current_image_path]
            count = 0
            
            # Progress?
            
            for img_path in target_files:
                if not img_path: continue
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    new_content = content
                    flags = 0 if settings['case_sensitive'] else re.IGNORECASE
                    try:
                        if settings['regex']:
                             new_content, n = re.subn(find_str, rep_str, content, flags=flags)
                             count += n
                        else:
                             pattern = re.compile(re.escape(find_str), flags)
                             new_content, n = pattern.subn(rep_str, content)
                             count += n
                             
                        if new_content != content:
                             with open(txt_path, 'w', encoding='utf-8') as f:
                                 f.write(new_content)
                                 
                    except Exception as e:
                        print(f"Replace error: {e}")
            
            self.statusBar().showMessage(f"已在 {len(target_files)} 個檔案中取代 {count} 處匹配")
            self.load_image() # Reload current to see changes
