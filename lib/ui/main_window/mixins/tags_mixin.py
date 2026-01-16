"""
標籤管理 Mixin

負責處理：
- Top/Custom/Tagger 標籤管理
- 標籤載入和儲存
- 標籤刷新和渲染
- 標籤按鈕事件

依賴的屬性：
- self.top_tags, self.custom_tags, self.tagger_tags: list
- self.current_image_path, self.current_folder_path: str
- self.flow_top, self.flow_custom, self.flow_tagger - 標籤流式佈局
- self.txt_edit - 文本編輯器
- self.settings, self.default_custom_tags_global
"""

from PyQt6.QtWidgets import QInputDialog
from lib.utils import (parse_boorutag_meta, extract_bracket_content, 
                       load_image_sidecar, save_image_sidecar, smart_parse_tags)
from pathlib import Path
import os
import json


class TagsMixin:
    """標籤管理 Mixin"""
    
    def build_top_tags_for_current_image(self):
        """建立當前圖片的 top tags"""
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

    def load_folder_custom_tags(self, folder_path):
        """載入資料夾的自定義標籤"""
        p = os.path.join(folder_path, ".custom_tags.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tags = data.get("custom_tags", [])
                tags = [str(t).strip() for t in tags if str(t).strip()]
                if not tags:
                    tags = list(self.default_custom_tags_global)
                return tags
            except:
                return list(self.default_custom_tags_global)
        else:
            tags = list(self.default_custom_tags_global)
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"custom_tags": tags}, f, ensure_ascii=False, indent=2)
            except:
                pass
            return tags

    def save_folder_custom_tags(self, folder_path, tags):
        """儲存資料夾的自定義標籤"""
        p = os.path.join(folder_path, ".custom_tags.json")
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"custom_tags": tags}, f, ensure_ascii=False, indent=2)
        except:
            pass
            
    def folder_custom_tags_path(self, folder_path):
        """獲取自定義標籤檔案路徑"""
        return os.path.join(folder_path, ".custom_tags.json")

    def add_custom_tag_dialog(self):
        """新增自定義標籤對話框"""
        if not self.current_folder_path:
            return
        
        tag, ok = QInputDialog.getText(self, self.tr("dialog_add_tag_title"), 
                                       self.tr("dialog_add_tag_label"))
        if not ok:
            return
        
        tag = str(tag).strip().replace("_", " ").strip()
        if not tag:
            return
        
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
        """載入當前圖片的 tagger tags"""
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
        """儲存 tagger tags 到圖片"""
        sidecar = load_image_sidecar(image_path)
        sidecar["tagger_tags"] = raw_tags_str
        save_image_sidecar(image_path, sidecar)

    def refresh_tags_tab(self):
        """刷新標籤頁面 - 用於切換圖片或執行打標後的全量更新"""
        active_text = self.txt_edit.toPlainText() if hasattr(self, 'txt_edit') else ""
        
        if hasattr(self, 'flow_top'):
            self.flow_top.render_tags_flow(
                smart_parse_tags(", ".join(self.top_tags)),
                active_text,
                self.settings
            )
        if hasattr(self, 'flow_custom'):
            self.flow_custom.render_tags_flow(
                smart_parse_tags(", ".join(self.custom_tags)),
                active_text,
                self.settings
            )
        if hasattr(self, 'flow_tagger'):
            self.flow_tagger.render_tags_flow(
                smart_parse_tags(", ".join(self.tagger_tags)),
                active_text,
                self.settings
            )
        if hasattr(self, 'refresh_nl_tab'):
            self.refresh_nl_tab()

    def sync_tags_highlighting(self):
        """僅同步標籤的高亮狀態，不重新渲染元件，解決打字卡頓問題"""
        if not hasattr(self, 'txt_edit'):
            return
        active_text = self.txt_edit.toPlainText()
        
        if hasattr(self, 'flow_top'):
            self.flow_top.sync_state(active_text)
        if hasattr(self, 'flow_custom'):
            self.flow_custom.sync_state(active_text)
        if hasattr(self, 'flow_tagger'):
            self.flow_tagger.sync_state(active_text)
        if hasattr(self, 'flow_nl'):
            self.flow_nl.sync_state(active_text)

    def on_tag_button_toggled(self, tag, checked):
        """標籤按鈕切換事件"""
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
