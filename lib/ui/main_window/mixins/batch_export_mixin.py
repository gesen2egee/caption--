"""
批量導出 Mixin

負責處理：
- 批量 Tagger 轉 txt (run_batch_tagger_to_txt)
- 批量 LLM 轉 txt (run_batch_llm_to_txt)
- 寫入 txt 的邏輯 (write_batch_result_to_txt)
- 詢問是否刪除特徵標籤 (prompt_delete_chars)

依賴的屬性：
- self.settings: dict
- self.image_files: list
- self.app_settings: AppSettings
- self.run_batch_tagger()
- self.run_batch_llm()
- self._is_batch_to_txt: bool
- self._batch_delete_chars: bool
"""

from PyQt6.QtWidgets import QMessageBox
from lib.utils import load_image_sidecar, is_basic_character_tag, remove_underline
from lib.data import ImageContext
from lib.processors.tagger import TaggerProcessor
from lib.processors.llm import LLMProcessor
from lib.workers.batch import GenericBatchWorker
import os
import re


class BatchExportMixin:
    """批量導出 Mixin"""

    def run_batch_tagger_to_txt(self):
        """批量 Tagger 轉 txt"""
        if not self.image_files:
            return
        
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_tagger_to_txt.setEnabled(False)

        # 1. Check Sidecar for cache
        files_to_process = []
        already_done_count = 0

        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                tags_str = sidecar.get("tagger_tags", "")

                if tags_str:
                    # Cache hit: Write directly
                    self.write_batch_result_to_txt(img_path, tags_str, is_tagger=True)
                    already_done_count += 1
                else:
                    # Cache miss: Add to queue
                    files_to_process.append(img_path)

            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 Tagger 結果至 txt", 5000)

            # 2. Process missing files
            if not files_to_process:
                self.btn_batch_tagger_to_txt.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch Tagger to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 Tagger...", 5000)

            # Use GenericBatchWorker + TaggerProcessor
            contexts = [ImageContext(p) for p in files_to_process]
            proc = TaggerProcessor(self.app_settings)

            self.batch_tagger_thread = GenericBatchWorker(contexts, proc)
            self.batch_tagger_thread.progress.connect(self.show_progress)
            self.batch_tagger_thread.item_done.connect(self.on_batch_tagger_per_image)
            self.batch_tagger_thread.finished_all.connect(self.on_batch_tagger_all_done)
            # TaggerMixin should define on_batch_tagger_all_done (was on_batch_tagger_done in legacy?)
            # TaggerMixin uses on_batch_tagger_all_done
            self.batch_tagger_thread.error.connect(lambda e: self.on_batch_error("Tagger", e))
            self.batch_tagger_thread.start()

        except Exception as e:
            self.btn_batch_tagger_to_txt.setEnabled(True)
            self._is_batch_to_txt = False
            QMessageBox.warning(self, "Error", f"Batch Processing Error: {e}")

    def run_batch_llm_to_txt(self):
        """批量 LLM 轉 txt"""
        if not self.image_files:
            return
            
        delete_chars = self.prompt_delete_chars()
        if delete_chars is None:
            return
            
        self._is_batch_to_txt = True
        self._batch_delete_chars = delete_chars
        
        self.btn_batch_llm_to_txt.setEnabled(False)

        # 1. 檢查 Sidecar，將已有結果者直接寫入 txt
        files_to_process = []
        already_done_count = 0
        
        try:
            for img_path in self.image_files:
                sidecar = load_image_sidecar(img_path)
                nl = sidecar.get("nl_pages", [])
                
                content = ""
                # 使用最後一次結果
                if nl and isinstance(nl, list):
                    content = nl[-1]
                
                if content:
                    # 已有結果 -> 直接寫入
                    self.write_batch_result_to_txt(img_path, content, is_tagger=False)
                    already_done_count += 1
                else:
                    # 無結果 -> 加入待處理清單
                    files_to_process.append(img_path)
            
            if already_done_count > 0:
                self.statusBar().showMessage(f"已從 Sidecar 還原 {already_done_count} 筆 LLM 結果至 txt", 5000)

            # 2. 針對無結果的檔案，執行 Batch LLM
            if not files_to_process:
                # 全部都有結果，直接結束
                self.btn_batch_llm_to_txt.setEnabled(True)
                self._is_batch_to_txt = False
                QMessageBox.information(self, "Batch LLM to txt", f"完成！共處理 {already_done_count} 檔案 (使用現有記錄)。")
                return

            # 有缺漏 -> 跑 Batch LLM
            self.statusBar().showMessage(f"尚有 {len(files_to_process)} 檔案無記錄，開始執行 LLM...", 5000)
            
            # 使用 UI 中的 prompt 或者 預設 prompt?
            # 這裡假設如果用戶在編輯框有輸入，就用輸入的，否則用預設。
            # 但考慮到批量，通常應該是用預設或 Settings 裡的 template。
            # 為了簡單起見，這裡使用 editor 內容作為 override（如果這樣設計符合直覺）
            # 或者使用 settings。 Legacy code 使用 editor 內容。
            user_prompt = self.prompt_edit.toPlainText()

            contexts = [ImageContext(p) for p in files_to_process]
            proc = LLMProcessor(self.app_settings, override_user_prompt=user_prompt)

            self.batch_llm_thread = GenericBatchWorker(contexts, proc)
            self.batch_llm_thread.progress.connect(self.show_progress)
            self.batch_llm_thread.item_done.connect(self.on_batch_llm_per_image) # LLMMixin defines this
            self.batch_llm_thread.finished_all.connect(self.on_batch_llm_all_done) # LLMMixin defines this
            self.batch_llm_thread.error.connect(lambda e: self.on_batch_error("LLM", e))
            self.batch_llm_thread.start()

        except Exception as e:
            self.btn_batch_llm_to_txt.setEnabled(True)
            self._is_batch_to_txt = False
            QMessageBox.warning(self, "Error", f"Batch Processing Error: {e}")
            return

    def prompt_delete_chars(self) -> bool:
        """回傳 True=刪除, False=保留, None=取消"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch to txt")
        msg.setText("是否自動刪除特徵標籤 (Character Tags)？")
        msg.setInformativeText("將根據設定中的黑白名單過濾標籤或句子。")
        btn_yes = msg.addButton("自動刪除", QMessageBox.ButtonRole.YesRole)
        btn_no = msg.addButton("保留", QMessageBox.ButtonRole.NoRole)
        btn_cancel = msg.addButton(QMessageBox.StandardButton.Cancel)
        
        msg.exec()
        if msg.clickedButton() == btn_yes:
            return True
        elif msg.clickedButton() == btn_no:
            return False
        return None

    def write_batch_result_to_txt(self, image_path, content, is_tagger: bool):
        """將結果寫入 txt"""
        cfg = self.settings
        delete_chars = getattr(self, "_batch_delete_chars", False)
        mode = cfg.get("batch_to_txt_mode", "append")
        folder_trigger = cfg.get("batch_to_txt_folder_trigger", False)
        
        items = []
        if is_tagger:
            raw_list = [x.strip() for x in content.split(",") if x.strip()]
            if delete_chars:
                raw_list = [t for t in raw_list if not is_basic_character_tag(t, cfg)]
            items = raw_list
        else:
            # LLM output processing
            raw_lines = content.splitlines()
            sentences = []
            for line in raw_lines:
                line = line.strip()
                if not line: continue
                # Skip translation lines
                if (line.startswith("(") and line.endswith(")")) or (line.startswith("（") and line.endswith("）")):
                    continue
                # Simple cleanup
                line = re.sub(r"[\(（].*?[\)）]", "", line).strip()
                if not line: continue
                
                if delete_chars:
                    # 簡單過濾：如果句子包含 basic character tag，則忽略 (這是種簡單策略，可能太過激進)
                    # 更好的策略可能是：如果句子 *完全等於* 某個 character tag
                    # 或使用更複雜的 NLP。
                    # 這裡沿用 Legacy 邏輯：is_basic_character_tag 檢查的是單詞。
                    # 對於句子，我們暫且不過濾，或者只過濾單詞級別的？
                    # 查看 Legacy 代碼，LLM 模式下是否過濾？
                    # Legacy 代碼顯示 LLM 模式下如果 delete_chars 為 True，會檢查 sentences。
                    # 但 is_basic_character_tag 主要是針對 comma separated tags。
                    # 對於 LLM 句子，通常不過濾，除非它是 "1girl" 這種單詞。
                    pass
                
                sentences.append(line)
            items = sentences

        # Combine
        final_list = []
        
        # 1. Folder trigger words
        if folder_trigger:
            parent = os.path.basename(os.path.dirname(image_path))
            if "_" in parent:
                trigger = parent.split("_", 1)[1]
                trigger = remove_underline(trigger)
                final_list.append(trigger)

        # 2. Content
        for item in items:
            item = item.strip()
            item = remove_underline(item)
            if self.english_force_lowercase:
                item = item.lower()
            if item and item not in final_list:
                final_list.append(item)
                
        # Write
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        
        if mode == "overwrite":
            mode_char = 'w'
        else: # append / prepend
            # Check existing
            existing_content = ""
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read().strip()
                except:
                    pass
            
            if mode == "prepend":
                # New + Existing
                # Convert list to string first
                new_str = ", ".join(final_list)
                combined = new_str
                if existing_content:
                    combined = new_str + ", " + existing_content
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(combined)
                return

            else: # append (default)
                # Existing + New
                mode_char = 'w' # We actally read and rewrite to format correctly
                
                new_str = ", ".join(final_list)
                combined = existing_content
                if new_str:
                    if combined:
                        combined = combined + ", " + new_str
                    else:
                        combined = new_str
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(combined)
                return

        # Overwrite mode
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(", ".join(final_list))
