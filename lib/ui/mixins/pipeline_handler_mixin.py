from typing import TYPE_CHECKING
import os

from lib.pipeline.context import TaskResult
from lib.utils.parsing import extract_llm_content_and_postprocess

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class PipelineHandlerMixin:
    """
    Mixin handling Pipeline callback events.
    """
    def on_pipeline_progress(self, current, total, filename, speed=0.0):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        # 取得任務與模型資訊
        task_name = self.pipeline_manager._current_pipeline.name if self.pipeline_manager._current_pipeline else "TASK"
        model_info = task_name.upper()
        
        if "tagger" in task_name:
            # 簡化顯示，只取最後一個斜線後的名稱
            model = self.settings.get("tagger_model", "")
            if "/" in model: 
                 model = model.split("/")[-1]
            model_info = f"TAGGER ({model})"
        elif "llm" in task_name:
            model = self.settings.get("llm_model", "")
            model_info = f"LLM ({model})"
        elif "unmask" in task_name:
             mode = self.settings.get("mask_remover_mode", "base")
             model_info = f"UNMASK ({mode})"
        elif "mask_text" in task_name:
             model_info = "MASK TEXT (OCR)"
        elif "restore" in task_name:
             model_info = "RESTORE"
        
        # 速度
        if speed > 0:
            if speed < 1:
                speed_str = f"{1/speed:.2f} s/it"
            else:
                speed_str = f"{speed:.2f} it/s"
        else:
            speed_str = "..."

        base_msg = self.tr("status_batch_progress_fmt").replace("{current}", str(current)).replace("{total}", str(total)).replace("{filename}", filename)
        
        final_msg = f"{model_info} | {base_msg} | {speed_str}"
            
        self.progress_bar.setFormat(final_msg)
        self.btn_cancel_batch.setVisible(True)

    def on_pipeline_error(self, err_msg):
        self.statusBar().showMessage(self.tr("msg_error_fmt").replace("{msg}", str(err_msg)), 8000)
        self.progress_bar.setVisible(False)
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText(self.tr("btn_run_llm"))
        self.set_batch_ui_enabled(True) 

    def on_pipeline_image_done(self, image_path: str, output: TaskResult):
        if not output.success:
            print(f"Error for {image_path}: {output.error}")
            return
            
        if output.skipped:
             reason = output.skip_reason or self.tr("msg_task_skipped")
             self.statusBar().showMessage(f"{os.path.basename(image_path)}: {reason}", 5000)
             # If strictly single image mode, maybe show alert? But status bar is less intrusive.
             return

        task_name = self.pipeline_manager._current_pipeline.name if self.pipeline_manager._current_pipeline else ""
        
        # Tagger
        if "tagger" in task_name:
             if output.result_text:
                 self.save_tagger_tags_for_image(image_path, output.result_text)
                 
             if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                 self.tagger_tags = self.load_tagger_tags_for_current_image()
                 self.refresh_tags_tab()
             
             if getattr(self, "_is_batch_to_txt", False) and output.result_text:
                  self.write_batch_result_to_txt(image_path, output.result_text, is_tagger=True)

        # LLM
        elif "llm" in task_name:
             content = output.result_text or ""
             final_content = extract_llm_content_and_postprocess(content, self.english_force_lowercase)
             
             if final_content:
                 self.save_nl_for_image(image_path, final_content)
                 if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                     if final_content not in self.nl_pages:
                        self.nl_pages.append(final_content)
                     self.nl_page_index = len(self.nl_pages) - 1
                     self.nl_latest = final_content
                     self.refresh_nl_tab()
                     self.update_nl_page_controls()
                     self.on_text_changed()
                     
                 if getattr(self, "_is_batch_to_txt", False):
                      self.write_batch_result_to_txt(image_path, final_content, is_tagger=False)

        # Unmask or Mask Text
        elif "unmask" in task_name or "mask_text" in task_name:
             if output.result_data:
                  old_path = output.result_data.get("original_path")
                  new_path = output.result_data.get("result_path")
                  
                  # Mask Text special case: 0 boxes found
                  if "mask_text" in task_name and output.result_data.get("box_count", 0) == 0:
                      self.statusBar().showMessage(self.tr("msg_no_text_detected"), 4000)
                      # No new file created usually if box_count is 0, so new_path might be None
                  
                  elif new_path:
                      self._replace_image_path_in_list(old_path, new_path)
                      self.statusBar().showMessage(self.tr("status_done"), 3000)
                  
                  else:
                      # Result data exists but no new path? Maybe skipped internally without flag
                      pass

             if self.current_image_path:
                 self.load_image()

                 
        # Restore
        elif "restore" in task_name:
             if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                 self.load_image()
