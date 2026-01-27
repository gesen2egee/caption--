from typing import TYPE_CHECKING, List, Type, Optional, Dict, Any
import os

from lib.pipeline.context import TaskResult
from lib.utils.parsing import extract_llm_content_and_postprocess
from lib.core.dataclasses import ImageData, Settings, Prompt, FolderMeta
from lib.pipeline.tasks import BaseTask, TaggerTask, LLMTask, UnmaskTask, MaskTextTask, RestoreTask

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class PipelineHandlerMixin:
    """
    Mixin handling Task execution and callback events.
    Now directly manages the Task Thread (UI -> Task).
    """

    # ============================================================
    # Task Execution Management
    # ============================================================
    
    def is_task_running(self) -> bool:
        """Check if any task is currently running."""
        return getattr(self, "_current_task", None) is not None and self._current_task.isRunning()
    
    def stop_current_task(self):
        """Request stop for the current task."""
        if self._current_task:
            self._current_task.stop()

    def run_task(self, TaskClass: Type[BaseTask], images: List[ImageData], extra: Optional[Dict[str, Any]] = None):
        """Run a specified Task."""
        if self.is_task_running():
            self.on_pipeline_error("已有任務正在執行 (Task Running)")
            return

        settings_obj = self._get_current_settings_obj()
        
        # Create Task (Thread)
        self._current_task = TaskClass(
            images=images,
            settings=settings_obj,
            prompt=None, 
            folder=None, 
            extra=extra,
        )
        
        # Connect Signals directly to UI handlers
        self._current_task.progress.connect(self.on_pipeline_progress)
        self._current_task.image_done.connect(self.on_pipeline_image_done)
        self._current_task.batch_done.connect(
            lambda results: self._on_task_done(self._current_task.name, results)
        )
        self._current_task.error.connect(self.on_pipeline_error)
        
        # Start Thread
        self._current_task.start()

    def _get_current_settings_obj(self) -> Settings:
        """Helper to create Settings dataclass from current UI dict settings."""
        valid_keys = Settings.__annotations__.keys()
        clean_settings = {k: v for k, v in self.settings.items() if k in valid_keys}
        return Settings(**clean_settings)

    def _on_task_done(self, name: str, results: List[TaskResult]):
        """Internal callback when task thread finishes."""
        self.on_pipeline_done(name, results)
        self._current_task = None

    # ============================================================
    # Convenience Methods (Helpers)
    # ============================================================

    def run_tagger(self, images: List[ImageData]):
        self.run_task(TaggerTask, images)

    def run_llm(self, images: List[ImageData], user_prompt: str = None, system_prompt: str = None):
        extra = {}
        if user_prompt: extra["user_prompt"] = user_prompt
        if system_prompt: extra["system_prompt"] = system_prompt
        self.run_task(LLMTask, images, extra=extra)

    def run_unmask(self, images: List[ImageData]):
        self.run_task(UnmaskTask, images)

    def run_mask_text(self, images: List[ImageData]):
        self.run_task(MaskTextTask, images)

    def run_restore(self, images: List[ImageData]):
        self.run_task(RestoreTask, images)

    # ============================================================
    # Signal Handlers
    # ============================================================
    def on_pipeline_progress(self, current, total, filename, speed=0.0):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        # 取得任務與模型資訊
        task_name = self._current_task.name if self._current_task else "TASK"
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
        
        if speed > 0:
            if speed < 1:
                speed_str = f"{1/speed:.2f} s/it"
            else:
                speed_str = f"{speed:.2f} it/s"
        else:
            speed_str = "..."

        # Format: [MODEL] Filename at StatusBar
        self.statusBar().showMessage(f"{model_info} | {os.path.basename(filename)}")

        # Helper text on ProgressBar: (current/total) - Speed
        final_msg = f"{current}/{total} | {speed_str}"
            
        self.progress_bar.setFormat(final_msg)
        self.btn_cancel_batch.setVisible(True)
        self.btn_cancel_batch.setEnabled(True)

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

        task_name = self._current_task.name if self._current_task else ""
        
        # Tagger
        if "tagger" in task_name:
             if output.result_text:
                 self.save_tagger_tags_for_image(image_path, output.result_text)
                 
             if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                 self.tagger_tags = self.load_tagger_tags_for_current_image()
                 self.refresh_tags_tab()
             
             # Check write to txt (Batch Task flag OR UI Checkbox)
             write_to_txt = getattr(self, "_is_batch_to_txt", False)
             if not write_to_txt and hasattr(self, 'chk_tags_save_txt') and self.chk_tags_save_txt.isChecked():
                 write_to_txt = True

             if write_to_txt and output.result_text:
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
                     
                 # Check write to txt (Batch Task flag OR UI Checkbox)
                 write_to_txt = getattr(self, "_is_batch_to_txt", False)
                 if not write_to_txt and hasattr(self, 'chk_llm_save_txt') and self.chk_llm_save_txt.isChecked():
                     write_to_txt = True

                 if write_to_txt:
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
