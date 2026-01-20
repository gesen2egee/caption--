from typing import TYPE_CHECKING
import os

from lib.workers.worker import WorkerOutput
from lib.utils.parsing import extract_llm_content_and_postprocess

if TYPE_CHECKING:
    from lib.ui.main_window import MainWindow

class PipelineHandlerMixin:
    """
    Mixin handling Pipeline callback events.
    """
    def on_pipeline_progress(self, current, total, filename):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        msg = self.tr("status_batch_progress_fmt").replace("{current}", str(current)).replace("{total}", str(total)).replace("{filename}", filename)
        self.progress_bar.setFormat(msg)
        self.btn_cancel_batch.setVisible(True)

    def on_pipeline_error(self, err_msg):
        self.statusBar().showMessage(self.tr("msg_error_fmt").replace("{msg}", str(err_msg)), 8000)
        self.progress_bar.setVisible(False)
        self.btn_auto_tag.setEnabled(True)
        self.btn_auto_tag.setText(self.tr("btn_auto_tag"))
        self.btn_run_llm.setEnabled(True)
        self.btn_run_llm.setText(self.tr("btn_run_llm"))
        self.set_batch_ui_enabled(True) 

    def on_pipeline_image_done(self, image_path: str, output: WorkerOutput):
        if not output.success:
            print(f"Error for {image_path}: {output.error}")
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
             if output.result_data and "original_path" in output.result_data:
                  old_path = output.result_data.get("original_path")
                  new_path = output.result_data.get("result_path") or image_path
                  self._replace_image_path_in_list(old_path, new_path)
             
             if self.current_image_path:
                 self.load_image()

                 
        # Restore
        elif "restore" in task_name:
             if self.current_image_path and os.path.abspath(image_path) == os.path.abspath(self.current_image_path):
                 self.load_image()
