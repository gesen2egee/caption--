# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any

from lib.pipeline.context import TaskResult


def on_pipeline_progress(host: Any, current, total, filename, speed=0.0):
    host.progress_bar.setVisible(True)
    host.progress_bar.setMaximum(total)
    host.progress_bar.setValue(current)

    task_name = host._current_task.name if host._current_task else "TASK"
    model_info = task_name.upper()

    if "tagger" in task_name:
        model = host.settings.get("tagger_model", "")
        if "/" in model:
            model = model.split("/")[-1]
        model_info = f"TAGGER ({model})"
    elif "llm" in task_name:
        if host.settings.get("llm_provider") == "llm_llama_cpp_local":
            model = host.settings.get("llama_cpp_model_alias", "qwen35-vl-gguf")
        else:
            model = host.settings.get("llm_model", "")
        model_info = f"LLM ({model})"
    elif "image_process" in task_name:
        model = host.settings.get("image_process_model", "unsloth/FLUX.2-klein-4B-GGUF")
        model_info = f"IMG ({model})"
    elif "unmask" in task_name:
        mode = host.settings.get("mask_remover_mode", "base")
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

    host.statusBar().showMessage(f"{model_info} | {os.path.basename(filename)}")
    host.progress_bar.setFormat(f"{current}/{total} | {speed_str}")
    host.btn_cancel_batch.setVisible(True)
    host.btn_cancel_batch.setEnabled(True)


def on_pipeline_error(host: Any, err_msg):
    host.statusBar().showMessage(host.tr("msg_error_fmt").replace("{msg}", str(err_msg)), 8000)
    host.progress_bar.setVisible(False)
    host.btn_auto_tag.setEnabled(True)
    host.btn_auto_tag.setText(host.tr("btn_auto_tag"))
    host.btn_run_llm.setEnabled(True)
    host.btn_run_llm.setText(host.tr("btn_run_llm"))
    if hasattr(host, "btn_run_imgproc"):
        host.btn_run_imgproc.setEnabled(True)
        host.btn_run_imgproc.setText(host.tr("btn_run_imgproc"))
    host.set_batch_ui_enabled(True)


def on_pipeline_image_done(host: Any, image_path: str, output: TaskResult):
    if not output.success:
        print(f"Error for {image_path}: {output.error}")
        return

    if output.skipped:
        reason = output.skip_reason or host.tr("msg_task_skipped")
        host.statusBar().showMessage(f"{os.path.basename(image_path)}: {reason}", 5000)
        return

    task_name = host._current_task.name if host._current_task else ""
    current_image_path = host._runtime_current_image_path()

    if "tagger" in task_name:
        if output.result_text:
            host.save_tagger_tags_for_image(image_path, output.result_text)

        if current_image_path and os.path.abspath(image_path) == os.path.abspath(current_image_path):
            host.tagger_tags = host.load_tagger_tags_for_current_image()
            host.refresh_tags_tab()

        write_to_txt = getattr(host, "_is_batch_to_txt", False)
        if not write_to_txt and host._runtime_tagger_save_to_txt():
            write_to_txt = True

        if write_to_txt and output.result_text:
            host.write_batch_result_to_txt(image_path, output.result_text, is_tagger=True)

    elif "llm" in task_name:
        from lib.utils.parsing import extract_llm_content_and_postprocess

        content = output.result_text or ""
        final_content = extract_llm_content_and_postprocess(content, host.english_force_lowercase)

        if final_content:
            host.save_nl_for_image(image_path, final_content)
            if current_image_path and os.path.abspath(image_path) == os.path.abspath(current_image_path):
                if final_content not in host.nl_pages:
                    host.nl_pages.append(final_content)
                host.nl_page_index = len(host.nl_pages) - 1
                host.nl_latest = final_content
                host.refresh_nl_tab()
                host.update_nl_page_controls()
                host.on_text_changed()

            write_to_txt = getattr(host, "_is_batch_to_txt", False)
            if not write_to_txt and host._runtime_llm_save_to_txt():
                write_to_txt = True

            if write_to_txt:
                host.write_batch_result_to_txt(image_path, final_content, is_tagger=False)

    elif "image_process" in task_name:
        if output.result_data:
            old_path = output.result_data.get("original_path", image_path)
            new_path = output.result_data.get("result_path", image_path)
            if new_path and old_path and os.path.abspath(new_path) != os.path.abspath(old_path):
                host._replace_image_path_in_list(old_path, new_path)

        if current_image_path and os.path.abspath(image_path) == os.path.abspath(current_image_path):
            host.load_image()
        elif current_image_path and output.result_data:
            old_path = output.result_data.get("original_path")
            if old_path and os.path.abspath(old_path) == os.path.abspath(current_image_path):
                host.load_image()

    elif "unmask" in task_name or "mask_text" in task_name:
        if output.result_data:
            old_path = output.result_data.get("original_path")
            new_path = output.result_data.get("result_path")

            if "mask_text" in task_name and output.result_data.get("box_count", 0) == 0:
                host.statusBar().showMessage(host.tr("msg_no_text_detected"), 4000)
            elif new_path:
                host._replace_image_path_in_list(old_path, new_path)
                host.statusBar().showMessage(host.tr("status_done"), 3000)

        if current_image_path:
            host.load_image()

    elif "restore" in task_name:
        if output.result_data:
            old_path = output.result_data.get("original_path")
            new_path = output.result_data.get("result_path")
            if old_path and new_path and os.path.abspath(old_path) != os.path.abspath(new_path):
                host._replace_image_path_in_list(old_path, new_path)
        if current_image_path and os.path.abspath(image_path) == os.path.abspath(current_image_path):
            host.load_image()
