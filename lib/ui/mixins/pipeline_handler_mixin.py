from typing import TYPE_CHECKING, List, Optional, Dict, Any

import lib.runtime.command_actions as command_actions
import lib.runtime.control_plane as control_plane
import lib.runtime.command_catalog as command_catalog
import lib.runtime.host_support as host_support
import lib.runtime.pipeline_callbacks as pipeline_callbacks
import lib.runtime.runtime_api as runtime_api
import lib.runtime.runtime_regression as runtime_regression
import lib.runtime.state_adapter as state_adapter
import lib.runtime.task_facade as task_facade
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

    _get_runtime_event_bus = host_support.get_runtime_event_bus
    _get_command_registry = host_support.get_command_registry
    _build_initial_runtime_state = host_support.build_initial_runtime_state
    _get_runtime_app_state = host_support.get_runtime_app_state
    _get_runtime_state_store = host_support.get_runtime_state_store
    _update_runtime_state_section = host_support.update_runtime_state_section
    _replace_runtime_state_section = host_support.replace_runtime_state_section
    _get_task_runner = host_support.get_task_runner
    _subscribe_runtime_command_state = host_support.subscribe_runtime_command_state
    _get_runtime_ui_spec_store = host_support.get_runtime_ui_spec_store
    _get_runtime_service_watcher = host_support.get_runtime_service_watcher
    _start_runtime_http_bridge = host_support.start_runtime_http_bridge
    _maybe_start_runtime_http_bridge = host_support.maybe_start_runtime_http_bridge
    _maybe_start_runtime_service_watch = host_support.maybe_start_runtime_service_watch
    _stop_runtime_http_bridge = host_support.stop_runtime_http_bridge
    _stop_runtime_service_watch = host_support.stop_runtime_service_watch
    _register_runtime_command = host_support.register_runtime_command

    def _register_runtime_commands(self) -> None:
        if getattr(self, "_runtime_commands_registered", False):
            return
        command_catalog.register_runtime_commands(self)
        self._runtime_commands_registered = True

    get_runtime_events = runtime_api.get_runtime_events
    get_runtime_event_profiles = runtime_api.get_runtime_event_profiles_api
    get_runtime_state = runtime_api.get_runtime_state

    get_agent_manifest = control_plane.get_agent_manifest

    get_ui_spec = runtime_api.get_ui_spec
    get_ui_spec_override = runtime_api.get_ui_spec_override
    _emit_ui_spec_updated = runtime_api.emit_ui_spec_updated
    update_ui_spec = runtime_api.update_ui_spec
    replace_ui_spec_override = runtime_api.replace_ui_spec_override
    patch_ui_node = runtime_api.patch_ui_node
    reset_ui_spec = runtime_api.reset_ui_spec

    get_runtime_capabilities = control_plane.get_runtime_capabilities

    command_shutdown_app = runtime_api.command_shutdown_app

    get_runtime_bridge_status = control_plane.get_runtime_bridge_status

    _runtime_settings_dict = runtime_api.runtime_settings_dict
    _replace_runtime_settings_dict = runtime_api.replace_runtime_settings_dict
    get_runtime_settings = runtime_api.get_runtime_settings
    get_runtime_settings_schema = runtime_api.get_runtime_settings_schema
    update_runtime_settings = runtime_api.update_runtime_settings

    get_runtime_workers = control_plane.get_runtime_workers
    command_list_reloadable_runtime_services = control_plane.command_list_reloadable_runtime_services
    command_get_reload_policy = control_plane.command_get_reload_policy
    command_reload_runtime_services = control_plane.command_reload_runtime_services
    get_runtime_service_watch_status = control_plane.get_runtime_service_watch_status
    command_start_runtime_service_watch = control_plane.command_start_runtime_service_watch
    command_stop_runtime_service_watch = control_plane.command_stop_runtime_service_watch
    get_worker_services_status = control_plane.get_worker_services_status
    command_reload_worker_services = control_plane.command_reload_worker_services
    command_stop_worker_services = control_plane.command_stop_worker_services

    get_task_status = runtime_api.get_task_status

    _sync_runtime_settings_state = state_adapter.sync_runtime_settings_state
    _runtime_state_section = state_adapter.runtime_state_section
    _runtime_selection_value = state_adapter.runtime_selection_value
    _runtime_controls_value = state_adapter.runtime_controls_value
    _runtime_content_value = state_adapter.runtime_content_value
    _runtime_current_image_path = state_adapter.runtime_current_image_path
    _runtime_loaded_image_paths = state_adapter.runtime_loaded_image_paths
    _runtime_all_image_paths = state_adapter.runtime_all_image_paths
    _runtime_filtered_image_paths = state_adapter.runtime_filtered_image_paths
    _runtime_prompt_text = state_adapter.runtime_prompt_text
    _runtime_image_process_prompt_text = state_adapter.runtime_image_process_prompt_text
    _runtime_txt_content = state_adapter.runtime_txt_content
    _runtime_tagger_save_to_txt = state_adapter.runtime_tagger_save_to_txt
    _runtime_llm_save_to_txt = state_adapter.runtime_llm_save_to_txt
    _selection_command_result = state_adapter.selection_command_result
    _ui_command_result = state_adapter.ui_command_result
    _content_command_result = state_adapter.content_command_result
    _set_runtime_selection_values = state_adapter.set_runtime_selection_values
    _sync_runtime_selection_state = state_adapter.sync_runtime_selection_state
    _sync_runtime_task_state = state_adapter.sync_runtime_task_state
    _sync_runtime_ui_state = state_adapter.sync_runtime_ui_state
    _sync_runtime_content_state = state_adapter.sync_runtime_content_state
    _sync_runtime_controls_state = state_adapter.sync_runtime_controls_state
    _sync_runtime_tags_state = state_adapter.sync_runtime_tags_state

    execute_command = runtime_api.execute_command

    _build_images_from_paths = task_facade.build_images_from_paths
    _get_loaded_image_paths = task_facade.get_loaded_image_paths
    _start_named_runtime_task = task_facade.start_named_runtime_task

    command_open_directory = command_actions.command_open_directory
    command_set_root_dir = command_actions.command_set_root_dir
    command_apply_filter = command_actions.command_apply_filter
    command_clear_filter = command_actions.command_clear_filter
    command_prev_image = command_actions.command_prev_image
    command_next_image = command_actions.command_next_image
    command_first_image = command_actions.command_first_image
    command_last_image = command_actions.command_last_image
    command_jump_to_index = command_actions.command_jump_to_index
    command_set_view_mode = command_actions.command_set_view_mode
    command_set_active_tab = command_actions.command_set_active_tab
    command_set_control_value = command_actions.command_set_control_value
    command_delete_current_image = command_actions.command_delete_current_image
    command_add_custom_tag = command_actions.command_add_custom_tag
    command_use_default_prompt = command_actions.command_use_default_prompt
    command_use_custom_prompt = command_actions.command_use_custom_prompt
    command_use_default_image_prompt = command_actions.command_use_default_image_prompt
    command_set_prompt_text = command_actions.command_set_prompt_text
    command_set_image_process_prompt_text = command_actions.command_set_image_process_prompt_text
    command_set_txt_content = command_actions.command_set_txt_content
    command_open_find_replace = command_actions.command_open_find_replace
    command_editor_undo = command_actions.command_editor_undo
    command_editor_redo = command_actions.command_editor_redo
    command_prev_nl_page = command_actions.command_prev_nl_page
    command_next_nl_page = command_actions.command_next_nl_page
    command_start_runtime_bridge = command_actions.command_start_runtime_bridge
    command_stop_runtime_bridge = command_actions.command_stop_runtime_bridge
    command_scan_workers = command_actions.command_scan_workers
    command_run_tagger = command_actions.command_run_tagger
    command_run_tagger_loaded = command_actions.command_run_tagger_loaded
    command_run_llm = command_actions.command_run_llm
    command_run_llm_loaded = command_actions.command_run_llm_loaded
    command_run_image_process = command_actions.command_run_image_process
    command_run_image_process_loaded = command_actions.command_run_image_process_loaded
    command_run_unmask = command_actions.command_run_unmask
    command_run_unmask_loaded = command_actions.command_run_unmask_loaded
    command_run_mask_text = command_actions.command_run_mask_text
    command_run_mask_text_loaded = command_actions.command_run_mask_text_loaded
    command_run_restore = command_actions.command_run_restore
    command_run_restore_loaded = command_actions.command_run_restore_loaded
    command_smoke_command = command_actions.command_smoke_command
    command_run_runtime_regression = runtime_regression.command_run_runtime_regression
    
    is_task_running = task_facade.is_task_running
    stop_current_task = task_facade.stop_current_task
    run_task = task_facade.run_task
    _get_current_settings_obj = task_facade.get_current_settings_obj
    _on_task_done = task_facade.on_task_done

    # ============================================================
    # Convenience Methods (Helpers)
    # ============================================================

    run_tagger = task_facade.run_tagger
    run_llm = task_facade.run_llm
    run_unmask = task_facade.run_unmask
    run_mask_text = task_facade.run_mask_text
    run_image_process = task_facade.run_image_process
    run_restore = task_facade.run_restore

    # ============================================================
    # Signal Handlers
    # ============================================================
    on_pipeline_progress = pipeline_callbacks.on_pipeline_progress
    on_pipeline_error = pipeline_callbacks.on_pipeline_error
    on_pipeline_image_done = pipeline_callbacks.on_pipeline_image_done
