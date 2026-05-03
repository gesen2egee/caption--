export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonObject
  | JsonValue[];

export interface JsonObject {
  [key: string]: JsonValue;
}

export interface RuntimeEvent {
  name: string;
  payload: JsonObject;
  timestamp: string;
}

export interface RuntimeEventProfile {
  description: string;
  include_prefixes: string[];
  exclude_prefixes: string[];
}

export interface CommandMetadata {
  category: string;
  category_label?: string;
  access_modes?: string[];
  read_only?: boolean;
  writes_files?: boolean;
  destructive?: boolean;
  requires_confirmation?: boolean;
  mutates_ui?: boolean;
  mutates_settings?: boolean;
  mutates_runtime?: boolean;
  long_running?: boolean;
  risk_level?: "low" | "medium" | "high" | string;
  notes?: string[];
}

export interface RuntimeErrorInfo {
  code: string;
  type: string;
  message: string;
  source: string;
  command_name?: string;
  access_mode?: string;
  allowed_access_modes?: string[];
  status_code?: number;
  traceback?: string;
  details?: JsonObject;
}

export interface AgentManifest {
  agent_mode: string;
  available_modes: string[];
  default_mode: string;
  skills_note: string;
  recommended_event_profile: string;
  event_profiles: Record<string, RuntimeEventProfile>;
  recommended_commands: string[];
  allowed_commands: string[];
  restricted_commands: string[];
  command_examples: Record<string, CommandRequest>;
  command_metadata?: Record<string, CommandMetadata>;
  restricted_command_metadata?: Record<string, CommandMetadata>;
  ui_mutation_allowed: boolean;
  settings_mutation_allowed: boolean;
  destructive_file_ops_allowed: boolean;
  bridge: BridgeStatus;
  routes: Record<string, string>;
  endpoints: Record<string, string | null>;
}

export interface BridgeStatus {
  enabled: boolean;
  host: string | null;
  port: number | null;
  url: string | null;
}

export interface WorkerInfo {
  name: string;
  display_name: string;
}

export interface WorkerServiceInfo {
  category: string;
  worker_name: string;
  pid?: number | null;
  alive?: boolean;
  started_at?: number | null;
  uptime_seconds?: number;
  request_count?: number;
  stderr_tail?: string[];
  python_executable?: string;
}

export interface WorkersSummary {
  categories: Record<string, WorkerInfo[]>;
  available_categories: string[];
  runtime_mode?: string;
  service_manager?: {
    service_count: number;
    services: WorkerServiceInfo[];
  };
}

export interface RuntimeSelectionState {
  root_dir_path: string;
  current_image_path: string;
  current_index: number;
  image_count: number;
  loaded_image_paths?: string[];
  all_image_paths?: string[];
  filtered_image_paths?: string[];
  has_raw_backup?: boolean;
  filter_active?: boolean;
  filter_query?: string;
  filter_tags?: boolean;
  filter_text?: boolean;
}

export interface RuntimeTaskState {
  running: boolean;
  task_name: string | null;
}

export interface RuntimeCommandEntry {
  command_name?: string | null;
  args?: JsonValue;
  kwargs?: JsonValue;
  result?: JsonValue;
  error?: string | null;
  error_info?: RuntimeErrorInfo | null;
  timestamp?: string | null;
}

export interface RuntimeCommandsState {
  last_started?: RuntimeCommandEntry | null;
  last_finished?: RuntimeCommandEntry | null;
  last_failed?: RuntimeCommandEntry | null;
}

export interface RuntimeUiState {
  current_view_mode: number;
  temp_view_mode: number | null;
  current_prompt_mode: string;
  nl_page_index: number;
  nl_page_count: number;
  current_tab_index: number;
}

export interface RuntimeContentState {
  prompt_text: string;
  image_process_prompt_text: string;
  txt_content: string;
  nl_latest: string;
}

export interface RuntimeControlsState {
  current_index: number;
  filter_query: string;
  filter_tags: boolean;
  filter_text: boolean;
  view_mode: number;
  tagger_save_to_txt: boolean;
  llm_save_to_txt: boolean;
  [key: string]: JsonValue;
}

export interface RuntimeTagsState {
  folder_meta: string[];
  custom: string[];
  tagger: string[];
  nl: string[];
  translations?: Record<string, string>;
}

export interface RuntimeState {
  settings: JsonObject;
  selection: RuntimeSelectionState;
  task: RuntimeTaskState;
  commands?: RuntimeCommandsState;
  ui: RuntimeUiState;
  controls: RuntimeControlsState;
  content: RuntimeContentState;
  tags: RuntimeTagsState;
}

export interface UiSpecNode {
  id?: string;
  type?: string;
  title?: string;
  hidden?: boolean;
  children?: UiSpecNode[];
  sections?: UiSpecNode[];
  tabs?: UiSpecNode[];
  controls?: UiSpecNode[];
  [key: string]: unknown;
}

export interface LegacyUiSpec {
  version: string;
  layout: UiSpecNode;
  state: RuntimeState;
}

export interface CommandRequest {
  args?: JsonValue[];
  kwargs?: JsonObject;
}

export interface UiCaptureSaveItem {
  name: string;
  scope: string;
  png_data_url: string;
  metadata: JsonObject;
}

export interface UiCaptureSavedArtifact {
  name: string;
  scope: string;
  image_path: string;
  metadata_path: string;
  metadata: JsonObject;
}

export interface UiCaptureSaveResponse {
  ok: boolean;
  capture_dir: string;
  captures: UiCaptureSavedArtifact[];
}

export interface CommandResponse<T = JsonValue> {
  ok: boolean;
  result?: T;
  error?: string;
  error_info?: RuntimeErrorInfo;
}

export interface CommandsSummary {
  commands: string[];
  task_status: RuntimeTaskState;
  event_count: number;
  state: RuntimeState;
  bridge: BridgeStatus;
}

export interface SettingField {
  key: string;
  group: string;
  type: string;
  default: JsonValue;
  value: JsonValue;
  options?: JsonValue[];
}

export interface SettingsSchema {
  version: string;
  field_count: number;
  groups: string[];
  fields: SettingField[];
}

export interface RuntimeCapabilities {
  commands: string[];
  command_access_modes?: Record<string, string[]>;
  command_metadata?: Record<string, CommandMetadata>;
  bridge: BridgeStatus;
  workers: WorkersSummary;
  settings_schema_version?: string;
  ui_spec_version?: string;
  ui_spec_override_active?: boolean;
  control_ids?: string[];
  spec_patch_supported?: boolean;
  command_result_tracking?: boolean;
  structured_errors?: boolean;
  event_streaming?: boolean;
  event_history_limit?: number;
  event_filtering?: boolean;
  event_profiles?: Record<string, RuntimeEventProfile>;
  agent_modes?: string[];
  runtime_host_backend?: string;
  task_runtime_backend?: string;
  task_runtime_dispatcher?: string;
  task_runtime_listener_api?: boolean;
  task_runtime_cancel_events?: boolean;
  worker_runtime_mode?: string;
  worker_service_reload_supported?: boolean;
  runtime_service_reload_supported?: boolean;
  runtime_reloadable_services?: Record<string, string>;
  runtime_service_watch_supported?: boolean;
  runtime_service_watch_status?: {
    enabled: boolean;
    interval_seconds?: number | null;
    service_count?: number;
    services?: Record<string, { module?: string; path?: string; mtime?: number | null }>;
    last_reload?: JsonValue;
    last_error?: string | null;
  };
  qt_residual_components?: string[];
  command_examples?: Record<string, CommandRequest>;
}
