# Agent Runtime Refactor

## Goal
- Keep current UI layout and user-facing behavior as close as possible.
- Replace the current PyQt-coupled architecture with a new runtime-oriented architecture.
- Enable development-grade Agent workflows:
  - Agent can trigger app commands directly.
  - Agent can observe runtime events and hot-update signals.
  - UI spec can be updated without restarting the whole app.
  - Backend configuration and orchestration can be adjusted at runtime where safe.

## Agent Mode
- Mode: development-grade
- Agent responsibilities:
  - update UI spec
  - trigger commands
  - inspect runtime events
  - run smoke tests
  - optionally patch backend code and request worker reload
- Guardrails:
  - all agent-executable functions must go through an explicit command registry
  - all runtime updates must emit structured events
  - destructive operations should stay opt-in

## Skills Note
- No dedicated session skill was required for this decomposition step.
- Future work should keep a short note in task logs when Agent-facing interfaces or test hooks change.
- If later steps involve browser automation or external UI testing, use the `playwright` skill.
- Use `tasks/feature_template.md` as the default shape for new runtime-first features.

## Current Architecture Baseline

### Entry / Runtime
- `run.bat` activates `venv`, patches NVIDIA DLL paths, then runs `python caption.py`.
- `caption.py` creates `PyQt6.QtWidgets.QApplication` and opens `lib.ui.main_window.MainWindow`.

### UI
- Main UI is concentrated in `lib/ui/main_window.py`.
- UI behavior is split with PyQt mixins:
  - `batch_mixin.py`
  - `navigation_mixin.py`
  - `editor_mixin.py`
  - `processing_mixin.py`
  - `settings_mixin.py`
  - `pipeline_handler_mixin.py`
- There are custom PyQt widgets/dialogs:
  - `components/tag_flow.py`
  - `components/stroke.py`
  - `dialogs/find_replace.py`
  - `dialogs/settings_dialog.py`

### Task / Worker Flow
- Tasks live in `lib/pipeline/tasks/`.
- `BaseTask` inherits `QThread` and exposes `pyqtSignal`.
- `PipelineHandlerMixin` directly connects task progress/results to UI widgets.
- Workers are discovered via `lib/workers/registry.py`.
- Business logic inside task `execute()` methods is largely reusable.

### Reusable Core
- `lib/core/dataclasses.py`
- `lib/core/settings.py`
- most of `lib/utils/*`
- most worker implementations under `lib/workers/*`

## Main Refactor Constraint
- Do not redesign product behavior in phase 1.
- Do not redesign page layout in phase 1.
- First target is architectural replacement with compatibility behavior.

## What Must Be Decoupled First

### 1. Task runtime from Qt
Current problem:
- `BaseTask` is a `QThread`
- task lifecycle is tied to PyQt signal delivery

Target:
- create a UI-agnostic task runner
- events become plain structured payloads instead of `pyqtSignal`

Suggested replacement:
- `TaskRunner`
- `TaskEvent`
- `TaskHandle`
- optional worker-process isolation later

### 2. UI actions from widget callbacks
Current problem:
- button handlers call app methods directly
- methods assume widget state exists

Target:
- move callable behavior into a command layer

Suggested replacement:
- `CommandRegistry`
- commands such as:
  - `open_folder`
  - `select_image`
  - `run_tagger`
  - `run_llm`
  - `run_image_process`
  - `run_unmask`
  - `run_mask_text`
  - `run_restore`
  - `save_settings`
  - `reload_worker`

### 3. Runtime state from widget instances
Current problem:
- `MainWindow` stores app state and UI references together

Target:
- introduce app/session state that can be consumed by either PyQt or React

Suggested replacement:
- `AppState`
- `SessionState`
- `SelectionState`
- `SettingsState`
- `TaskState`

### 4. UI definition from imperative widget construction
Current problem:
- UI tree is hard-coded with PyQt widgets

Target:
- describe the existing layout in a JSON-compatible UI spec
- keep first version close to current layout

## Proposed New Architecture

### Layer 1: Core domain
- image metadata
- settings
- file operations
- sidecar operations
- worker registry
- task execute logic

### Layer 2: Application runtime
- command registry
- task runner
- event bus
- app/session state store
- hot-reload manager

### Layer 3: Python model workers
- keep model calls in Python
- keep heavy model dependencies isolated from UI runtime
- allow independent restart/reload when possible

### Layer 4: UI runtime
- React renderer
- TypeScript runtime
- JSON UI spec
- receives state snapshots and event updates

### Layer 5: Agent integration
- subscribes to selected runtime events
- can invoke commands through the same registry
- can update UI spec in development mode
- can run smoke tests without driving raw widgets

## Hot Update Scope

### Safe runtime updates
- UI spec
- labels and layout metadata
- visibility and ordering
- settings values
- model selection
- prompt templates
- command wiring

### Controlled runtime updates
- worker reload
- command implementation swap behind stable command IDs
- task orchestration policy changes

### Not guaranteed hot-swappable
- arbitrary Python code changes inside worker logic
- low-level model runtime changes during active jobs
- deep dependency reload across all imported modules

## Agent Event Surface
- `ui.spec.updated`
- `settings.updated`
- `command.started`
- `command.finished`
- `command.failed`
- `task.started`
- `task.progress`
- `task.image_done`
- `task.batch_done`
- `task.failed`
- `worker.scan.started`
- `worker.scan.finished`
- `worker.reloaded`
- `editor.find_replace.completed`
- `image.deleted`
- `image.stroke_erased`
- `tags.custom.added`

## Agent Command Surface
- `app.get_state`
- `app.get_ui_spec`
- `app.update_ui_spec`
- `settings.get`
- `settings.update`
- `workers.scan`
- `workers.list`
- `workers.reload`
- `task.run_tagger`
- `task.run_llm`
- `task.run_image_process`
- `task.run_unmask`
- `task.run_mask_text`
- `task.run_restore`
- `task.cancel`
- `test.smoke_command`

## Current Runtime Guardrails
- command registry now stores structured metadata for each command:
  - category
  - access modes
  - read-only vs mutating
  - writes-files
  - destructive
  - requires-confirmation
  - mutates-ui
  - mutates-settings
  - mutates-runtime
  - long-running
  - risk-level
- runtime capabilities expose `command_metadata`
- agent manifest exposes:
  - `command_metadata` for allowed commands
  - `restricted_command_metadata` for blocked commands in the selected mode
- safe/development mode now has both:
  - execution-time gating
  - inspectable guardrail metadata for Agent and web clients
- the React shell now follows the selected mode for actual command execution and disables blocked actions up front instead of relying only on backend `403` responses

## Current Entry State
- `caption.py` now supports `--ui=qt|web|shell|service`
- `web` mode now starts a pure Python headless runtime host plus the runtime bridge
- `shell` mode keeps the legacy Qt window visible while a Vite dev shell is used as the primary runtime UI
- `service` mode starts the same pure Python headless runtime host without opening a browser
- `run.bat` is now the single launcher:
  - `run.bat qt`
  - `run.bat web`
  - `run.bat shell`
  - `run.bat service`
- default `run.bat` behavior prefers `web` when `frontend/dist/index.html` exists, otherwise falls back to `qt`
- headless web/service mode now has a formal runtime exit command: `app.shutdown`

## Current Task Runtime State
- `BaseTask` no longer subclasses `QThread`
- execution backend is now plain Python thread
- UI/runtime callback delivery is marshalled back through a Qt callback dispatcher
- task progress/result/error delivery is no longer signal-only:
  - `BaseTask.add_listener(...)`
  - `BaseTask.run_inline()`
  - `BaseTask.execute_pipeline()`
- `TaskRunner` now prefers the listener API and dispatches lifecycle callbacks onto the Qt main thread
- `task.batch_done` now includes `cancelled`
- a separate `task.cancelled` event is now emitted after cooperative stop
- legacy compat workers are now plain Python threads too
- worker scan background execution is also plain Python thread now
- current Qt execution residue is effectively limited to:
  - `runtime.callback_dispatcher`

## Phase Breakdown

### Phase 0: Baseline and compatibility
- document current flows
- preserve current behavior as migration target
- add task log for architecture migration

### Phase 1: Extract runtime boundary
- introduce `CommandRegistry`
- introduce `EventBus`
- introduce UI-agnostic `TaskRunner`
- keep current PyQt UI as adapter client

### Phase 2: Extract app state
- move non-widget state out of `MainWindow`
- make PyQt UI read/write through state + commands

### Phase 3: Add development test hooks
- expose command execution API
- expose runtime event stream
- add smoke-test commands for Agent

### Phase 4: Add new UI runtime
- implement React shell
- define JSON spec for current layout
- map command bindings to existing behavior

### Phase 5: Switch default UI
- keep Python workers
- change startup flow
- keep a fallback path to legacy PyQt during transition

## First Concrete Extraction Targets
- `lib/pipeline/tasks/base_task.py`
- `lib/ui/mixins/pipeline_handler_mixin.py`
- `lib/ui/main_window.py`
- `lib/core/settings.py`
- `lib/workers/registry.py`

## Acceptance Criteria For Phase 1
- current PyQt UI still works
- task execution no longer requires UI handlers to own lifecycle logic
- runtime emits structured events for progress and completion

## Latest Progress
- added `lib/runtime/backend_host.py`, a pure Python runtime host for `web` and `service` modes
- `caption.py --ui=web` no longer creates `QApplication/MainWindow`; only `qt` and `shell` do
- `run.bat service` now starts a headless runtime bridge without opening a browser
- runtime capabilities now expose `runtime_host_backend`, and headless mode reports:
  - `runtime_host_backend=headless-python`
  - `task_runtime_dispatcher=direct-thread-callback`
  - `qt_residual_components=[]`
- runtime state store and event bus are now guarded by locks, so concurrent task/event updates are safer in headless/service mode
- headless runtime import path no longer eagerly imports `PyQt6`; the callback dispatcher and PyQt-dependent mixins now lazy-load Qt only when the legacy desktop path or Qt-specific dialogs/canvas are actually used
- extracted pure backend services for file selection/filter/delete and editor find-replace:
  - `lib/runtime/batch_service.py`
  - `lib/runtime/selection_service.py`
  - `lib/runtime/editor_service.py`
- extracted pure backend processing service:
  - `lib/runtime/processing_service.py`
- `NavigationMixin`, `EditorMixin`, and `BatchMixin` now delegate core file/text/batch sidecar operations to those services, reducing UI-mixin ownership of domain logic
- `ProcessingMixin` now delegates mask loading, prompt resolution, and stroke-eraser image rewriting to the pure backend processing service
- extracted backend services are now consumed as reloadable modules, and runtime exposes:
  - `backend.list_reloadables`
  - `backend.reload_services`
  - `backend.watch_reload_start`
  - `backend.watch_reload_status`
  - `backend.watch_reload_stop`
- runtime event profiles now include `backend.*`, so agent/web control-plane subscriptions can observe manual reload and auto-reload events without switching to a custom namespace filter.
- backend service auto-reload watcher now defers module reload while a task is running, emits `backend.services.auto_reload_deferred`, and resumes applying pending changes once the runtime is idle again.
- current reloadable backend services:
  - `selection`
  - `editor`
  - `batch`
  - `processing`
  - `task`
  - `state_projection`
- low-level task orchestration is now centralized in `lib/runtime/task_service.py`, so `task.run_*` commands and convenience task launchers no longer need to import pipeline task classes directly inside `PipelineHandlerMixin`
- runtime state section shaping is now centralized in `lib/runtime/state_projection.py`, reducing `PipelineHandlerMixin` ownership of state serialization details
- headless host now supports runtime-driven:
  - folder loading
  - filtering/navigation
  - prompt/text updates
  - batch/current-image command execution
  - worker service reload/stop
  - backend service hot reload without restarting the host
- `run.bat web`, `run.bat shell`, and `run.bat service` now default to enabling backend service auto-reload watching.
- direct smoke tests now pass for:
  - `RuntimeBackendHost` import/startup
  - `selection.set_root_dir`
  - `editor.find_replace`
  - `task.run_restore` with `worker_runtime_mode=service`
  - `python caption.py --ui=service` with live `/health`, `/capabilities`, and `/agent/manifest`
- `editor.find_replace` now supports direct command execution with structured kwargs and structured result payload.
- `image.delete_current` now supports explicit `confirm=true` for agent/web execution and returns moved-file details.
- `tags.add_custom` now returns structured result data and emits `tags.custom.added`.
- runtime capabilities now include `command_examples` so agent/web tooling can discover safe invocation shapes.
- the web shell now includes a `Runtime Tools` panel for direct custom-tag, find/replace, and delete-current flows without opening desktop dialogs.
- `batch.run_tagger`, `batch.run_llm`, `batch.run_image_process`, `batch.run_unmask`, `batch.run_mask_text`, and `batch.run_restore` now exist as first-class runtime commands.
- legacy batch-only branches such as `save_to_txt`, `delete_chars`, sidecar restore short-circuit, and restore confirmation are now exposed as structured command inputs/results instead of only living inside `QMessageBox` flow.
- the runtime `ui-spec` batch buttons now call batch commands instead of falling back to raw `task.run_*_loaded`.
- single-image commands now also have high-level wrappers: `action.run_tagger_current`, `action.run_llm_current`, `action.run_image_process_current`, `action.run_unmask_current`, `action.run_mask_text_current`, `action.run_restore_current`.
- single-image preflight checks such as missing `{tags}` context, OCR disabled, and missing raw backup are now surfaced as structured `reason` payloads instead of only being visible through desktop modal dialogs.
- runtime selection state now includes `has_raw_backup`, allowing the web renderer to disable restore when no backup exists.
- command telemetry now includes summarized `args`, `kwargs`, and `result` in `command.started` / `command.finished`, and runtime state now keeps `commands.last_started`, `commands.last_finished`, and `commands.last_failed`.
- the HTTP bridge now exposes `GET /events/stream` as SSE, and the web shell now prefers event-driven refresh with slow polling only as fallback.
- runtime state events now emit section/key deltas instead of full snapshots on every update, reducing SSE payload size and event-log bloat.
- the web shell now applies known SSE updates directly to local state, settings schema, workers, and ui-spec, with full refresh kept as fallback for unknown events.
- runtime event history is now capped, and `/events` supports bounded reads, so long-running sessions do not grow event memory and initial event fetch payloads without limit.
- concrete section events such as `content.updated` and `selection.updated` now stand on their own; the runtime no longer emits a duplicate generic `state.updated` for every section sync.
- event history and SSE now support prefix-based include/exclude filtering, so Agent consumers can subscribe to higher-signal event slices instead of all diagnostic noise.
- the runtime now exposes named event profiles: `control` for high-signal command/task/worker side effects, `diagnostic` for verbose stable namespaces, and `all` for the unfiltered stream.
- the runtime now exposes `app.get_agent_manifest` and `/agent/manifest`, so Agent clients can bootstrap from one document instead of manually composing capabilities, event profiles, and bridge endpoints.
- the agent manifest now returns both stable relative `routes` and absolute `endpoints`, so it remains usable before the bridge URL is known.
- the agent manifest now supports `safe` and `development` command surfaces. `safe` excludes destructive file ops, settings mutation, and UI-spec mutation; `development` exposes the full refactor/debug surface.
- command execution itself is now mode-aware through registry metadata. Safe/development is no longer only advisory in the manifest; the bridge enforces it via `?mode=...`.
- `action.run_stroke_eraser_current` now exists as a structured current-image command, so stroke eraser can be triggered from Agent/web without opening the PyQt drawing dialog.
- the web shell `Runtime Tools` panel now accepts uploaded mask images and sends them through the stroke-eraser command surface.
- commands can be invoked without clicking widgets
- at least one smoke test can run through command execution

## Current Progress
- Added `lib/runtime` with:
  - `events.py`
  - `commands.py`
  - `state.py`
  - `task_runner.py`
- Migrated `lib/ui/mixins/pipeline_handler_mixin.py` to start tasks through `TaskRunner`
- Added runtime command registration for:
  - `app.get_runtime_events`
  - `app.get_runtime_state`
  - `app.get_task_status`
  - `app.get_ui_spec`
  - `app.get_ui_spec_override`
  - `app.update_ui_spec`
  - `app.replace_ui_spec_override`
  - `app.patch_ui_node`
  - `app.reset_ui_spec`
  - `app.get_capabilities`
  - `bridge.get_status`
  - `bridge.start`
  - `bridge.stop`
  - `settings.get`
  - `settings.schema`
  - `settings.update`
  - `workers.list`
  - `workers.scan`
  - `selection.set_root_dir`
  - `selection.apply_filter`
  - `selection.clear_filter`
  - `selection.prev`
  - `selection.next`
  - `selection.first`
  - `selection.last`
  - `selection.jump_to_index`
  - `ui.set_view_mode`
  - `ui.set_active_tab`
  - `content.set_prompt_text`
  - `content.set_image_process_prompt_text`
  - `content.set_txt_content`
  - `task.run_tagger`
  - `task.run_tagger_loaded`
  - `task.run_llm`
  - `task.run_llm_loaded`
  - `task.run_image_process`
  - `task.run_image_process_loaded`
  - `task.run_unmask`
  - `task.run_unmask_loaded`
  - `task.run_mask_text`
  - `task.run_mask_text_loaded`
  - `task.run_restore`
  - `task.run_restore_loaded`
  - `task.cancel`
  - `test.smoke_command`
- Initialized runtime interfaces during `MainWindow` startup
- Added a runtime state store with sections:
  - `settings`
  - `selection`
  - `task`
  - `ui`
  - `controls`
  - `content`
  - `tags`
- Added `lib/runtime/ui_spec.py` to expose a serializable legacy layout spec
- Added ephemeral runtime UI spec override support:
  - full override replace
  - deep-merge patch
  - node-by-id patch
  - reset to base spec
- Added `lib/runtime/http_bridge.py` for optional localhost access to:
  - state
  - events
  - ui spec
  - ui spec override
  - command execution
  - current preview image
  - capabilities metadata
  - worker metadata
  - settings schema
  - built frontend static assets
- Updated current PyQt action entrypoints to execute runtime commands instead of directly starting tasks
- Added runtime state sync for:
  - settings updates
  - folder/image selection changes
  - task lifecycle changes
  - prompt/text edits
  - control widgets
  - tag collections
  - current tab and view mode
- Added initial TypeScript-side runtime contract in `frontend/runtime/`
- Added a React/Vite migration shell in `frontend/` that consumes the live runtime bridge
- Added UI spec editing in the React shell:
  - quick node patch buttons
  - full override JSON editor
  - shell respects spec `title` and `hidden`
- Runtime commands now resolve prompt text, save-to-txt toggles, current image path, and loaded image lists through runtime state helpers first, with widget reads kept only as compatibility fallback.
- Selection runtime state now also mirrors `all_image_paths` and `filtered_image_paths`, and navigation/filter/editor commands tolerate missing filter/editor widgets more cleanly.
- Added a spec-renderer preview in the React shell that renders:
  - split/panel/tabs/toolbar
  - image preview
  - tag flows
  - text areas / plain text
  - progress
  - generic control bindings
- Promoted the spec renderer to the primary runtime workspace in the React shell
- Extended the legacy UI spec with:
  - utility actions
  - control labels
  - select options
  - section titles for tag/text areas
- Added schema-driven settings editing in the React shell using `settings.schema`
- Added `run_runtime_shell.bat` to start PyQt + HTTP bridge + Vite shell together
- Added `run_runtime_web.bat` to start PyQt + HTTP bridge + built web shell together

## Verification
- `python -m compileall lib\\runtime lib\\ui\\main_window.py lib\\ui\\mixins\\pipeline_handler_mixin.py lib\\ui\\mixins\\settings_mixin.py lib\\ui\\mixins\\navigation_mixin.py lib\\ui\\mixins\\processing_mixin.py lib\\ui\\mixins\\batch_mixin.py` passed
- Smoke test passed by creating `QApplication` + `MainWindow` and calling:
  - `window.execute_command("test.smoke_command")`
  - `window.execute_command("app.get_runtime_state")`
  - `window.execute_command("app.get_ui_spec")`
  - `window.execute_command("settings.update", {...})`
- Runtime bridge command test passed by calling:
  - `window.execute_command("bridge.start", ...)`
  - `window.execute_command("bridge.get_status")`
  - `window.execute_command("bridge.stop")`
- Runtime content / path command smoke test passed by calling:
  - `window.execute_command("content.set_prompt_text", ...)`
  - `window.execute_command("selection.set_root_dir", ...)`
- Runtime UI spec smoke test passed by calling:
  - `window.execute_command("app.patch_ui_node", node_id="nl_tab", patch={...})`
  - `window.execute_command("app.get_ui_spec_override")`
  - `window.execute_command("app.reset_ui_spec")`
- Runtime controls smoke test passed by calling:
  - `window.execute_command("ui.set_control_value", control_id="view_mode", value=2)`
  - `window.execute_command("ui.set_control_value", control_id="tagger_save_to_txt", value=False)`
  - `window.execute_command("ui.set_control_value", control_id="filter_query", value="background")`
- Runtime fallback smoke test passed by temporarily removing widgets and verifying:
  - `prompt_text`
  - `image_process_prompt_text`
  - `tagger_save_to_txt`
  - `llm_save_to_txt`
  - `loaded_image_paths`
  still resolve from runtime state
- Headless-ish selection/editor smoke test passed by temporarily removing filter/editor display widgets and verifying:
  - `apply_filter()`
  - `command_jump_to_index(...)`
  - `run_find_replace(..., scope_all=True)`
  - `delete_current_image(require_confirmation=False)`
  still work from runtime-backed selection/content state
- HTTP bridge responded successfully for:
  - `/health`
  - `/state`
  - `/commands`
  - `/bridge`
  - `/ui-spec-override`
  - `/preview/current` with expected `404` when no image is selected
  - `POST /commands/bridge.get_status`
- Real app process bridge test passed for:
  - `/`
  - `/capabilities`
  - `/workers`
  - `/settings-schema`
- Frontend shell build passed:
  - `cd frontend && npm run build`

## Bridge Notes
- Optional startup via environment:
  - `CAPTION_RUNTIME_HTTP=1`
  - `CAPTION_RUNTIME_HTTP_HOST=127.0.0.1`
  - `CAPTION_RUNTIME_HTTP_PORT=8765`
- The test script showed successful bridge responses, but the hosting Python process still returned exit code `1` after the Qt event loop ended. The bridge itself responded correctly; the non-zero exit appears to be related to the test harness process shutdown rather than request handling.

## Notes
- Python should remain the model execution layer.
- The new UI runtime should not call model code directly.
- The compatibility path matters more than elegance in the first pass.
- Command and bridge failures now expose structured `error_info` payloads with stable `code`, `source`, and `message`, and runtime command state stores that structured error data in `commands.last_failed`.
- Runtime state now has a central in-process `RuntimeAppState` layer ahead of `RuntimeStateStore`, so `get_runtime_state()` and runtime section reads no longer depend directly on the event projection snapshot.
- Runtime settings and root-directory updates now go through app-state-first helpers before syncing legacy `self.settings` / selection fields, and task startup now reads settings from that central app state instead of directly from widget-owned state.
- High-frequency selection writes such as root-dir changes, current index changes, filter application/clear, and image list swaps now begin to flow through central selection helpers before being projected back to runtime state, and headless `command_apply_filter(query=..., use_text=...)` now works without requiring filter widgets.
- Agent/capabilities surface metadata has been extracted into `lib/runtime/agent_surface.py`, so `PipelineHandlerMixin` no longer owns the static safe/development command lists, command metadata rules, and example payload catalog directly.
- Low-level task status/start/stop orchestration has now been extracted into `lib/runtime/task_orchestrator.py`, and the runtime reload catalog includes that module too.
- Runtime state read/write/result/sync helpers have now been extracted into `lib/runtime/state_adapter.py`, and runtime capabilities/manifest/bridge/workers control plane helpers have been extracted into `lib/runtime/control_plane.py`.
- `PipelineHandlerMixin` dropped again from 1450 lines to 1299 lines after delegating state-adapter and control-plane logic into runtime modules, and both modules are now part of the backend reload catalog.
- Runtime host lifecycle glue has now been extracted into `lib/runtime/host_support.py`, covering event bus, command registry bootstrapping, central app-state/store wiring, task-runner creation, command-state subscription, ui-spec store creation, bridge startup/shutdown, and service-watch startup/shutdown.
- `PipelineHandlerMixin` is now down to 1106 lines after delegating host-support glue as well, and `host_support` is now part of the backend reload catalog.
- Runtime command registration has now been extracted into `lib/runtime/command_catalog.py`, so `PipelineHandlerMixin` no longer carries the long static command binding list directly.
- `PipelineHandlerMixin` is now down to 1025 lines after delegating command registration too, and `command_catalog` is now part of the backend reload catalog.
- Runtime command handler bodies have now been extracted into `lib/runtime/command_actions.py`, and `PipelineHandlerMixin` now binds those commands as method aliases instead of owning the handler implementations directly.
- `PipelineHandlerMixin` is now down to 685 lines after delegating the selection/ui/content/task command actions too, and `command_actions` is now part of the backend reload catalog.
- Runtime task façade helpers have now been extracted into `lib/runtime/task_facade.py`, including task start/stop helpers, settings object creation, task completion handling, and the convenience `run_*` entrypoints.
- Runtime pipeline progress/error/image-done callbacks have now been extracted into `lib/runtime/pipeline_callbacks.py`, and the headless memory controls were extended with `isVisible()` so those callback paths can be smoke-tested without Qt widgets.
- `PipelineHandlerMixin` is now down to 445 lines after delegating task façade and pipeline callback logic too, and both `task_facade` and `pipeline_callbacks` are now part of the backend reload catalog.
- Runtime API helpers have now been extracted into `lib/runtime/runtime_api.py`, covering runtime event history, runtime state snapshots, ui-spec mutation APIs, structured settings update APIs, shutdown scheduling, and command execution helpers.
- After aliasing host-support/control-plane/state-adapter/runtime-api helpers directly, `PipelineHandlerMixin` is now down to 184 lines and is effectively a thin runtime adapter instead of a central implementation hub.
- Worker/model reload policy is now formalized in `lib/runtime/reload_policy.py` and exposed through `backend.get_reload_policy`, with explicit recommendations for:
  - extracted runtime services -> `backend.reload_services`
  - worker implementation modules in service mode -> `workers.services_reload`
  - pipeline task modules -> `restart_host`
- A repo-level regression harness now exists at `scripts/runtime_regression.py`, covering runtime surface, selection/filter flow, ui-spec mutation/reset, settings updates, pipeline callbacks, and reload-policy recommendations.
- The regression suite is also callable from the runtime surface as `test.run_runtime_regression`, and it now includes command-state tracking plus a lightweight task runner / event-flow check in addition to the earlier smoke checks.
- Worker invocation now carries structured worker error taxonomy as well, via `lib/workers/errors.py`, `WorkerOutput.error_info`, and task-level `TaskResult.error_info`.
- The runtime regression suite now verifies missing-worker behavior in both `inprocess` and `service` runtime modes, and both paths currently resolve to the stable code `worker_not_found`.
- The regression suite now also validates real worker-service lifecycle behavior with the lightweight `text_filter_lists` worker, covering `invoke -> workers.services_reload -> workers.services_stop -> invoke` and confirming a fresh process is spawned after stop/restart.
