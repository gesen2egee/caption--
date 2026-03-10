# Feature Dependency Map

## Why New Features Feel Hard

The current codebase is organized around technical layers and window behaviors, not around feature ownership.

That means one user-facing feature usually spans all of these at once:
- widget creation in `lib/ui/main_window.py`
- single-image action handling in `lib/ui/mixins/processing_mixin.py`
- batch action handling in `lib/ui/mixins/batch_mixin.py`
- menu wiring in `lib/ui/mixins/settings_mixin.py`
- task dispatch and callback wiring in `lib/ui/mixins/pipeline_handler_mixin.py`
- task execution in `lib/pipeline/tasks/*.py`
- worker implementation in `lib/workers/*.py`
- settings schema/defaults in `lib/core/dataclasses.py` and `lib/core/settings.py`
- optional sidecar/file post-processing in `lib/utils/*.py`

This is the root cause.

## Structural Diagnosis

### Problem 1: No single feature owner
Each feature is spread across:
- UI entry
- UI state logic
- task dispatch
- task implementation
- worker implementation
- settings
- result synchronization

No single module answers: "this is the Tagger feature" or "this is the Mask Text feature".

### Problem 2: `MainWindow` is both view and orchestrator
`lib/ui/main_window.py` is not just layout.
It also acts as:
- state container
- feature switchboard
- widget registry
- runtime coordinator

This makes every feature change gravitate back into UI code.

### Problem 3: Settings are duplicated across layers
A new configurable feature usually touches:
- `DEFAULT_APP_SETTINGS`
- `Settings` dataclass
- settings dialog controls
- task config mapping
- worker config mapping

That is too many synchronization points for one concern.

### Problem 4: Features are entered through callbacks, not commands
The real entrypoint for a feature is often:
- a button click
- a menu action
- a mixin method

There is no stable app-level command like:
- `task.run_tagger`
- `task.run_mask_text`
- `settings.update`

Without commands:
- Agent integration is awkward
- testing is awkward
- hot-reload is limited
- feature ownership stays blurry

## Current Dependency Chains

### 1. Tagger

#### Current path
- UI button is wired in `lib/ui/main_window.py`
- single-image action lives in `lib/ui/mixins/processing_mixin.py`
- batch action lives in `lib/ui/mixins/batch_mixin.py`
- task dispatch is handled by `lib/ui/mixins/pipeline_handler_mixin.py`
- task implementation lives in `lib/pipeline/tasks/tagger_task.py`
- worker resolution uses `lib/workers/registry.py`
- worker implementation lives in `lib/workers/tagger_imgutils_generic.py` or `lib/workers/tagger_imgutils_tagging_local.py`
- result persistence uses `lib/utils/sidecar.py`
- image creation helpers come from `lib/utils/file_ops.py`
- settings are defined in `lib/core/dataclasses.py` and `lib/core/settings.py`
- settings UI is in `lib/ui/dialogs/settings_dialog.py`

#### Why changes spread
- changing model options touches settings, dialog, task config, and possibly worker behavior
- changing result handling touches task, sidecar handling, and UI refresh logic
- adding a new Tagger mode touches both single and batch flows separately

#### Future ownership target
- `features/tagger/commands.py`
- `features/tagger/service.py`
- `features/tagger/settings.py`
- `features/tagger/ui_spec.json`

### 2. LLM Caption / NL Generation

#### Current path
- UI button and prompt editor are in `lib/ui/main_window.py`
- single-image action lives in `lib/ui/mixins/processing_mixin.py`
- batch action lives in `lib/ui/mixins/batch_mixin.py`
- task dispatch lives in `lib/ui/mixins/pipeline_handler_mixin.py`
- task implementation lives in `lib/pipeline/tasks/llm_task.py`
- tag context uses `lib/utils/tag_context.py`
- output post-processing uses `lib/utils/parsing.py`
- sidecar persistence uses `lib/utils/sidecar.py`
- workers live in `lib/workers/vlm_openrouter_api.py` and `lib/workers/llm_llama_cpp_local.py`
- settings live in `lib/core/dataclasses.py`, `lib/core/settings.py`, and `lib/ui/dialogs/settings_dialog.py`

#### Why changes spread
- prompt source, prompt template, and provider selection are split between UI text fields, settings, task extra payload, and worker config
- switching provider behavior affects task config mapping and settings UI
- NL output handling is mixed between task persistence and UI tab refresh behavior

#### Future ownership target
- `features/llm/commands.py`
- `features/llm/service.py`
- `features/llm/prompting.py`
- `features/llm/settings.py`
- `features/llm/ui_spec.json`

### 3. Image Process

#### Current path
- image-process tab and controls are in `lib/ui/main_window.py`
- single-image action lives in `lib/ui/mixins/processing_mixin.py`
- batch action lives in `lib/ui/mixins/batch_mixin.py`
- task dispatch lives in `lib/ui/mixins/pipeline_handler_mixin.py`
- task implementation lives in `lib/pipeline/tasks/image_process_task.py`
- backup logic uses `lib/utils/file_ops.py`
- sidecar persistence uses `lib/utils/sidecar.py`
- worker implementation is `lib/workers/image_flux2_klein_gguf_local.py`
- settings live in `lib/core/dataclasses.py`, `lib/core/settings.py`, and `lib/ui/dialogs/settings_dialog.py`

#### Why changes spread
- prompt defaults, server settings, model assets, and task parameters are all configured in separate places
- UI text fields and task `extra` payload must stay synchronized
- result path replacement logic is handled elsewhere in navigation/UI code

#### Future ownership target
- `features/image_process/commands.py`
- `features/image_process/service.py`
- `features/image_process/settings.py`
- `features/image_process/ui_spec.json`

### 4. Unmask / Background Removal

#### Current path
- menu action is wired in `lib/ui/mixins/settings_mixin.py`
- single-image action lives in `lib/ui/mixins/processing_mixin.py`
- batch action lives in `lib/ui/mixins/batch_mixin.py`
- task dispatch lives in `lib/ui/mixins/pipeline_handler_mixin.py`
- task implementation lives in `lib/pipeline/tasks/unmask_task.py`
- worker implementation lives in `lib/workers/mask_transparent_background_local.py`
- file replacement and refresh behavior touch navigation logic
- settings live in `lib/core/dataclasses.py`, `lib/core/settings.py`, and `lib/ui/dialogs/settings_dialog.py`

#### Why changes spread
- there are separate entrypoints for menu, batch, and task execution
- filtering logic for eligible images lives outside the task
- file replacement and display refresh are not owned by the feature module itself

#### Future ownership target
- `features/unmask/commands.py`
- `features/unmask/service.py`
- `features/unmask/settings.py`
- `features/unmask/ui_spec.json`

### 5. Mask Text

#### Current path
- menu action is wired in `lib/ui/mixins/settings_mixin.py`
- single-image action lives in `lib/ui/mixins/processing_mixin.py`
- batch action lives in `lib/ui/mixins/batch_mixin.py`
- task dispatch lives in `lib/ui/mixins/pipeline_handler_mixin.py`
- task implementation lives in `lib/pipeline/tasks/mask_text_task.py`
- OCR and mask worker usage go through `lib/workers/mask_text_local.py` and `lib/workers/detect_imgutils_ocr_local.py`
- advanced post-processing uses `lib/utils/image_processing.py`
- backup logic uses `lib/utils/file_ops.py`
- settings live in `lib/core/dataclasses.py`, `lib/core/settings.py`, and `lib/ui/dialogs/settings_dialog.py`

#### Why changes spread
- the feature mixes OCR settings, file output settings, alpha post-processing, and batch skip policy
- part of the logic is in task execution, part is in batch pre-filtering, and part is in UI state updates
- this is a feature with especially high setting surface area, so it suffers most from settings duplication

#### Future ownership target
- `features/mask_text/commands.py`
- `features/mask_text/service.py`
- `features/mask_text/settings.py`
- `features/mask_text/ui_spec.json`

### 6. Restore

#### Current path
- menu action is wired in `lib/ui/mixins/settings_mixin.py`
- single-image action lives in `lib/ui/mixins/processing_mixin.py`
- batch action lives in `lib/ui/mixins/batch_mixin.py`
- task dispatch lives in `lib/ui/mixins/pipeline_handler_mixin.py`
- task implementation lives in `lib/pipeline/tasks/restore_task.py`
- backup existence checks and file restore behavior use `lib/utils/file_ops.py`
- list refresh and current-image replacement affect navigation/UI behavior

#### Why changes spread
- restore is simple in domain terms but still pays the full UI/task/batch/callback cost
- even a small behavior tweak must be reflected in menu wiring, task callbacks, and image list synchronization

#### Future ownership target
- `features/restore/commands.py`
- `features/restore/service.py`
- `features/restore/ui_spec.json`

## Cross-Cutting Hotspots

### Settings hotspot
The same concern appears in too many places:
- `lib/core/settings.py`
- `lib/core/dataclasses.py`
- `lib/ui/dialogs/settings_dialog.py`
- task config assembly in `lib/pipeline/tasks/*.py`

Recommended fix:
- introduce a feature-local settings schema
- generate UI/settings binding from schema where possible

### Dispatch hotspot
Task dispatch is concentrated in:
- `lib/ui/mixins/pipeline_handler_mixin.py`

Recommended fix:
- replace UI-owned dispatch with app-owned commands:
  - `task.run_tagger`
  - `task.run_llm`
  - `task.run_image_process`
  - `task.run_unmask`
  - `task.run_mask_text`
  - `task.run_restore`

### Result synchronization hotspot
Result handling is split between:
- task execution
- sidecar persistence
- UI refresh logic
- file list mutation

Recommended fix:
- each feature returns a structured result event
- a central state reducer updates app state
- UI only reacts to state changes

## Core Insight

The codebase is currently:
- UI-driven
- callback-driven
- layer-driven

But it needs to become:
- command-driven
- event-driven
- feature-owned

That is the real diagnosis.

## Refactor Principle

For any new feature, the desired future shape is:
- one command entrypoint
- one feature service
- one settings schema
- one result event contract
- one UI spec section

If a future feature still requires edits across `main_window.py`, multiple mixins, task code, settings code, and worker glue just to become usable, the architecture has not been fixed yet.
