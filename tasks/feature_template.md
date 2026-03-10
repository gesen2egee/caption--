# Feature Template

## Purpose
- Add new features through the runtime-first architecture instead of patching scattered UI callbacks.
- Keep feature work aligned with:
  - headless backend execution
  - command-driven invocation
  - runtime state synchronization
  - web/spec UI compatibility
  - Agent-safe execution surfaces

## Use This Template When
- adding a new single-image action
- adding a new batch action
- adding a new backend utility command
- exposing a new Agent-facing capability
- adding a new settings-backed workflow

## Design Rules
- Put business logic in a pure backend service first.
- Register one explicit command name for each externally callable feature.
- Return structured results instead of relying on `None` or free-form exceptions.
- Update runtime state if the feature affects UI-visible data.
- Add UI spec wiring only after the backend command works headlessly.
- If the feature should hot-reload, add it to runtime reloadables.

## Recommended File Placement

### 1. Pure backend logic
- Put feature logic in one of:
  - `lib/runtime/selection_service.py`
  - `lib/runtime/editor_service.py`
  - `lib/runtime/batch_service.py`
  - `lib/runtime/processing_service.py`
  - `lib/runtime/task_service.py`
- If none fit cleanly, add a focused new service such as:
  - `lib/runtime/export_service.py`
  - `lib/runtime/metadata_service.py`

### 2. Command registration
- Register the command in:
  - `lib/ui/mixins/pipeline_handler_mixin.py`
- Use stable names such as:
  - `action.run_<feature>_current`
  - `batch.run_<feature>`
  - `selection.<feature>`
  - `content.<feature>`
  - `backend.<feature>`

### 3. Runtime state
- If the feature changes app-visible state, update one or more sections:
  - `selection`
  - `content`
  - `controls`
  - `task`
  - `ui`
  - `tags`
- If a new state shape is needed, add projection logic in:
  - `lib/runtime/state_projection.py`

### 4. UI wiring
- Add or patch the runtime UI spec in:
  - `lib/runtime/ui_spec.py`
- If the feature needs special rendering behavior, extend:
  - `frontend/src/specRenderer.tsx`

### 5. Reload support
- If the feature lives in a runtime service and should support backend hot reload, register it in:
  - `lib/runtime/service_reload.py`

## Command Contract

### Minimum command result
```python
{
    "completed": False,
    "reason": "no_current_image",
    "message": "No image selected.",
}
```

### Result guidelines
- Always return a dictionary for new commands.
- Prefer machine-readable `reason` values.
- Keep `message` concise and user-facing.
- Include feature-specific fields only when they add real value.
- If the command can fail structurally, prefer stable error codes over ad-hoc exception strings.

## Error Handling
- Runtime and bridge surfaces should expose:
  - `error`
  - `error_info.code`
  - `error_info.source`
  - `error_info.message`
- Prefer codes such as:
  - `invalid_input`
  - `confirmation_required`
  - `worker_unavailable`
  - `file_not_found`
  - `command_access_denied`
- New frontend or Agent-facing features should be able to branch on error code without parsing free-form text.

### Good `reason` examples
- `no_current_image`
- `task_running`
- `invalid_input`
- `ocr_disabled`
- `no_backup`
- `confirmation_required`
- `worker_unavailable`

## Feature Checklist

### Backend
- [ ] Logic exists in a runtime service.
- [ ] Logic can run without widget instances.
- [ ] Logic returns structured data.

### Command surface
- [ ] Command is registered in `pipeline_handler_mixin.py`.
- [ ] Access mode is set correctly:
  - `safe`
  - `development`
  - `internal`
- [ ] Guardrail metadata is set:
  - `destructive`
  - `requires_confirmation`
  - `writes_files`
  - `long_running`
  - `risk_level`
- [ ] Command example is added if Agent/web shell should use it.

### Runtime state
- [ ] State updates go through runtime state sync.
- [ ] New state keys are reflected in `state_projection.py` if needed.

### UI
- [ ] `ui_spec.py` is updated if the feature needs visible controls.
- [ ] `specRenderer.tsx` is updated only if existing nodes are insufficient.
- [ ] The feature still works headlessly without UI wiring.

### Hot reload
- [ ] Service is added to `service_reload.py` if reloadable.
- [ ] Feature behavior is safe under deferred reload while a task is active.

### Validation
- [ ] `python -m compileall ...` passes for changed Python files.
- [ ] `npm run build` passes if frontend files changed.
- [ ] At least one headless smoke test runs through the command surface.

## Suggested Workflow
1. Implement the pure backend service logic.
2. Verify it with a direct Python smoke test.
3. Register the runtime command.
4. Add structured result payloads and guardrail metadata.
5. Sync runtime state.
6. Expose the feature in `ui_spec.py` only if needed.
7. Add reload support if the feature belongs to hot-reloadable runtime services.
8. Verify via:
   - headless command call
   - web shell or spec UI
   - event/state inspection

## Example Skeleton

### Backend service
```python
def plan_current_export_action(current_image_path: str | None, output_dir: str | None) -> dict:
    if not current_image_path:
        return {
            "completed": False,
            "reason": "no_current_image",
            "message": "No image selected.",
        }
    if not output_dir:
        return {
            "completed": False,
            "reason": "invalid_input",
            "message": "No output directory configured.",
        }
    return {
        "completed": True,
        "reason": "ready",
        "message": "Ready to export.",
        "image_path": current_image_path,
        "output_dir": output_dir,
    }
```

### Command registration
```python
self._register_command(
    "action.run_export_current",
    self.command_run_export_current,
    access_modes=("development", "internal"),
    metadata={
        "category": "processing",
        "writes_files": True,
        "destructive": False,
        "requires_confirmation": False,
        "risk_level": "medium",
    },
    example={
        "name": "action.run_export_current",
        "kwargs": {},
    },
)
```

## Anti-Patterns
- Do not put core feature logic directly in widget callbacks.
- Do not make new features depend on `QTextEdit`, `QCheckBox`, or dialog instances.
- Do not return raw strings when a structured result is possible.
- Do not add frontend-only behavior before the headless command works.
- Do not add a new task path if an existing runtime service can own the logic.

## Notes
- Existing legacy Qt paths may still remain for compatibility, but new features should treat them as adapters.
- Prefer extending runtime services over growing `PipelineHandlerMixin`.
- If a new feature exposes Agent-facing behavior, update manifest/examples/guardrails in the same change.
