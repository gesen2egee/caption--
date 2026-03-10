# Caption Runtime Shell

This folder now contains a minimal React/Vite shell that talks to the Python runtime bridge.

## What it does

- consumes `GET /state`, `GET /events`, `GET /ui-spec`
- consumes `GET /ui-spec-override`, `GET /workers`, `GET /settings-schema`
- executes runtime commands through `POST /commands/{name}`
- shows current preview via `GET /preview/current`
- keeps the legacy split-panel structure as a migration shell
- supports ephemeral UI spec override and node-by-id patching
- renders a schema-driven settings editor from runtime metadata
- includes a spec-renderer preview that consumes `ui-spec` directly
- uses runtime `controls` state plus `ui.set_control_value` for checkbox/select bindings
- uses the spec renderer as the primary runtime workspace, with hand-written panels kept for inspector/editor duties
- includes a `Runtime Tools` panel for direct command-style custom-tag, find/replace, and delete-current flows
- includes a `Runtime Tools` path for stroke eraser too, by uploading a mask image and sending it to the runtime command surface
- exposes `command_examples` in runtime capabilities so the shell can show callable payload shapes
- routes legacy batch buttons through dedicated `batch.run_*` commands so save-to-txt and confirm flows survive the migration
- routes single-image buttons through dedicated `action.run_*_current` commands so preflight checks survive the migration too
- shows a `Command Trace` panel driven from runtime state so agent/web-driven commands are easier to inspect
- prefers `GET /events/stream` SSE for low-latency refresh, with periodic polling kept as fallback
- applies section-level `state.updated` deltas from SSE, so most UI changes no longer force a full `/state` refetch
- caps in-process event history and fetches `/events` with a bounded `limit`, so long-running sessions do not keep growing the payload forever
- supports `include_prefix` / `exclude_prefix` event filtering on bridge history and SSE endpoints, so agent/debug tooling can subscribe to a cleaner event slice
- exposes named event profiles too: `control`, `diagnostic`, and `all`
- exposes an agent manifest at `/agent/manifest` with recommended event profile, endpoint map, and command examples
- the agent manifest also includes stable relative `routes`, so Agent clients can bootstrap even before the localhost bridge URL is known
- the agent manifest is mode-aware: `mode=safe` excludes destructive file ops, settings mutation, and UI-spec mutation, while `mode=development` exposes the full refactor surface
- runtime command execution now enforces those modes too: `POST /commands/{name}?mode=safe` is gated by command metadata, not just documented by the manifest
- runtime capabilities and agent manifests now expose `command_metadata`, so shell/Agent clients can see guardrails like `writes_files`, `destructive`, `requires_confirmation`, and `risk_level` before execution
- the shell now executes commands in the currently selected `safe` or `development` mode, and disables blocked controls locally before the bridge has to reject them
- bridge and command failures now expose structured `error_info` with stable `code`, `source`, and `message`, so shell/Agent clients can branch on errors without parsing free-form exception text
- runtime state is now backed by a central in-process app state before being projected to the evented state store, which reduces direct dependence on widget-only state for shell/Agent reads

## Development

1. Start the desktop app with bridge enabled:
   - `run.bat shell`
   - use [`run_runtime_shell.bat`](/E:/caption--/run_runtime_shell.bat)
   - or use [`run_runtime_web.bat`](/E:/caption--/run_runtime_web.bat) after a production build
   - or set `CAPTION_RUNTIME_HTTP=1` before `python caption.py`
2. Open the Vite shell:
   - default URL: `http://127.0.0.1:5173`
3. Build:
   - `npm run build`

## Notes

- Vite proxies `/api` to `http://127.0.0.1:8765` by default.
- After `npm run build`, the Python bridge can serve the built shell directly at `http://127.0.0.1:8765/`.
- `run.bat web` now starts that built shell through a pure Python headless host, so the runtime web UI is no longer piggybacking on a hidden `QApplication/MainWindow`.
- `run.bat service` starts the same headless backend/bridge without opening a browser, which is useful for Agent or external shell clients.
- importing the headless host path no longer pulls `PyQt6` into `sys.modules`; Qt is now lazy on the runtime path and only shows up for legacy desktop-only surfaces.
- the shell now includes a `Shutdown App` action backed by `app.shutdown`, so headless web mode has a formal exit path.
- This is a migration shell, not the final DSL renderer.
- The existing PyQt UI remains the source of truth while the runtime boundary is being extracted.
- `editor.find_replace` and `image.delete_current` are now safe to call as structured commands; the old PyQt dialogs remain only as compatibility entry points.
- batch commands now carry legacy options such as `save_to_txt`, `delete_chars`, and `confirm`, and the web renderer adds browser confirms where the old app used modal dialogs.
- current-image commands now return structured `reason` values like `confirm_missing_tags_required` and `no_backup`; the renderer handles the actionable ones with browser-side confirmation.
- command events now carry summarized inputs/results, and the runtime state mirrors the latest started/finished/failed command for easier debugging.
- the runtime bridge now serves an SSE stream at `/events/stream`; the shell subscribes to it and only falls back to slower polling when live streaming is unavailable.
- runtime state events now send section/key deltas instead of the full state blob; the shell patches local state directly and only does full refresh on events it does not understand.
- event history is now capped in-process, and `GET /events?limit=...` returns only the most recent slice needed by the shell.
- section state updates no longer emit a duplicate generic `state.updated`; consumers should use `content.updated`, `selection.updated`, `commands.updated`, and other concrete section events.
- `GET /events` and `GET /events/stream` now accept `include_prefix` and `exclude_prefix` query parameters for cleaner agent/debug subscriptions.
- `GET /events` and `GET /events/stream` also accept `profile=control|diagnostic|all`; the shell uses `control` for event history while keeping the live SSE stream unfiltered for local state patching.
- `/agent/manifest` is the preferred bootstrap surface for external Agent clients; it returns the recommended `control` event profile, bridge endpoints, and example command payloads.
- when the bridge is not yet enabled, `/agent/manifest` still returns relative `routes`; once the bridge has a base URL, `endpoints` are filled with absolute URLs.
- use `/agent/manifest?mode=safe` for constrained operators and `/agent/manifest?mode=development` for refactor/debug agents.
- command routes inside the manifest already include the matching `?mode=...` query, so safe agents do not need to invent their own gating convention.
- `action.run_stroke_eraser_current` can now be called without opening the PyQt stroke dialog, using either `mask_path` or uploaded PNG/base64 mask data.
- `command_metadata` is the preferred preflight surface for UI/Agent guardrails; for example, `image.delete_current` is marked `destructive + requires_confirmation`, while `settings.update` is marked `mutates_settings + writes_files`.
- task execution now uses plain Python threads for pipeline tasks, while callback delivery is marshalled back through a Qt callback dispatcher. `BaseTask` also exposes a plain listener API plus `run_inline()` / `execute_pipeline()`, and runtime capabilities report that via `task_runtime_backend`, `task_runtime_dispatcher`, and `task_runtime_listener_api`.
- the old compat workers, worker-scan background thread, and tag-flow UI signaling all use plain Python/threaded proxies now; in headless mode runtime capabilities report `runtime_host_backend=headless-python` and `qt_residual_components=[]`.
- current-image and batch commands now prefer runtime state helpers for prompt text, save-to-txt toggles, current image path, and loaded image paths, so the shell/Agent path is less tightly coupled to live widget instances.
- selection state now also includes `all_image_paths` and `filtered_image_paths`, and the legacy navigation/filter/editor flows are more tolerant of missing Qt widgets when driven through runtime commands.
- worker execution can now run in independent Python service processes via `worker_runtime_mode=service`, and the shell/runtime can reload or stop those services at runtime.
- extracted pure backend services can also be hot-reloaded in place via:
  - `backend.list_reloadables`
  - `backend.reload_services`
- automatic backend-service watch mode is also available via:
  - `backend.watch_reload_start`
  - `backend.watch_reload_status`
  - `backend.watch_reload_stop`
- the shell now exposes both worker-service reload and backend-service reload, so code changes in `selection/editor/batch/processing` services can be applied without restarting the whole host.
- backend reloadable modules now also include `task` orchestration and `state_projection`, so task wiring and runtime state-shape changes can be hot-reloaded too.
- `run.bat web`, `run.bat shell`, and `run.bat service` now opt into backend-service auto reload by default.
