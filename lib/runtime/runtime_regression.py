# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lib.core.dataclasses import ImageData
from lib.runtime.http_bridge import RuntimeHttpBridge
from lib.pipeline.context import TaskResult
from lib.pipeline.tasks.base_task import BaseTask
from lib.workers import WorkerInput, invoke_worker


Check = Tuple[str, Callable[[Any, Path], Dict[str, Any]]]


class _DemoRuntimeTask(BaseTask):
    @property
    def name(self) -> str:
        return "demo_runtime_regression"

    def execute(self, context) -> TaskResult:
        return TaskResult(success=True, result_text="ok", image=context.image)


def _prepare_fixture() -> Path:
    root = Path(tempfile.mkdtemp(prefix="caption_runtime_regression_"))
    (root / "a.png").write_bytes(b"png")
    (root / "a.txt").write_text("needle text", encoding="utf-8")
    (root / "b.png").write_bytes(b"png")
    (root / "b.txt").write_text("other", encoding="utf-8")
    return root


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02) -> None:
    deadline = time.time() + max(timeout, interval)
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("runtime regression timed out while waiting for task completion")


def check_runtime_surface(host: Any, root: Path) -> Dict[str, Any]:
    smoke = host.execute_command("test.smoke_command", access_mode="internal")
    assert "app.get_runtime_state" in smoke["commands"], smoke
    state = host.get_runtime_state()
    assert "settings" in state and "selection" in state, state
    return {
        "commands": len(smoke["commands"]),
        "state_sections": sorted(state.keys()),
    }


def check_selection_commands(host: Any, root: Path) -> Dict[str, Any]:
    host.command_set_root_dir(str(root))
    filtered = host.command_apply_filter(query="needle", use_tags=False, use_text=True)
    assert filtered["match_count"] == 1, filtered
    cleared = host.command_clear_filter()
    assert cleared["match_count"] == 2, cleared
    return {
        "filtered_count": filtered["match_count"],
        "cleared_count": cleared["match_count"],
    }


def check_ui_spec(host: Any, root: Path) -> Dict[str, Any]:
    original_title = host.get_ui_spec().get("title")
    updated = host.update_ui_spec({"title": "runtime-regression"})
    assert updated.get("title") == "runtime-regression", updated
    reset = host.reset_ui_spec()
    assert reset.get("title") == original_title, reset
    return {
        "title_restored": reset.get("title"),
    }


def check_settings(host: Any, root: Path) -> Dict[str, Any]:
    updated = host.update_runtime_settings({"ui_language": "zh_tw"})
    assert updated["ui_language"] == "zh_tw", updated
    return {
        "ui_language": updated["ui_language"],
    }


def check_callbacks(host: Any, root: Path) -> Dict[str, Any]:
    host.on_pipeline_progress(1, 3, str(root / "a.png"), speed=0.5)
    assert "a.png" in host.statusBar().last_message, host.statusBar().last_message
    assert host.progress_bar.isVisible() is True
    host.on_pipeline_error("boom")
    assert host.progress_bar.isVisible() is False
    return {
        "status_message": host.statusBar().last_message,
    }


def check_reload_policy(host: Any, root: Path) -> Dict[str, Any]:
    runtime_policy = host.command_get_reload_policy("lib/runtime/command_actions.py")
    worker_policy = host.command_get_reload_policy("lib/workers/image_restore_raw.py")
    pipeline_policy = host.command_get_reload_policy("lib/pipeline/tasks/base_task.py")
    assert runtime_policy["recommendation"]["recommended_action"] == "backend.reload_services", runtime_policy
    assert worker_policy["recommendation"]["recommended_action"] in {"workers.services_reload", "restart_host"}, worker_policy
    assert pipeline_policy["recommendation"]["recommended_action"] == "restart_host", pipeline_policy
    return {
        "runtime_action": runtime_policy["recommendation"]["recommended_action"],
        "worker_action": worker_policy["recommendation"]["recommended_action"],
        "pipeline_action": pipeline_policy["recommendation"]["recommended_action"],
    }


def check_command_tracking(host: Any, root: Path) -> Dict[str, Any]:
    payload = host.execute_command("settings.get", access_mode="internal")
    assert isinstance(payload, dict), payload
    commands_state = host.get_runtime_state().get("commands", {})
    last_finished = dict(commands_state.get("last_finished") or {})
    assert last_finished.get("command_name") == "settings.get", last_finished
    return {
        "last_finished": last_finished.get("command_name"),
    }


def check_task_flow(host: Any, root: Path) -> Dict[str, Any]:
    host._current_task = None
    image = ImageData(path=str(root / "a.png"))
    before_events = len(host._get_runtime_event_bus().history())
    host.run_task(_DemoRuntimeTask, [image])
    _wait_until(lambda: not host.is_task_running())
    events = host._get_runtime_event_bus().history()
    task_events = [event.name for event in events[before_events:] if str(event.name).startswith("task.")]
    assert "task.started" in task_events, task_events
    assert "task.progress" in task_events, task_events
    assert "task.image_done" in task_events, task_events
    assert "task.batch_done" in task_events, task_events
    task_state = host.get_runtime_state().get("task", {})
    assert task_state.get("running") is False, task_state
    return {
        "task_events": task_events,
    }


def check_worker_errors(host: Any, root: Path) -> Dict[str, Any]:
    settings = deepcopy(host._get_current_settings_obj())
    worker_input = WorkerInput(settings=settings)

    missing_inprocess = invoke_worker(
        "LLM",
        "__missing_worker__",
        config={},
        worker_input=worker_input,
        settings=settings,
    )
    assert missing_inprocess.success is False, missing_inprocess
    assert (missing_inprocess.error_info or {}).get("code") == "worker_not_found", missing_inprocess.error_info

    settings.worker_runtime_mode = "service"
    missing_service = invoke_worker(
        "LLM",
        "__missing_worker__",
        config={},
        worker_input=WorkerInput(settings=settings),
        settings=settings,
    )
    assert missing_service.success is False, missing_service
    assert (missing_service.error_info or {}).get("code") == "worker_not_found", missing_service.error_info

    return {
        "inprocess_code": (missing_inprocess.error_info or {}).get("code"),
        "service_code": (missing_service.error_info or {}).get("code"),
    }


def check_worker_service_lifecycle(host: Any, root: Path) -> Dict[str, Any]:
    settings = deepcopy(host._get_current_settings_obj())
    settings.worker_runtime_mode = "service"
    config = {"blacklist": ["needle"], "whitelist": []}
    worker_kwargs = {
        "config": config,
        "worker_input": WorkerInput(settings=settings, extra={"text": "needle text"}),
        "settings": settings,
    }

    category = "OTHER"
    worker_name = "text_filter_lists"
    host.command_stop_worker_services(category=category, worker_name=worker_name)
    try:
        first = invoke_worker(category, worker_name, **worker_kwargs)
        assert first.success is True, first
        assert (first.result_data or {}).get("is_filtered") is True, first.result_data

        status_before = host.get_worker_services_status()
        matching_before = [
            item for item in list(status_before.get("services", []) or [])
            if item.get("category") == category and item.get("worker_name") == worker_name
        ]
        assert matching_before, status_before
        first_pid = matching_before[0].get("pid")

        reloaded = host.command_reload_worker_services(category=category, worker_name=worker_name)
        assert int(reloaded.get("reloaded_count", 0)) >= 1, reloaded

        second = invoke_worker(category, worker_name, **worker_kwargs)
        assert second.success is True, second
        assert (second.result_data or {}).get("is_filtered") is True, second.result_data

        stopped = host.command_stop_worker_services(category=category, worker_name=worker_name)
        assert int(stopped.get("stopped_count", 0)) >= 1, stopped

        third = invoke_worker(category, worker_name, **worker_kwargs)
        assert third.success is True, third
        assert (third.result_data or {}).get("is_filtered") is True, third.result_data

        status_after_restart = host.get_worker_services_status()
        matching_after = [
            item for item in list(status_after_restart.get("services", []) or [])
            if item.get("category") == category and item.get("worker_name") == worker_name
        ]
        assert matching_after, status_after_restart
        restarted_pid = matching_after[0].get("pid")
    finally:
        host.command_stop_worker_services(category=category, worker_name=worker_name)

    return {
        "first_pid": first_pid,
        "restarted_pid": restarted_pid,
        "reload_count": reloaded.get("reloaded_count"),
        "stop_count": stopped.get("stopped_count"),
        "restart_spawned_new_process": bool(first_pid and restarted_pid and first_pid != restarted_pid),
    }


def check_http_bridge_disconnect_tolerance(host: Any, root: Path) -> Dict[str, Any]:
    bridge = RuntimeHttpBridge(host, port=0)
    bridge.start()
    try:
        handler_cls = bridge._server.RequestHandlerClass  # type: ignore[union-attr]

        class _DummyWfile:
            def write(self, data):
                raise ConnectionAbortedError(10053, "aborted")

        handler = handler_cls.__new__(handler_cls)
        handler.send_response = lambda status: None
        handler.send_header = lambda *args, **kwargs: None
        handler.end_headers = lambda: (_ for _ in ()).throw(ConnectionAbortedError(10053, "aborted during headers"))
        handler.wfile = _DummyWfile()
        handler.close_connection = False

        handler._send_bytes(200, b"test", "text/plain")
        assert handler.close_connection is True
        return {"disconnect_swallowed": True}
    finally:
        bridge.stop()


CHECKS: List[Check] = [
    ("runtime_surface", check_runtime_surface),
    ("selection_commands", check_selection_commands),
    ("ui_spec", check_ui_spec),
    ("settings", check_settings),
    ("callbacks", check_callbacks),
    ("reload_policy", check_reload_policy),
    ("command_tracking", check_command_tracking),
    ("task_flow", check_task_flow),
    ("worker_errors", check_worker_errors),
    ("worker_service_lifecycle", check_worker_service_lifecycle),
    ("http_bridge_disconnect_tolerance", check_http_bridge_disconnect_tolerance),
]


def run_runtime_regression(host: Any) -> Dict[str, Any]:
    root = _prepare_fixture()
    results: Dict[str, Any] = {"fixture_root": str(root), "checks": {}}
    for name, func in CHECKS:
        results["checks"][name] = func(host, root)
    return results


def command_run_runtime_regression(host: Any) -> Dict[str, Any]:
    return run_runtime_regression(host)
