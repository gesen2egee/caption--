# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import mimetypes
import os
import queue
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from lib.runtime.callback_dispatcher import RuntimeCallbackDispatcher
from lib.runtime.commands import CommandAccessError
from lib.runtime.errors import build_runtime_error_info
from lib.runtime.events import EventBus, resolve_event_filters


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _frontend_dist_dir() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "frontend",
            "dist",
        )
    )


class RuntimeCommandInvoker:
    """
    Marshal command execution onto the UI thread and wait for a result.
    """

    def __init__(self, owner):
        self._owner = owner
        self._lock = threading.Lock()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._dispatcher = RuntimeCallbackDispatcher.create_default()

    def invoke(
        self,
        command_name: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        access_mode: str = "internal",
    ) -> Any:
        request_id = uuid.uuid4().hex
        done = threading.Event()
        with self._lock:
            self._pending[request_id] = {
                "done": done,
                "result": None,
                "error": None,
            }

        def _dispatch_execute() -> None:
            self._execute_command(request_id, command_name, args or [], kwargs or {}, access_mode)

        if self._dispatcher is not None:
            self._dispatcher.dispatch(_dispatch_execute)
        else:
            _dispatch_execute()
        done.wait()

        with self._lock:
            payload = self._pending.pop(request_id)

        if payload["error"] is not None:
            raise payload["error"]
        return payload["result"]

    def _execute_command(self, request_id, command_name, args, kwargs, access_mode):
        result = None
        error = None
        try:
            result = self._owner.execute_command(
                command_name,
                *(args or []),
                access_mode=str(access_mode or "internal"),
                **(kwargs or {}),
            )
        except Exception as exc:
            error = exc

        with self._lock:
            payload = self._pending.get(request_id)
            if payload is not None:
                payload["result"] = result
                payload["error"] = error
                payload["done"].set()


@dataclass
class RuntimeHttpBridgeStatus:
    enabled: bool
    host: Optional[str]
    port: Optional[int]
    url: Optional[str]


class RuntimeHttpBridge:
    """
    Optional localhost bridge exposing commands, state, events, and UI spec.
    """

    def __init__(self, owner, host: str = "127.0.0.1", port: int = 8765):
        self._owner = owner
        self._host = host
        self._port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._invoker = RuntimeCommandInvoker(owner)

    @staticmethod
    def _parse_prefix_filters(query: Dict[str, list[str]], key: str) -> list[str]:
        raw_values = query.get(key, [])
        prefixes: list[str] = []
        for raw in raw_values:
            for item in str(raw or "").split(","):
                value = item.strip()
                if value:
                    prefixes.append(value)
        return prefixes

    def start(self) -> RuntimeHttpBridgeStatus:
        if self._server is not None:
            return self.status()

        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def _send(self, status_code: int, payload: Dict[str, Any]) -> None:
                body = _json_bytes(payload)
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))

            def _send_error_payload(self, status_code: int, error_info: Dict[str, Any]) -> None:
                payload = {
                    "ok": False,
                    "error": error_info.get("message", "Runtime error"),
                    "error_info": error_info,
                }
                self._send(status_code, payload)

            def _send_bridge_error(
                self,
                status_code: int,
                *,
                code: str,
                message: str,
                source: str = "bridge.http",
                details: Optional[Dict[str, Any]] = None,
            ) -> None:
                self._send_error_payload(
                    status_code,
                    build_runtime_error_info(
                        None,
                        code=code,
                        message=message,
                        source=source,
                        status_code=status_code,
                        details=details,
                    ),
                )

            def _send_exception(self, status_code: int, exc: BaseException, *, source: str, command_name: Optional[str] = None, access_mode: Optional[str] = None) -> None:
                self._send_error_payload(
                    status_code,
                    build_runtime_error_info(
                        exc,
                        source=source,
                        command_name=command_name,
                        access_mode=access_mode,
                        status_code=status_code,
                        include_traceback=status_code >= 500,
                    ),
                )

            def _invoke_json_command(
                self,
                command_name: str,
                *,
                args: Optional[list[Any]] = None,
                kwargs: Optional[Dict[str, Any]] = None,
                access_mode: str = "internal",
            ) -> Any:
                try:
                    return bridge._invoker.invoke(
                        command_name,
                        args=args,
                        kwargs=kwargs,
                        access_mode=access_mode,
                    )
                except CommandAccessError as exc:
                    self._send_exception(
                        403,
                        exc,
                        source="bridge.command",
                        command_name=command_name,
                        access_mode=access_mode,
                    )
                    return None
                except Exception as exc:
                    self._send_exception(
                        500,
                        exc,
                        source="bridge.command",
                        command_name=command_name,
                        access_mode=access_mode,
                    )
                    return None

            def _serve_static(self, relative_path: str) -> bool:
                dist_dir = _frontend_dist_dir()
                if not os.path.isdir(dist_dir):
                    return False

                relative_path = relative_path.lstrip("/")
                if not relative_path:
                    relative_path = "index.html"
                relative_path = os.path.normpath(relative_path)
                target_path = os.path.normpath(os.path.join(dist_dir, relative_path))

                if not target_path.startswith(dist_dir):
                    self._send(403, {"error": "forbidden"})
                    return True

                if os.path.isdir(target_path):
                    target_path = os.path.join(target_path, "index.html")

                if not os.path.isfile(target_path):
                    fallback = os.path.join(dist_dir, "index.html")
                    if relative_path != "index.html" and os.path.isfile(fallback):
                        with open(fallback, "rb") as handle:
                            self._send_bytes(200, handle.read(), "text/html; charset=utf-8")
                        return True
                    return False

                content_type = mimetypes.guess_type(target_path)[0] or "application/octet-stream"
                if content_type.startswith("text/"):
                    content_type = f"{content_type}; charset=utf-8"
                with open(target_path, "rb") as handle:
                    self._send_bytes(200, handle.read(), content_type)
                return True

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query or "")
                command_mode = str(query.get("mode", ["development"])[0] or "development").strip() or "development"
                if parsed.path == "/health":
                    self._send(200, {"ok": True, "bridge": bridge.status().__dict__})
                    return
                if parsed.path == "/commands":
                    payload = self._invoke_json_command(
                        "test.smoke_command",
                        kwargs={"mode": command_mode},
                        access_mode=command_mode,
                    )
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/capabilities":
                    payload = self._invoke_json_command(
                        "app.get_capabilities",
                        kwargs={"mode": command_mode},
                        access_mode=command_mode,
                    )
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/agent/manifest":
                    mode = query.get("mode", [None])[0]
                    payload = self._invoke_json_command("app.get_agent_manifest", kwargs={"mode": mode})
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/state":
                    payload = self._invoke_json_command("app.get_runtime_state")
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/events":
                    limit = None
                    profile = query.get("profile", [None])[0]
                    limit_raw = query.get("limit", [None])[0]
                    include_prefixes = bridge._parse_prefix_filters(query, "include_prefix")
                    exclude_prefixes = bridge._parse_prefix_filters(query, "exclude_prefix")
                    if limit_raw is not None:
                        try:
                            limit = max(0, int(limit_raw))
                        except (TypeError, ValueError):
                            self._send_bridge_error(
                                400,
                                code="invalid_limit",
                                message="Invalid limit value.",
                                details={"limit": limit_raw},
                            )
                            return
                    try:
                        profile_name, resolved_include, resolved_exclude = resolve_event_filters(
                            profile=profile,
                            include_prefixes=include_prefixes,
                            exclude_prefixes=exclude_prefixes,
                        )
                    except ValueError as exc:
                        self._send_exception(400, exc, source="bridge.events")
                        return
                    events_payload = self._invoke_json_command(
                        "app.get_runtime_events",
                        kwargs={
                            "limit": limit,
                            "profile": profile_name,
                            "include_prefixes": include_prefixes,
                            "exclude_prefixes": exclude_prefixes,
                        },
                    )
                    if events_payload is None:
                        return
                    self._send(
                        200,
                        {
                            "events": events_payload,
                            "limit": limit,
                            "profile": profile_name,
                        },
                    )
                    return
                if parsed.path == "/events/stream":
                    profile = query.get("profile", [None])[0]
                    try:
                        profile_name, resolved_include, resolved_exclude = resolve_event_filters(
                            profile=profile,
                            include_prefixes=bridge._parse_prefix_filters(query, "include_prefix"),
                            exclude_prefixes=bridge._parse_prefix_filters(query, "exclude_prefix"),
                        )
                    except ValueError as exc:
                        self._send_exception(400, exc, source="bridge.events")
                        return
                    bridge._serve_event_stream(
                        self,
                        include_prefixes=resolved_include,
                        exclude_prefixes=resolved_exclude,
                        profile_name=profile_name,
                    )
                    return
                if parsed.path == "/workers":
                    payload = self._invoke_json_command("workers.list")
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/settings-schema":
                    payload = self._invoke_json_command("settings.schema")
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/ui-spec":
                    payload = self._invoke_json_command("app.get_ui_spec")
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/ui-spec-override":
                    payload = self._invoke_json_command("app.get_ui_spec_override")
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/bridge":
                    payload = self._invoke_json_command("bridge.get_status")
                    if payload is None:
                        return
                    self._send(200, payload)
                    return
                if parsed.path == "/preview/current":
                    image_path = str(getattr(bridge._owner, "current_image_path", "") or "")
                    if not image_path or not os.path.isfile(image_path):
                        self._send_bridge_error(
                            404,
                            code="no_current_image",
                            message="No current image selected.",
                            source="bridge.preview",
                        )
                        return
                    with open(image_path, "rb") as handle:
                        body = handle.read()
                    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
                    self._send_bytes(200, body, content_type)
                    return
                if parsed.path == "/" or parsed.path.startswith("/assets/"):
                    if self._serve_static(parsed.path):
                        return
                self._send_bridge_error(
                    404,
                    code="not_found",
                    message="Route not found.",
                    details={"path": parsed.path},
                )

            def do_POST(self):
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query or "")
                if not parsed.path.startswith("/commands/"):
                    self._send_bridge_error(
                        404,
                        code="not_found",
                        message="Route not found.",
                        details={"path": parsed.path},
                    )
                    return

                command_name = parsed.path.split("/commands/", 1)[1]
                command_mode = str(query.get("mode", ["development"])[0] or "development").strip() or "development"
                try:
                    payload = self._read_json()
                    args = payload.get("args", [])
                    kwargs = payload.get("kwargs", {})
                    result = bridge._invoker.invoke(
                        command_name,
                        args=args,
                        kwargs=kwargs,
                        access_mode=command_mode,
                    )
                    self._send(200, {"ok": True, "result": result})
                except CommandAccessError as exc:
                    self._send_exception(
                        403,
                        exc,
                        source="bridge.command",
                        command_name=command_name,
                        access_mode=command_mode,
                    )
                except Exception as exc:
                    self._send_exception(
                        500,
                        exc,
                        source="bridge.command",
                        command_name=command_name,
                        access_mode=command_mode,
                    )

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.status()

    def _serve_event_stream(
        self,
        handler: BaseHTTPRequestHandler,
        include_prefixes: Optional[list[str]] = None,
        exclude_prefixes: Optional[list[str]] = None,
        profile_name: str = "all",
    ) -> None:
        event_bus_getter = getattr(self._owner, "_get_runtime_event_bus", None)
        if not callable(event_bus_getter):
            handler.send_response(500)
            handler.end_headers()
            return

        event_bus = event_bus_getter()
        event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        def push_event(event) -> None:
            if not EventBus._matches_event_name(
                event.name,
                include_prefixes=include_prefixes,
                exclude_prefixes=exclude_prefixes,
            ):
                return
            event_queue.put(
                {
                    "name": event.name,
                    "payload": event.payload,
                    "timestamp": event.timestamp,
                }
            )

        unsubscribe = event_bus.subscribe("*", push_event)
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("Connection", "keep-alive")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Access-Control-Allow-Headers", "Content-Type")
            handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            handler.end_headers()

            self._write_sse_event(
                handler,
                "runtime",
                {
                    "name": "stream.ready",
                    "payload": {"bridge": self.status().__dict__, "profile": profile_name},
                    "timestamp": None,
                },
            )

            while self._server is not None:
                try:
                    payload = event_queue.get(timeout=10.0)
                    self._write_sse_event(handler, "runtime", payload)
                except queue.Empty:
                    handler.wfile.write(b": ping\n\n")
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            handler.close_connection = True
            unsubscribe()

    @staticmethod
    def _write_sse_event(handler: BaseHTTPRequestHandler, event_name: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        body = f"event: {event_name}\n"
        for line in data.splitlines() or ["{}"]:
            body += f"data: {line}\n"
        body += "\n"
        handler.wfile.write(body.encode("utf-8"))
        handler.wfile.flush()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def status(self) -> RuntimeHttpBridgeStatus:
        enabled = self._server is not None
        url = f"http://{self._host}:{self._port}" if enabled else None
        return RuntimeHttpBridgeStatus(
            enabled=enabled,
            host=self._host if enabled else None,
            port=self._port if enabled else None,
            url=url,
        )
