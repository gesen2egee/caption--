# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from lib.runtime.events import EventBus
from lib.runtime.errors import build_runtime_error_info


@dataclass
class CommandDefinition:
    name: str
    handler: Callable[..., Any]
    access_modes: Tuple[str, ...] = field(default_factory=lambda: ("internal",))
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommandAccessError(PermissionError):
    def __init__(
        self,
        message: str,
        *,
        command_name: Optional[str] = None,
        access_mode: Optional[str] = None,
        allowed_access_modes: Optional[Iterable[str]] = None,
    ):
        super().__init__(message)
        self.command_name = command_name
        self.access_mode = access_mode
        self.allowed_access_modes = tuple(str(mode) for mode in (allowed_access_modes or []))


class CommandNotFoundError(KeyError):
    def __init__(self, command_name: str):
        super().__init__(f"Unknown command: {command_name}")
        self.command_name = command_name


def _truncate_text(value: str, limit: int = 220) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _summarize_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, dict):
        items = list(value.items())[:20]
        result = {str(key): _summarize_value(item, depth + 1) for key, item in items}
        if len(value) > 20:
            result["__truncated__"] = f"+{len(value) - 20} keys"
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:20]
        result = [_summarize_value(item, depth + 1) for item in items]
        if len(value) > 20:
            result.append(f"... +{len(value) - 20} items")
        return result
    return _truncate_text(repr(value))


class CommandRegistry:
    """
    Registry for app-level commands.

    Commands are the stable boundary that future UI runtimes and Agents should
    call instead of reaching into widgets or mixins directly.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._commands: Dict[str, CommandDefinition] = {}
        self._event_bus = event_bus

    @staticmethod
    def _normalize_access_modes(access_modes: Optional[Iterable[str]]) -> Tuple[str, ...]:
        if access_modes is None:
            return ("internal",)
        values = []
        for mode in access_modes:
            value = str(mode or "").strip().lower()
            if value and value not in values:
                values.append(value)
        return tuple(values or ["internal"])

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        access_modes: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._commands[name] = CommandDefinition(
            name=name,
            handler=handler,
            access_modes=self._normalize_access_modes(access_modes),
            metadata=deepcopy(metadata or {}),
        )

    def _get_command(self, name: str) -> CommandDefinition:
        command = self._commands.get(name)
        if command is None:
            raise CommandNotFoundError(name)
        return command

    def _assert_access(self, command: CommandDefinition, access_mode: str) -> None:
        normalized_mode = str(access_mode or "internal").strip().lower() or "internal"
        if normalized_mode in command.access_modes:
            return
        raise CommandAccessError(
            f"Command '{command.name}' is not available in access mode '{normalized_mode}'",
            command_name=command.name,
            access_mode=normalized_mode,
            allowed_access_modes=command.access_modes,
        )

    def execute(self, name: str, *args, access_mode: str = "internal", **kwargs) -> Any:
        command = self._get_command(name)
        normalized_mode = str(access_mode or "internal").strip().lower() or "internal"

        try:
            self._assert_access(command, normalized_mode)
            if self._event_bus is not None:
                self._event_bus.emit(
                    "command.started",
                    {
                        "command_name": name,
                        "access_mode": normalized_mode,
                        "command_metadata": deepcopy(command.metadata),
                        "args": _summarize_value(list(args)),
                        "kwargs": _summarize_value(kwargs),
                    },
                )
            result = command.handler(*args, **kwargs)
            if self._event_bus is not None:
                self._event_bus.emit(
                    "command.finished",
                    {
                        "command_name": name,
                        "access_mode": normalized_mode,
                        "command_metadata": deepcopy(command.metadata),
                        "result": _summarize_value(result),
                    },
                )
            return result
        except Exception as exc:
            error_info = build_runtime_error_info(
                exc,
                source="runtime.command",
                command_name=name,
                access_mode=normalized_mode,
            )
            if self._event_bus is not None:
                if isinstance(exc, CommandAccessError):
                    self._event_bus.emit(
                        "command.blocked",
                        {
                            "command_name": name,
                            "access_mode": normalized_mode,
                            "allowed_access_modes": list(command.access_modes),
                            "command_metadata": deepcopy(command.metadata),
                            "error": error_info["message"],
                            "error_info": error_info,
                        },
                    )
                self._event_bus.emit(
                    "command.failed",
                    {
                        "command_name": name,
                        "access_mode": normalized_mode,
                        "command_metadata": deepcopy(command.metadata),
                        "error": error_info["message"],
                        "error_info": error_info,
                    },
                )
            raise

    def has(self, name: str) -> bool:
        return name in self._commands

    def list_commands(self, access_mode: Optional[str] = None) -> List[str]:
        if access_mode is None:
            return sorted(self._commands.keys())
        normalized_mode = str(access_mode or "").strip().lower()
        return sorted(
            name
            for name, command in self._commands.items()
            if normalized_mode in command.access_modes
        )

    def get_command_access_modes(self) -> Dict[str, List[str]]:
        return {
            name: list(command.access_modes)
            for name, command in sorted(self._commands.items())
        }

    def get_command_metadata(self, access_mode: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        if access_mode is None:
            items = self._commands.items()
        else:
            normalized_mode = str(access_mode or "").strip().lower()
            items = (
                (name, command)
                for name, command in self._commands.items()
                if normalized_mode in command.access_modes
            )
        return {
            name: deepcopy(command.metadata)
            for name, command in sorted(items)
        }
