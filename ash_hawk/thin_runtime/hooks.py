from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from ash_hawk.thin_runtime.models import HookEvent, HookSpec, HookStage

HookHandler = Callable[[HookEvent], None]


class HookRegistry:
    def __init__(self, hooks: list[HookSpec]) -> None:
        self._hooks = {hook.name: hook for hook in hooks}

    def get(self, name: str) -> HookSpec:
        try:
            return self._hooks[name]
        except KeyError as exc:
            raise ValueError(f"Unknown thin runtime hook: {name}") from exc

    def list_hooks(self) -> list[HookSpec]:
        return list(self._hooks.values())


class HookDispatcher:
    def __init__(self, registry: HookRegistry) -> None:
        self._registry = registry
        self._handlers: dict[str, list[HookHandler]] = defaultdict(list)
        self._emitted: list[HookEvent] = []

    def register(self, hook_name: str, handler: HookHandler) -> None:
        self._registry.get(hook_name)
        self._handlers[hook_name].append(handler)

    def emit(self, hook_name: str, payload: dict[str, object] | None = None) -> HookEvent:
        spec = self._registry.get(hook_name)
        event = HookEvent(hook_name=hook_name, stage=spec.stage, payload=dict(payload or {}))
        self._emitted.append(event)
        for handler in self._handlers.get(hook_name, []):
            handler(event)
        return event

    def emitted(self) -> list[HookEvent]:
        return list(self._emitted)

    def stages(self) -> list[HookStage]:
        return [event.stage for event in self._emitted]
