from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine, Mapping
from contextlib import suppress
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _run_coroutine_sync(coro: Coroutine[object, object, _T]) -> _T:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        with suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()


def run_async(
    func: Callable[_P, Coroutine[object, object, _T]],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coroutine_sync(func(*args, **kwargs))

    result_container: dict[str, _T] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_container["result"] = _run_coroutine_sync(func(*args, **kwargs))
        except BaseException as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container["result"]


def extract_prompt(inputs: Mapping[str, object]) -> str:
    for key in ("prompt", "user_message", "message", "input"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


__all__ = ["extract_prompt", "run_async"]
