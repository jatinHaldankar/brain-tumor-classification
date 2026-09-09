"""Optional DeepEval tracing helpers."""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from deepeval.tracing import observe, trace_manager, update_current_span, update_current_trace
except Exception:  # pragma: no cover - fallback for environments without deepeval

    def observe(*decorator_args, **decorator_kwargs):
        def decorator(func: F) -> F:
            return func

        if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
            return decorator_args[0]
        return decorator

    def update_current_span(*args, **kwargs):
        return None

    def update_current_trace(*args, **kwargs):
        return None

    class _TraceManager:
        def configure(self, **kwargs):
            return None

    trace_manager = _TraceManager()


def configure_confident_tracing() -> None:
    confident_api_key = os.getenv("CONFIDENT_API_KEY")
    if confident_api_key:
        trace_manager.configure(confident_api_key=confident_api_key)
