"""Wrapper tích hợp Langfuse Tracing (tự động fallback an toàn nếu chưa cài langfuse)."""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def observe_span(name: str) -> Callable[[F], F]:
    """Decorator ghi nhận độ trễ thực thi và chuyển tiếp tới Langfuse @observe nếu có."""
    try:
        from langfuse.decorators import observe as langfuse_observe  # type: ignore[import-untyped]

        return langfuse_observe(name=name)  # type: ignore[return-value]
    except ImportError:
        pass

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug("Span [%s] hoàn thành trong %.2f ms", name, elapsed_ms)
            return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator
