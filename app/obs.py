"""Observability: structured JSON logs + Prometheus metrics.

Design notes:
- Logs are JSON so they can be shipped to Loki/ELK and queried by request_id.
- Audio bytes are NEVER logged (PII); only durations, sample rates, timings.
- A ContextVar carries the request id into every log line, including from
  worker threads that run model inference.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar

from prometheus_client import Counter, Gauge, Histogram

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                continue
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # uvicorn's default access log duplicates our middleware; silence it.
    for noisy in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True


REQUEST_LATENCY = Histogram(
    "vis_request_latency_seconds",
    "End-to-end HTTP request latency",
    ["route", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0),
)
STAGE_LATENCY = Histogram(
    "vis_stage_latency_seconds",
    "Latency of individual pipeline stages",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
REQUESTS_TOTAL = Counter("vis_requests_total", "HTTP requests", ["route", "status"])
ERRORS_TOTAL = Counter("vis_errors_total", "Handled errors", ["kind"])
WS_SESSIONS = Gauge("vis_active_ws_sessions", "Currently open WebSocket sessions")
MODEL_INFO = Gauge("vis_model_info", "Model metadata, value is always 1", ["model_id", "quantized"])


@contextmanager
def stage(name: str):
    """Time a pipeline stage; emits histogram + returns duration via `as ms`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(name).observe(time.perf_counter() - t0)


class StageTimer:
    """Collects per-request stage durations for the response/log payload."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: dict[str, int] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = int((time.perf_counter() - t0) * 1000)

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)
