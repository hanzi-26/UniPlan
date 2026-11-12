from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import signal
import threading
import time
from typing import Iterator

_evaluation_deadline = ContextVar("evaluation_deadline", default=None)
_request_audit = ContextVar("request_audit", default=None)


def evaluation_deadline_ns():
    return _evaluation_deadline.get()


def record_request_audit(**fields):
    """Preserve a selected configuration even if SIGALRM aborts its executor."""
    audit = _request_audit.get()
    if audit is not None:
        audit.update(fields)


class EvaluationDeadlineExceeded(BaseException):
    """Bypass per-path Exception handlers: the whole request has expired."""


def run_with_deadline(call, query, cap_ms, cancel_pending=None):
    """POSIX/main-thread evaluation boundary, including embedding and inference.

    SIGALRM interrupts Python/socket waits; the watchdog also cancels PostgreSQL
    independently so transaction cleanup does not wait for the old SQL timeout.
    Native kernels acknowledge signals on return; report any cancellation lag.
    """
    if cap_ms <= 0:
        raise ValueError("deadline budget must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("hard-deadline evaluation must run on the main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("another real-time alarm is already active")
    started = time.perf_counter_ns()
    token = _evaluation_deadline.set(started + int(cap_ms * 1_000_000))
    audit = {}
    audit_token = _request_audit.set(audit)
    previous = signal.getsignal(signal.SIGALRM)
    deadline_reached = False
    interruptible = False
    def expired(*_):
        nonlocal deadline_reached
        deadline_reached = True
        # A pending alarm may be delivered just after call() returns. It must
        # not interrupt result accounting or cleanup outside the request.
        if interruptible:
            raise EvaluationDeadlineExceeded()
    def cancel():
        if cancel_pending is not None:
            cancel_pending()
    watchdog = threading.Timer(cap_ms / 1000, cancel)
    watchdog.daemon = True
    signal.signal(signal.SIGALRM, expired)
    try:
        try:
            try:
                interruptible = True
                watchdog.start()
                signal.setitimer(signal.ITIMER_REAL, cap_ms / 1000)
                row = call()
                row = {**audit, **row}
            finally:
                # Keep disarming inside the protected region: SIGALRM can race
                # with the return from call() or the first cleanup instruction.
                interruptible = False
                signal.setitimer(signal.ITIMER_REAL, 0)
        except EvaluationDeadlineExceeded:
            row = {"query_id": query.query_id, "k": query.k, "article_ids": [],
                   "path_id": None, **audit, "error_type": "E2EDeadlineExceeded",
                   "error_message": "end-to-end evaluation deadline reached"}
        observed = (time.perf_counter_ns() - started) / 1_000_000
        censored = deadline_reached or row.get("error_type") == "E2EDeadlineExceeded" or observed >= cap_ms
        deadline_reached = censored
        if censored:
            if row.get("error_type") and row["error_type"] != "E2EDeadlineExceeded":
                row["backend_error_type"] = row["error_type"]
                row["backend_error_message"] = row.get("error_message")
            row.update(article_ids=[], error_type="E2EDeadlineExceeded", result_full_k=False,
                       result_fill_count=0, deadline_pass=False)
        row.update(e2e_latency_ms=observed, timed_out=censored, latency_censored=censored,
                   completion_latency_ms=None if censored else observed,
                   latency_lower_bound_ms=cap_ms if censored else None,
                   deadline_budget_ms=cap_ms, cancellation_lag_ms=max(0.0, observed-cap_ms) if censored else 0.0,
                   deadline_protocol="e2e-cancel-v1")
        return row
    finally:
        interruptible = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        if not deadline_reached:
            watchdog.cancel()
        # Do not let the old watchdog cancel the next request on a reused
        # connection. The PostgreSQL cancellation itself has a 1-second timeout.
        watchdog.join()
        signal.signal(signal.SIGALRM, previous)
        _evaluation_deadline.reset(token)
        _request_audit.reset(audit_token)


@dataclass
class RequestTimer:
    """One monotonic request clock with non-overlapping named spans."""

    latency_cap_ms: float
    started_ns: int = field(default_factory=time.perf_counter_ns)
    spans_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_cap_ms <= 0:
            raise ValueError("latency_cap_ms must be positive")

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if name in self.spans_ms:
            raise ValueError(f"timing span already recorded: {name}")
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.spans_ms[name] = (time.perf_counter_ns() - started) / 1_000_000

    @property
    def total_ms(self) -> float:
        return (time.perf_counter_ns() - self.started_ns) / 1_000_000

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self.latency_cap_ms - self.total_ms)

    @property
    def deadline_ns(self) -> int:
        return self.started_ns + int(self.latency_cap_ms * 1_000_000)

    def finish(self) -> dict[str, float | bool | dict[str, float]]:
        total = self.total_ms
        accounted = sum(self.spans_ms.values())
        return {
            "e2e_latency_ms": total,
            "deadline_pass": total <= self.latency_cap_ms,
            "framework_overhead_ms": max(0.0, total - accounted),
            "spans_ms": dict(self.spans_ms),
        }
