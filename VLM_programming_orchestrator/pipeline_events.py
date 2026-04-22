"""Thread-safe event channel between the orchestrator (worker thread) and GUI.

The orchestrator posts events such as state changes and log lines onto the
queue. Human-review requests are also sent through the queue: the worker
thread blocks on a `threading.Event` while the GUI thread renders the dialog
and sets the result.
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


# Event type constants
EV_STATE = "state_changed"
EV_LOG = "log"
EV_CYCLE_DONE = "cycle_done"
EV_CYCLE_ERROR = "cycle_error"
EV_HUMAN_REVIEW = "human_review_request"
EV_SKIP_VALIDATION = "skip_validation_request"
EV_LLM_STREAM_START = "llm_stream_start"
EV_LLM_STREAM_CHUNK = "llm_stream_chunk"
EV_LLM_STREAM_END = "llm_stream_end"


@dataclass
class PipelineRequest:
    """A request from the worker that needs a synchronous reply from the GUI."""
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None


class PipelineEvents:
    """A simple wrapper around `queue.Queue` plus a stop flag."""

    def __init__(self) -> None:
        self.q: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self.stop_flag = threading.Event()

    # ---- producer side (worker thread) -------------------------------
    def post(self, kind: str, **payload) -> None:
        self.q.put((kind, payload))

    def state_changed(self, state_name: str, attempt: int = 0) -> None:
        self.post(EV_STATE, state=state_name, attempt=attempt)

    def cycle_done(self) -> None:
        self.post(EV_CYCLE_DONE)

    def cycle_error(self, message: str) -> None:
        self.post(EV_CYCLE_ERROR, message=message)

    def request_human_review(self) -> tuple[bool, str]:
        req = PipelineRequest()
        self.post(EV_HUMAN_REVIEW, request=req)
        req.done.wait()
        return req.result if req.result is not None else (False, "")

    def llm_stream_start(self, label: str = "") -> None:
        self.post(EV_LLM_STREAM_START, label=label)

    def llm_stream_chunk(self, chunk_type: str, content: str) -> None:
        self.post(EV_LLM_STREAM_CHUNK, chunk_type=chunk_type, content=content)

    def llm_stream_end(self, success: bool = True, message: str = "") -> None:
        self.post(EV_LLM_STREAM_END, success=success, message=message)

    def request_skip_validation(self) -> bool:
        req = PipelineRequest()
        self.post(EV_SKIP_VALIDATION, request=req)
        req.done.wait()
        return bool(req.result)

    # ---- consumer side (GUI thread) ----------------------------------
    def drain(self):
        """Yield all currently-queued events without blocking."""
        while True:
            try:
                yield self.q.get_nowait()
            except queue.Empty:
                return

    # ---- stop signalling ---------------------------------------------
    def request_stop(self) -> None:
        self.stop_flag.set()

    def stop_requested(self) -> bool:
        return self.stop_flag.is_set()


class QueueLogHandler(logging.Handler):
    """Forward log records onto the PipelineEvents queue as `log` events."""

    def __init__(self, events: PipelineEvents) -> None:
        super().__init__()
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # pylint: disable=W0718
            msg = record.getMessage()
        self.events.post(EV_LOG, level=record.levelname, message=msg)
