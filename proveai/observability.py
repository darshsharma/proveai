"""Observability — Langfuse tracing + structured event records.

Langfuse is optional. If not installed, everything degrades to no-ops.
The module also writes structured step records to a JSONL file so that
cross-run analysis works without Langfuse.

Hierarchy:
    Trace: game-run-{run_id}
      └─ Span: turn-{N}-{agent_id}
            ├─ Span: decide-{agent_id}        (LLM / mock decision)
            ├─ Span: tool-{tool_name}          (tool execution)
            ├─ Span: behavioral-update         (legibility inference)
            └─ Event: step-record              (structured record)
      └─ Event: game-over
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import DecisionMetrics, ToolCall
    from .state import AgentState, GameState
    from .tools import ToolResult


# ---------------------------------------------------------------------------
# Structured step record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    run_id: str
    turn_number: int
    agent_id: str
    trace_id: str
    llm_latency_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    agent_location_x: int          # column
    agent_location_y: int          # row
    has_key: bool
    received_message_hash: str
    intended_tool: str
    intended_args: dict[str, Any]
    resolved_tool_output: str
    tool_success: bool
    state: str                     # BehavioralState value
    distance_to_goal: int          # Manhattan to door, -1 if unknown
    # --- additional diagnostic fields ---
    consecutive_drift_count: int
    known_positions_count: int
    knows_key_location: bool
    knows_door_location: bool
    partner_distance: int          # Manhattan to other agent
    game_turn_parity: str          # "even" / "odd" — helps spot ordering bugs
    exploration_coverage: float    # fraction of passable cells visited
    timestamp_utc: str = ""


def _find_cell(grid: tuple, cell_value: str) -> tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell.value == cell_value:
                return (r, c)
    return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _count_passable(grid: tuple) -> int:
    from .state import Cell
    return sum(1 for row in grid for cell in row if cell != Cell.OBSTACLE)


def build_step_record(
    run_id: str,
    trace_id: str,
    turn: int,
    agent_id: str,
    agent_state: "AgentState",
    game_state: "GameState",
    tool_call: "ToolCall",
    tool_result: "ToolResult",
    messages: list[str],
) -> StepRecord:
    from .state import Cell

    door_pos = _find_cell(game_state.grid, Cell.DOOR.value)
    dist = _manhattan(agent_state.position, door_pos) if door_pos else -1

    other_agents = [a for aid, a in game_state.agents.items() if aid != agent_id]
    partner_dist = _manhattan(agent_state.position, other_agents[0].position) if other_agents else -1

    passable = _count_passable(game_state.grid)
    coverage = len(agent_state.known_positions) / passable if passable else 0.0

    msg_blob = "|".join(messages) if messages else ""
    msg_hash = hashlib.md5(msg_blob.encode()).hexdigest()[:12] if msg_blob else ""

    return StepRecord(
        run_id=run_id,
        turn_number=turn,
        agent_id=agent_id,
        trace_id=trace_id,
        llm_latency_ms=tool_call.metrics.llm_latency_ms,
        total_tokens=tool_call.metrics.total_tokens,
        prompt_tokens=tool_call.metrics.prompt_tokens,
        completion_tokens=tool_call.metrics.completion_tokens,
        agent_location_x=agent_state.position[1],
        agent_location_y=agent_state.position[0],
        has_key=agent_state.has_key,
        received_message_hash=msg_hash,
        intended_tool=tool_call.tool_name,
        intended_args=tool_call.tool_args,
        resolved_tool_output=tool_result.text,
        tool_success=tool_result.success,
        state=agent_state.behavioral_state.value,
        distance_to_goal=dist,
        consecutive_drift_count=agent_state.consecutive_drift_count,
        known_positions_count=len(agent_state.known_positions),
        knows_key_location=agent_state.knows_key_location,
        knows_door_location=agent_state.knows_door_location,
        partner_distance=partner_dist,
        game_turn_parity="even" if turn % 2 == 0 else "odd",
        exploration_coverage=round(coverage, 4),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ---------------------------------------------------------------------------
# JSONL logger (always available, no external deps)
# ---------------------------------------------------------------------------

class StepRecordLogger:
    """Appends StepRecords as JSONL to a file. One file per run."""

    def __init__(self, output_dir: str | Path = "traces") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._path: Path | None = None

    def open(self, run_id: str) -> Path:
        self._path = self._dir / f"{run_id}.jsonl"
        self._file = self._path.open("a")
        return self._path

    def log(self, record: StepRecord) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(asdict(record), default=str) + "\n")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# Span handle (real or no-op)
# ---------------------------------------------------------------------------

class SpanHandle:
    """Wraps a Langfuse span (or nothing)."""

    __slots__ = ("_span",)

    def __init__(self, span: Any = None) -> None:
        self._span = span

    @property
    def active(self) -> bool:
        return self._span is not None

    def span(self, **kwargs: Any) -> "SpanHandle":
        if self._span is None:
            return _NULL_SPAN
        return SpanHandle(self._span.span(**kwargs))

    def event(self, **kwargs: Any) -> None:
        if self._span is not None:
            self._span.event(**kwargs)

    def generation(self, **kwargs: Any) -> "SpanHandle":
        if self._span is None:
            return _NULL_SPAN
        return SpanHandle(self._span.generation(**kwargs))

    def end(self, **kwargs: Any) -> None:
        if self._span is not None:
            self._span.end(**kwargs)

    def update(self, **kwargs: Any) -> None:
        if self._span is not None:
            self._span.update(**kwargs)


_NULL_SPAN = SpanHandle(None)


# ---------------------------------------------------------------------------
# Tracer facade
# ---------------------------------------------------------------------------

class TracerFacade:
    """Thin wrapper around Langfuse. No-ops gracefully when langfuse is missing.

    Usage:
        tracer = TracerFacade(enabled=True)
        tracer.start_game_trace(run_id, {...})
        ...
        tracer.flush()
    """

    def __init__(self, enabled: bool = True) -> None:
        self._client: Any = None
        self._trace: Any = None
        self._trace_id: str = ""
        if enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse()
            except ImportError:
                pass

    @property
    def active(self) -> bool:
        return self._client is not None

    @property
    def trace_id(self) -> str:
        return self._trace_id

    # -- game-level trace --------------------------------------------------

    def start_game_trace(self, run_id: str, metadata: dict[str, Any]) -> None:
        self._trace_id = run_id
        if not self.active:
            return
        self._trace = self._client.trace(
            id=run_id,
            name="game-run",
            metadata=metadata,
        )

    def end_game_trace(self, output: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.update(output=output)

    # -- turn span ---------------------------------------------------------

    def start_turn_span(self, turn: int, agent_id: str, metadata: dict[str, Any] | None = None) -> SpanHandle:
        if self._trace is None:
            return _NULL_SPAN
        span = self._trace.span(
            name=f"turn-{turn}-{agent_id}",
            metadata={"turn": turn, "agent_id": agent_id, **(metadata or {})},
        )
        return SpanHandle(span)

    # -- decide span (LLM generation) -------------------------------------

    def start_decide_span(self, parent: SpanHandle, agent_id: str, input_data: dict[str, Any]) -> SpanHandle:
        if not parent.active:
            return _NULL_SPAN
        return parent.generation(
            name=f"decide-{agent_id}",
            input=input_data,
            model="mock",  # overridden by real LLM agent
        )

    def end_decide_span(
        self,
        handle: SpanHandle,
        output: dict[str, Any],
        metrics: "DecisionMetrics | None" = None,
    ) -> None:
        if not handle.active:
            return
        update_kwargs: dict[str, Any] = {"output": output}
        if metrics is not None:
            update_kwargs["usage"] = {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
            }
            update_kwargs["metadata"] = {"llm_latency_ms": metrics.llm_latency_ms}
        handle.update(**update_kwargs)
        handle.end()

    # -- tool span ---------------------------------------------------------

    def start_tool_span(self, parent: SpanHandle, tool_name: str, tool_args: dict[str, Any]) -> SpanHandle:
        if not parent.active:
            return _NULL_SPAN
        return parent.span(
            name=f"tool-{tool_name}",
            input={"tool": tool_name, "args": tool_args},
        )

    def end_tool_span(self, handle: SpanHandle, output: dict[str, Any]) -> None:
        if not handle.active:
            return
        handle.update(output=output)
        handle.end()

    # -- behavioral update span -------------------------------------------

    def start_behavioral_span(self, parent: SpanHandle) -> SpanHandle:
        if not parent.active:
            return _NULL_SPAN
        return parent.span(name="behavioral-update")

    def end_behavioral_span(self, handle: SpanHandle, old_state: str, new_state: str) -> None:
        if not handle.active:
            return
        handle.update(output={"from": old_state, "to": new_state})
        handle.end()

    # -- step record event ------------------------------------------------

    def log_step_record(self, parent: SpanHandle, record: StepRecord) -> None:
        if not parent.active:
            return
        parent.event(
            name="step-record",
            metadata=asdict(record),
        )

    # -- game-over event --------------------------------------------------

    def log_game_over(self, result: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.event(name="game-over", metadata=result)

    # -- flush -------------------------------------------------------------

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


def generate_run_id() -> str:
    return str(uuid.uuid4())
