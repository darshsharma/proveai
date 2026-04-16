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
    at_door: bool                  # agent currently standing on door cell
    partner_distance: int          # Manhattan to other agent
    game_turn_parity: str          # "even" / "odd" — helps spot ordering bugs
    # Cross-turn tracking
    position_revisit_count: int = 0    # # of times agent previously stood on current cell
    moves_with_key_count: int = 0      # cumulative moves this agent has made since picking up key
    llm_raw_content: str = ""          # raw assistant message text (if any)
    llm_raw_tool_calls_json: str = ""  # raw tool_calls JSON string (if any)
    timestamp_utc: str = ""


def _find_cell(grid: tuple, cell_value: str) -> tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell.value == cell_value:
                return (r, c)
    return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



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
    bus: "Any" = None,
) -> StepRecord:
    from .state import Cell
    from .events import EventType

    door_pos = _find_cell(game_state.grid, Cell.DOOR.value)
    dist = _manhattan(agent_state.position, door_pos) if door_pos else -1

    other_agents = [a for aid, a in game_state.agents.items() if aid != agent_id]
    partner_dist = _manhattan(agent_state.position, other_agents[0].position) if other_agents else -1

    msg_blob = "|".join(messages) if messages else ""
    msg_hash = hashlib.md5(msg_blob.encode()).hexdigest()[:12] if msg_blob else ""

    # Cross-turn counters derived from event log.
    revisit_count = 0
    moves_with_key = 0
    if bus is not None:
        pickup_turn: int | None = None
        for ev in bus.events_for_agent(agent_id):
            if ev.event_type == EventType.TOOL_RESOLUTION and ev.payload.success:
                if ev.payload.tool_name == "pickup" and pickup_turn is None:
                    pickup_turn = ev.turn
                if ev.payload.tool_name == "move" and ev.turn < turn:
                    if ev.payload.actual_position == agent_state.position:
                        revisit_count += 1
                    if pickup_turn is not None and ev.turn >= pickup_turn:
                        moves_with_key += 1

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
        at_door=(door_pos is not None and agent_state.position == door_pos),
        partner_distance=partner_dist,
        game_turn_parity="even" if turn % 2 == 0 else "odd",
        position_revisit_count=revisit_count,
        moves_with_key_count=moves_with_key,
        llm_raw_content=tool_call.raw_content,
        llm_raw_tool_calls_json=tool_call.raw_tool_calls_json,
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
# Run-level summary (cross-turn metrics written at game end)
# ---------------------------------------------------------------------------

def _msg_mentions(content: str, keywords: tuple[str, ...]) -> bool:
    up = content.upper()
    return any(k in up for k in keywords)


def write_run_summary(
    run_id: str,
    final_state: "GameState",
    bus: "Any",
    output_dir: str | Path = "traces",
) -> Path:
    """Compute cross-turn metrics and write them as JSON next to the JSONL trace.

    Metrics:
      - per-agent: total_turns, total_moves, total_revisits, max_visits_on_cell,
        moves_with_key, at_door_final
      - key_pickup_turn, partner_learned_key_turn, key_info_delay_turns
      - first_door_sight_turn, partner_learned_door_turn, door_info_delay_turns
    """
    from .events import EventType

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.summary.json"

    events = bus.events
    agent_ids = sorted(final_state.agents.keys())

    # Per-agent metrics
    per_agent: dict[str, dict[str, Any]] = {}
    for aid in agent_ids:
        visits: dict[tuple[int, int], int] = {}
        total_moves = 0
        moves_with_key = 0
        pickup_turn: int | None = None
        for ev in events:
            if ev.agent_id != aid:
                continue
            if ev.event_type != EventType.TOOL_RESOLUTION or not ev.payload.success:
                continue
            if ev.payload.tool_name == "pickup" and pickup_turn is None:
                pickup_turn = ev.turn
            if ev.payload.tool_name == "move":
                total_moves += 1
                pos = ev.payload.actual_position
                if pos is not None:
                    visits[pos] = visits.get(pos, 0) + 1
                if pickup_turn is not None and ev.turn >= pickup_turn:
                    moves_with_key += 1
        total_revisits = sum(v - 1 for v in visits.values() if v > 1)
        max_visits = max(visits.values()) if visits else 0
        door_pos = _find_cell(final_state.grid, "D")
        per_agent[aid] = {
            "final_position": final_state.agents[aid].position,
            "has_key_final": final_state.agents[aid].has_key,
            "at_door_final": (door_pos is not None
                              and final_state.agents[aid].position == door_pos),
            "total_moves": total_moves,
            "total_revisits": total_revisits,
            "max_visits_on_single_cell": max_visits,
            "pickup_turn": pickup_turn,
            "moves_with_key": moves_with_key,
        }

    # Key info propagation
    key_pickup_turn: int | None = None
    key_pickup_by: str | None = None
    for ev in events:
        if (ev.event_type == EventType.TOOL_RESOLUTION
                and ev.payload.tool_name == "pickup"
                and ev.payload.success):
            key_pickup_turn = ev.turn
            key_pickup_by = ev.agent_id
            break

    partner_learned_key_turn: int | None = None
    if key_pickup_by is not None:
        key_keywords = ("KEY", "PICKED", "GOT THE KEY", "HAVE THE KEY", "HAS KEY")
        for ev in events:
            if (ev.event_type == EventType.MESSAGE_DELIVERED
                    and ev.agent_id != key_pickup_by
                    and ev.turn >= (key_pickup_turn or 0)
                    and _msg_mentions(ev.payload.get("content", ""), key_keywords)):
                partner_learned_key_turn = ev.turn
                break

    key_delay = (partner_learned_key_turn - key_pickup_turn
                 if partner_learned_key_turn is not None and key_pickup_turn is not None
                 else None)

    # Door info propagation — first observe resolution whose description
    # reveals a DOOR in an adjacent cell.
    first_door_sight_turn: int | None = None
    door_sighter: str | None = None
    for ev in events:
        if (ev.event_type == EventType.TOOL_RESOLUTION
                and ev.payload.tool_name == "observe"
                and ev.payload.success):
            desc = ev.payload.description
            if ": D" in desc or desc.startswith("D") or ", D" in desc:
                first_door_sight_turn = ev.turn
                door_sighter = ev.agent_id
                break

    partner_learned_door_turn: int | None = None
    if door_sighter is not None:
        door_keywords = ("DOOR",)
        for ev in events:
            if (ev.event_type == EventType.MESSAGE_DELIVERED
                    and ev.agent_id != door_sighter
                    and ev.turn >= (first_door_sight_turn or 0)
                    and _msg_mentions(ev.payload.get("content", ""), door_keywords)):
                partner_learned_door_turn = ev.turn
                break

    door_delay = (partner_learned_door_turn - first_door_sight_turn
                  if partner_learned_door_turn is not None and first_door_sight_turn is not None
                  else None)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "result": "WIN" if final_state.win else "LOSS",
        "total_turns": final_state.turn,
        "per_agent": per_agent,
        "key_info_propagation": {
            "key_pickup_turn": key_pickup_turn,
            "key_pickup_by": key_pickup_by,
            "partner_learned_key_turn": partner_learned_key_turn,
            "key_info_delay_turns": key_delay,
        },
        "door_info_propagation": {
            "first_door_sight_turn": first_door_sight_turn,
            "door_sighter": door_sighter,
            "partner_learned_door_turn": partner_learned_door_turn,
            "door_info_delay_turns": door_delay,
        },
    }

    with path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    return path


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

    def start_decide_span(
        self,
        parent: SpanHandle,
        agent_id: str,
        input_data: dict[str, Any],
        model: str = "mock",
    ) -> SpanHandle:
        if not parent.active:
            return _NULL_SPAN
        return parent.generation(
            name=f"decide-{agent_id}",
            input=input_data,
            model=model,
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

    # -- scores ------------------------------------------------------------

    def score(
        self,
        name: str,
        value: float | int,
        data_type: str = "NUMERIC",
        comment: str | None = None,
    ) -> None:
        """Attach a score to the current game trace."""
        if self._client is None or not self._trace_id:
            return
        try:
            self._client.score(
                trace_id=self._trace_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment,
            )
        except Exception:
            # Don't let scoring failures break the run.
            pass

    def score_game(self, final_state: "GameState", bus: "Any") -> None:
        """Write the canonical set of game scores to Langfuse."""
        from .events import EventType
        from .state import Cell

        # Key ever picked up?
        key_pickups = [
            e for e in bus.events
            if e.event_type == EventType.TOOL_RESOLUTION
            and e.payload.tool_name == "pickup"
            and e.payload.success
        ]
        agents_with_key_final = sum(1 for a in final_state.agents.values() if a.has_key)

        # Door observed by any agent?
        door_sighted_by: set[str] = set()
        for e in bus.events:
            if (e.event_type == EventType.TOOL_RESOLUTION
                    and e.payload.tool_name == "observe"
                    and e.payload.success):
                desc = e.payload.description
                if ": D" in desc or desc.startswith("D") or ", D" in desc:
                    door_sighted_by.add(e.agent_id)

        # Agents standing on the door at end-of-game
        door_pos = None
        for r, row in enumerate(final_state.grid):
            for c, cell in enumerate(row):
                if cell == Cell.DOOR:
                    door_pos = (r, c)
        agents_at_door_final = sum(
            1 for a in final_state.agents.values() if a.position == door_pos
        )

        total_drifts = sum(
            1 for e in bus.events if e.event_type == EventType.DRIFT_DETECTED
        )
        total_messages = sum(
            1 for e in bus.events if e.event_type == EventType.MESSAGE_SENT
        )

        self.score("success", 1 if final_state.win else 0, "BOOLEAN")
        self.score("key_found", 1 if key_pickups else 0, "BOOLEAN")
        self.score("door_found", 1 if door_sighted_by else 0, "BOOLEAN")
        self.score("agents_with_key_final", agents_with_key_final, "NUMERIC")
        self.score("agents_at_door_final", agents_at_door_final, "NUMERIC")
        self.score("door_found_by_n_agents", len(door_sighted_by), "NUMERIC")
        self.score("total_turns", final_state.turn, "NUMERIC")
        self.score("total_drifts", total_drifts, "NUMERIC")
        self.score("total_messages_sent", total_messages, "NUMERIC")

    # -- flush -------------------------------------------------------------

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


def generate_run_id() -> str:
    return str(uuid.uuid4())
