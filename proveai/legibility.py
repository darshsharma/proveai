"""Legibility layer — behavioral state inference and human-readable trace.

Answers three questions for a human observer:
  1. What happened?   (event log)
  2. Why did it happen? (behavioral state + intent/resolution)
  3. What should change? (drift summary + stuck detection)
"""

from __future__ import annotations

from dataclasses import replace

from .events import DriftPayload, Event, EventBus, EventType
from .state import BehavioralState, GameState


# ---------------------------------------------------------------------------
# Behavioral state inference
# ---------------------------------------------------------------------------

def infer_behavioral_state(state: GameState, agent_id: str, bus: EventBus) -> BehavioralState:
    """Infer the current behavioral state of an agent from game state + event history."""
    agent = state.agents[agent_id]

    # STUCK_RECOVERY: 3+ consecutive drifts
    if agent.consecutive_drift_count >= 3:
        return BehavioralState.STUCK_RECOVERY

    # BACKTRACKING: agent is revisiting recent positions
    if len(agent.recent_positions) >= 4:
        unique = set(agent.recent_positions[-4:])
        if len(unique) <= 2:
            return BehavioralState.BACKTRACKING

    # COMMUNICATING: last tool was send_message
    if agent.last_tool == "send_message":
        return BehavioralState.COMMUNICATING

    # WAITING_AT_DOOR: agent is on the door cell
    r, c = agent.position
    from .state import Cell
    if state.grid[r][c] == Cell.DOOR:
        return BehavioralState.WAITING_AT_DOOR

    # DELIVERING_KEY: has key and knows door location
    if agent.has_key and agent.knows_door_location:
        return BehavioralState.DELIVERING_KEY

    # NAVIGATING_TO_KEY: knows key location, doesn't have key
    if agent.knows_key_location and not agent.has_key:
        # Check if key still exists on the grid
        key_exists = any(
            state.grid[row][col] == Cell.KEY
            for row in range(len(state.grid))
            for col in range(len(state.grid[0]))
        )
        if key_exists:
            return BehavioralState.NAVIGATING_TO_KEY

    # NAVIGATING_TO_DOOR: knows door location (and either has key or key is collected by partner)
    if agent.knows_door_location:
        return BehavioralState.NAVIGATING_TO_DOOR

    # GUIDING_PARTNER: agent has useful info and sent a message recently
    recent_events = bus.events_for_agent(agent_id)
    recent_messages = [e for e in recent_events[-6:] if e.event_type == EventType.MESSAGE_SENT]
    if recent_messages and (agent.knows_key_location or agent.knows_door_location):
        return BehavioralState.GUIDING_PARTNER

    return BehavioralState.EXPLORING_BLIND


def update_behavioral_state(state: GameState, agent_id: str, bus: EventBus) -> GameState:
    """Re-infer and update the behavioral state for the given agent."""
    new_bs = infer_behavioral_state(state, agent_id, bus)
    old_bs = state.agents[agent_id].behavioral_state

    if new_bs != old_bs:
        bus.emit(Event(
            EventType.STATE_TRANSITION,
            state.turn,
            agent_id,
            {"from": old_bs.value, "to": new_bs.value},
        ))

    return state.with_agent(agent_id, behavioral_state=new_bs)


# ---------------------------------------------------------------------------
# Trace formatting (human-readable)
# ---------------------------------------------------------------------------

def format_turn_summary(turn: int, bus: EventBus) -> str:
    """Produce a human-readable summary for a single turn."""
    turn_events = [e for e in bus.events if e.turn == turn]
    if not turn_events:
        return f"Turn {turn}: (no events)"

    lines = [f"--- Turn {turn} ---"]
    for ev in turn_events:
        if ev.event_type == EventType.TOOL_INTENT:
            lines.append(f"  INTENT  [{ev.agent_id}] {ev.payload.description}")
        elif ev.event_type == EventType.TOOL_RESOLUTION:
            marker = "OK" if ev.payload.success else "FAIL"
            lines.append(f"  RESULT  [{ev.agent_id}] [{marker}] {ev.payload.description}")
        elif ev.event_type == EventType.DRIFT_DETECTED:
            lines.append(f"  ⚠ DRIFT [{ev.agent_id}] {ev.payload.drift_type}: expected='{ev.payload.intent_description}' actual='{ev.payload.resolution_description}'")
        elif ev.event_type == EventType.STATE_TRANSITION:
            lines.append(f"  STATE   [{ev.agent_id}] {ev.payload['from']} → {ev.payload['to']}")
        elif ev.event_type == EventType.MESSAGE_SENT:
            lines.append(f"  MSG     [{ev.agent_id}] → {ev.payload['recipient']}: \"{ev.payload['content']}\"")
        elif ev.event_type == EventType.MESSAGE_DELIVERED:
            lines.append(f"  DELIVER [{ev.agent_id}] received message from {ev.payload['sender']}")
        elif ev.event_type == EventType.GAME_OVER:
            lines.append(f"  🏁 GAME OVER: {ev.payload}")

    return "\n".join(lines)


def format_game_summary(bus: EventBus) -> str:
    """Full game summary with drift statistics and behavioral state timeline."""
    all_events = bus.events
    if not all_events:
        return "No events recorded."

    max_turn = max(e.turn for e in all_events)

    lines = ["=" * 60, "GAME TRACE", "=" * 60]
    for t in range(max_turn + 1):
        summary = format_turn_summary(t, bus)
        if "no events" not in summary:
            lines.append(summary)

    # Drift summary
    drifts = [e for e in all_events if e.event_type == EventType.DRIFT_DETECTED]
    lines.append("")
    lines.append("=" * 60)
    lines.append("DRIFT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total drifts: {len(drifts)}")
    for agent_id in sorted({e.agent_id for e in all_events}):
        agent_drifts = [d for d in drifts if d.agent_id == agent_id]
        lines.append(f"  {agent_id}: {len(agent_drifts)} drifts")
        by_type: dict[str, int] = {}
        for d in agent_drifts:
            by_type[d.payload.drift_type] = by_type.get(d.payload.drift_type, 0) + 1
        for dtype, count in sorted(by_type.items()):
            lines.append(f"    {dtype}: {count}")

    # State transition timeline
    transitions = [e for e in all_events if e.event_type == EventType.STATE_TRANSITION]
    if transitions:
        lines.append("")
        lines.append("=" * 60)
        lines.append("BEHAVIORAL STATE TIMELINE")
        lines.append("=" * 60)
        for t in transitions:
            lines.append(f"  Turn {t.turn:>3} [{t.agent_id}] {t.payload['from']} → {t.payload['to']}")

    return "\n".join(lines)
