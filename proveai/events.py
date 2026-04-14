"""Event bus — intent/resolution payloads and drift detection.

Every tool call emits an IntentPayload (what the agent expected) and a
ResolutionPayload (what actually happened). When they diverge a
DRIFT_DETECTED event is also emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(str, Enum):
    TOOL_INTENT = "TOOL_INTENT"
    TOOL_RESOLUTION = "TOOL_RESOLUTION"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    STATE_TRANSITION = "STATE_TRANSITION"
    MESSAGE_SENT = "MESSAGE_SENT"
    MESSAGE_DELIVERED = "MESSAGE_DELIVERED"
    GAME_OVER = "GAME_OVER"
    WIN_CONDITION_CHECK = "WIN_CONDITION_CHECK"


@dataclass(frozen=True)
class IntentPayload:
    """What the agent expected to happen."""
    agent_id: str
    tool_name: str
    description: str          # e.g. "Moving NORTH to (3, 5)"
    expected_position: tuple[int, int] | None = None
    expected_pickup: str | None = None
    raw_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolutionPayload:
    """What actually happened."""
    agent_id: str
    tool_name: str
    description: str          # e.g. "Agent remains at (3, 4) — wall"
    actual_position: tuple[int, int] | None = None
    success: bool = True
    reason: str = ""


@dataclass(frozen=True)
class DriftPayload:
    """Emitted when intent ≠ resolution."""
    agent_id: str
    tool_name: str
    intent_description: str
    resolution_description: str
    drift_type: str            # e.g. "MOVE_BLOCKED", "PICKUP_EMPTY"


@dataclass(frozen=True)
class Event:
    event_type: EventType
    turn: int
    agent_id: str
    payload: IntentPayload | ResolutionPayload | DriftPayload | dict[str, Any]


class EventBus:
    """Append-only event log. The single source of truth for legibility."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def emit(self, event: Event) -> None:
        self._events.append(event)

    def emit_tool_call(
        self,
        turn: int,
        intent: IntentPayload,
        resolution: ResolutionPayload,
    ) -> DriftPayload | None:
        """Emit intent + resolution, and auto-detect drift. Returns drift if any."""
        self.emit(Event(EventType.TOOL_INTENT, turn, intent.agent_id, intent))
        self.emit(Event(EventType.TOOL_RESOLUTION, turn, resolution.agent_id, resolution))

        drift: DriftPayload | None = None
        if not resolution.success:
            drift_type = f"{intent.tool_name.upper()}_FAILED"
            drift = DriftPayload(
                agent_id=intent.agent_id,
                tool_name=intent.tool_name,
                intent_description=intent.description,
                resolution_description=resolution.description,
                drift_type=drift_type,
            )
            self.emit(Event(EventType.DRIFT_DETECTED, turn, intent.agent_id, drift))

        return drift

    def events_for_agent(self, agent_id: str) -> list[Event]:
        return [e for e in self._events if e.agent_id == agent_id]

    def drifts_for_agent(self, agent_id: str) -> list[Event]:
        return [
            e for e in self._events
            if e.agent_id == agent_id and e.event_type == EventType.DRIFT_DETECTED
        ]
