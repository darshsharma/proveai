"""Immutable game state objects.

All state is frozen. Updates produce new instances via dataclasses.replace().
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Grid cell types
# ---------------------------------------------------------------------------

class Cell(str, Enum):
    EMPTY = "."
    KEY = "K"
    DOOR = "D"
    OBSTACLE = "#"


# ---------------------------------------------------------------------------
# Behavioral states (legibility layer)
# ---------------------------------------------------------------------------

class BehavioralState(str, Enum):
    EXPLORING_BLIND = "EXPLORING_BLIND"
    NAVIGATING_TO_KEY = "NAVIGATING_TO_KEY"
    NAVIGATING_TO_DOOR = "NAVIGATING_TO_DOOR"
    WAITING_AT_DOOR = "WAITING_AT_DOOR"
    GUIDING_PARTNER = "GUIDING_PARTNER"
    STUCK_RECOVERY = "STUCK_RECOVERY"
    BACKTRACKING = "BACKTRACKING"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentState:
    agent_id: str
    position: tuple[int, int]
    has_key: bool = False
    last_tool: str | None = None
    behavioral_state: BehavioralState = BehavioralState.EXPLORING_BLIND
    # Tracks what the agent has seen so far (cell positions it knows about)
    known_positions: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    # Whether agent knows where key/door is
    knows_key_location: bool = False
    knows_door_location: bool = False
    consecutive_drift_count: int = 0
    # History of recent positions for backtracking detection
    recent_positions: tuple[tuple[int, int], ...] = ()


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Message:
    sender: str
    recipient: str
    content: str
    turn: int


# ---------------------------------------------------------------------------
# Top-level game state
# ---------------------------------------------------------------------------

Grid = tuple[tuple[Cell, ...], ...]


@dataclass(frozen=True)
class GameState:
    grid: Grid
    agents: dict[str, AgentState]          # agent_id -> AgentState
    pending_messages: tuple[Message, ...]   # undelivered messages
    turn: int = 0
    current_agent_id: str = "agent_0"
    game_over: bool = False
    win: bool = False
    max_turns: int = 200

    # --- helper updaters (return new GameState) ---

    def with_agent(self, agent_id: str, **kwargs: Any) -> GameState:
        """Return new GameState with one agent's fields updated."""
        old = self.agents[agent_id]
        new_agent = replace(old, **kwargs)
        new_agents = {**self.agents, agent_id: new_agent}
        return replace(self, agents=new_agents)

    def next_turn(self) -> GameState:
        """Advance turn counter and swap current agent."""
        agent_ids = sorted(self.agents.keys())
        idx = agent_ids.index(self.current_agent_id)
        next_id = agent_ids[(idx + 1) % len(agent_ids)]
        return replace(self, turn=self.turn + 1, current_agent_id=next_id)

    def add_message(self, msg: Message) -> GameState:
        return replace(self, pending_messages=self.pending_messages + (msg,))

    def pop_messages_for(self, agent_id: str) -> tuple[GameState, list[Message]]:
        """Return (new_state_without_those_messages, list_of_messages)."""
        deliver = [m for m in self.pending_messages if m.recipient == agent_id]
        keep = tuple(m for m in self.pending_messages if m.recipient != agent_id)
        return replace(self, pending_messages=keep), deliver

    def set_win(self) -> GameState:
        return replace(self, game_over=True, win=True)

    def set_loss(self, reason: str = "") -> GameState:
        return replace(self, game_over=True, win=False)
