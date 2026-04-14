"""Agent interface and mock LLM implementation.

The MockAgent uses simple heuristics to explore, pick up the key, navigate
to the door, and communicate. It serves as a placeholder for real LLM agents.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .state import Cell, GameState


@dataclass
class DecisionMetrics:
    """Metrics from the agent's decision process (LLM call or mock)."""
    llm_latency_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ToolCall:
    tool_name: str
    tool_args: dict[str, Any]
    metrics: DecisionMetrics = field(default_factory=DecisionMetrics)


class BaseAgent(ABC):
    """Interface that both mock and real LLM agents implement."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def decide(
        self,
        state: GameState,
        observation: str,
        messages: list[str],
    ) -> ToolCall:
        """Given current context, return a single tool call for this turn."""
        ...


class MockAgent(BaseAgent):
    """Heuristic-based mock that exercises the full tool set.

    Strategy:
      1. Observe on the first turn or when surroundings are unknown.
      2. If standing on the key, pick it up.
      3. If key/door location is known, navigate toward it.
      4. Otherwise, move randomly (avoiding known walls).
      5. Occasionally send messages to share discoveries.
    """

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        super().__init__(agent_id)
        self.rng = random.Random(seed)
        self.turn_count = 0
        self._last_observation: dict[str, str] | None = None
        self._known_key_pos: tuple[int, int] | None = None
        self._known_door_pos: tuple[int, int] | None = None
        self._shared_key_info = False
        self._shared_door_info = False

    def decide(
        self,
        state: GameState,
        observation: str,
        messages: list[str],
    ) -> ToolCall:
        self.turn_count += 1
        agent = state.agents[self.agent_id]
        r, c = agent.position

        # Parse incoming messages for key/door intel
        for msg in messages:
            if "KEY at" in msg:
                try:
                    coords = msg.split("KEY at ")[1].strip("()")
                    row, col = map(int, coords.split(","))
                    self._known_key_pos = (row, col)
                except (IndexError, ValueError):
                    pass
            if "DOOR at" in msg:
                try:
                    coords = msg.split("DOOR at ")[1].strip("()")
                    row, col = map(int, coords.split(","))
                    self._known_door_pos = (row, col)
                except (IndexError, ValueError):
                    pass

        # First move or every ~5 turns: observe
        if self.turn_count == 1 or self.turn_count % 5 == 0:
            return ToolCall("observe", {})

        # Parse last observation to learn surroundings
        self._parse_observation(observation, r, c)

        # Standing on key? Pick it up
        if state.grid[r][c] == Cell.KEY:
            return ToolCall("pickup", {})

        # Share discoveries if haven't yet
        if self._known_key_pos and not self._shared_key_info:
            self._shared_key_info = True
            kr, kc = self._known_key_pos
            return ToolCall("send_message", {"content": f"KEY at ({kr},{kc})"})

        if self._known_door_pos and not self._shared_door_info:
            self._shared_door_info = True
            dr, dc = self._known_door_pos
            return ToolCall("send_message", {"content": f"DOOR at ({dr},{dc})"})

        # Navigate toward goal
        target = self._pick_target(agent)
        if target:
            direction = self._direction_toward(r, c, target, state)
            if direction:
                return ToolCall("move", {"direction": direction})

        # Random exploration: pick a non-wall direction
        return self._random_move(r, c, state)

    def _parse_observation(self, obs: str, r: int, c: int) -> None:
        """Extract key/door positions from observation text."""
        direction_offsets = {
            "NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1),
        }
        for part in obs.split(","):
            part = part.strip()
            for dir_name, (dr, dc) in direction_offsets.items():
                if part.startswith(dir_name + ":"):
                    val = part.split(":")[1].strip()
                    nr, nc = r + dr, c + dc
                    if val.startswith(Cell.KEY.value):
                        self._known_key_pos = (nr, nc)
                    elif val.startswith(Cell.DOOR.value):
                        self._known_door_pos = (nr, nc)

    def _pick_target(self, agent: Any) -> tuple[int, int] | None:
        """Decide where to go."""
        if agent.has_key and self._known_door_pos:
            return self._known_door_pos
        if not agent.has_key and self._known_key_pos:
            # Check if key still exists (might have been picked up)
            return self._known_key_pos
        if self._known_door_pos:
            return self._known_door_pos
        return None

    def _direction_toward(
        self, r: int, c: int, target: tuple[int, int], state: GameState,
    ) -> str | None:
        """Pick the best cardinal direction toward target, avoiding walls."""
        tr, tc = target
        candidates: list[tuple[str, int]] = []
        directions = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}
        for name, (dr, dc) in directions.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and state.grid[nr][nc] != Cell.WALL:
                dist = abs(nr - tr) + abs(nc - tc)
                candidates.append((name, dist))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        return None

    def _random_move(self, r: int, c: int, state: GameState) -> ToolCall:
        directions = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}
        valid = []
        for name, (dr, dc) in directions.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and state.grid[nr][nc] != Cell.WALL:
                valid.append(name)
        if valid:
            return ToolCall("move", {"direction": self.rng.choice(valid)})
        # Fully stuck — just try north (will drift)
        return ToolCall("move", {"direction": "NORTH"})
