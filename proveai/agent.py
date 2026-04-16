"""Agent interface, mock heuristic, and real LLM implementation.

The MockAgent uses simple heuristics to explore, pick up the key, navigate
to the door, and communicate. The LLMAgent wraps the Groq API (Qwen3 32B)
with OpenAI-compatible tool calling.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
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
    # Raw LLM output (empty for MockAgent).
    raw_content: str = ""
    raw_tool_calls_json: str = ""


class BaseAgent(ABC):
    """Interface that both mock and real LLM agents implement."""

    model_name: str = "mock"

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def decide(
        self,
        state: GameState,
        observation: str,
        messages: list[str],
        prev_tool_output: str | None = None,
    ) -> ToolCall:
        """Given current context, return a single tool call for this turn.

        `prev_tool_output` is the result of the tool this agent invoked on its
        previous turn (observe/move/pickup/send_message), delivered here with
        one-turn latency. It is None on the first turn.
        """
        ...


class MockAgent(BaseAgent):
    """Heuristic-based mock that exercises the full tool set.

    Strategy:
      1. Observe on the first turn or when surroundings are unknown.
      2. If standing on the key, pick it up.
      3. If key/door location is known, navigate toward it.
      4. Otherwise, move randomly (avoiding obstacles).
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
        prev_tool_output: str | None = None,
    ) -> ToolCall:
        self.turn_count += 1
        # MockAgent parses previous observe results if any
        if prev_tool_output:
            self._parse_observation(prev_tool_output, *state.agents[self.agent_id].position)
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

        # Random exploration: pick a non-obstacle direction
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
        """Pick the best cardinal direction toward target, avoiding obstacles."""
        tr, tc = target
        candidates: list[tuple[str, int]] = []
        directions = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}
        for name, (dr, dc) in directions.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and state.grid[nr][nc] != Cell.OBSTACLE:
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
            if 0 <= nr < 8 and 0 <= nc < 8 and state.grid[nr][nc] != Cell.OBSTACLE:
                valid.append(name)
        if valid:
            return ToolCall("move", {"direction": self.rng.choice(valid)})
        # Fully stuck — just try north (will drift)
        return ToolCall("move", {"direction": "NORTH"})


# ---------------------------------------------------------------------------
# LLM agent (Groq + Qwen3 32B, OpenAI-compatible tool calling)
# ---------------------------------------------------------------------------

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "qwen/qwen3-32b"

TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": (
                "Move one cell in a cardinal direction. Fails if blocked by an "
                "obstacle or the grid edge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["NORTH", "SOUTH", "EAST", "WEST"],
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": (
                "Look at the four adjacent cells and the current cell. Result "
                "is delivered on your NEXT turn."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pickup",
            "description": "Pick up the key if you are standing on it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a short message to the other agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are {agent_id} in a cooperative 8x8 dungeon with a partner agent.

Goal: BOTH agents must end their turn on the single DOOR cell, and at least one
of you must be carrying the KEY.

Cells: '.' empty, 'K' key, 'D' door, '#' obstacle (impassable). Grid edges are
impassable too. Coordinates are (row, col) with (0,0) at the top-left.

You act one tool per turn. Available tools: move, observe, pickup, send_message.

IMPORTANT: You are STATELESS. You have no memory of prior turns. Every turn you
are given only: your position, whether you have the key, what cell you are
standing on, any message from your partner, and the result of your previous
tool call (if any). Plan each turn from this snapshot alone — you cannot
recall past observations.

Because memory is not retained, share every useful discovery with your partner
immediately via send_message using plain text like "KEY at (3,5)" or
"DOOR at (6,2)". The partner message you receive IS your only way to know
about cells you haven't just observed.

Always respond with exactly one tool call. Do not output prose."""


class LLMAgent(BaseAgent):
    """Groq-backed LLM agent. STATELESS: every decide() call sends a fresh
    [system, user] pair — no chat history, no tool_call_id threading. The
    "memory" the model gets per turn is exactly what the user message contains:
    physical state, current cell, partner message, and previous tool result.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str | None = None,
        model: str = GROQ_DEFAULT_MODEL,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(agent_id)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "LLMAgent requires GROQ_API_KEY (env var or api_key param)."
            )
        self.model = model
        self.model_name = model
        self.temperature = temperature
        self.system_prompt = SYSTEM_PROMPT.format(agent_id=agent_id)

    def decide(
        self,
        state: GameState,
        observation: str,
        messages: list[str],
        prev_tool_output: str | None = None,
    ) -> ToolCall:
        # Build a single, self-contained user prompt for this turn.
        parts = [observation]
        if prev_tool_output:
            parts.append(f"Previous tool result: {prev_tool_output}")
        if messages:
            parts.append("Messages from partner:\n" + "\n".join(messages))
        else:
            parts.append("Messages from partner: (none)")
        user_content = "\n\n".join(parts)

        request_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        metrics = DecisionMetrics()
        t0 = time.perf_counter()
        response = self._call_groq(request_messages)
        metrics.llm_latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = response.get("usage") or {}
        metrics.prompt_tokens = int(usage.get("prompt_tokens", 0))
        metrics.completion_tokens = int(usage.get("completion_tokens", 0))
        metrics.total_tokens = int(usage.get("total_tokens", 0))

        choices = response.get("choices") or []
        if not choices:
            return ToolCall("observe", {"_fallback_reason": "empty response"}, metrics=metrics)

        msg = choices[0].get("message") or {}
        raw_content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        raw_tc_json = json.dumps(tool_calls)

        if not tool_calls:
            return ToolCall(
                "observe", {"_fallback_reason": "no tool_calls"},
                metrics=metrics, raw_content=raw_content, raw_tool_calls_json=raw_tc_json,
            )

        fn = tool_calls[0].get("function") or {}
        name = fn.get("name") or "observe"
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {}

        return ToolCall(
            tool_name=name, tool_args=args, metrics=metrics,
            raw_content=raw_content, raw_tool_calls_json=raw_tc_json,
        )

    def _call_groq(self, request_messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": request_messages,
            "tools": TOOL_SCHEMA,
            "tool_choice": "required",
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            GROQ_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Groq sits behind Cloudflare and 403s the default Python UA.
                "User-Agent": "proveai/0.1 (+https://github.com/proveai)",
                "Accept": "application/json",
            },
            method="POST",
        )
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and attempt < max_attempts - 1:
                    # Honor the "try again in Xs" hint if present, else backoff.
                    m = re.search(r"try again in ([\d.]+)s", err_body)
                    wait = float(m.group(1)) + 0.5 if m else (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Groq HTTP {e.code}: {err_body}") from e
        raise RuntimeError("Groq: exhausted retries")
