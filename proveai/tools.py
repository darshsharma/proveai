"""Agent tools — move, observe, pickup, send_message.

Each tool:
  1. Computes an IntentPayload (what the agent expects).
  2. Validates against current state.
  3. Produces a new GameState (or returns the old one on failure).
  4. Computes a ResolutionPayload (what actually happened).
  5. Emits both to the EventBus (drift auto-detected).
  6. Returns (new_state, result_text) so the agent sees the outcome.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .dungeon import SIZE
from .events import EventBus, IntentPayload, ResolutionPayload
from .state import Cell, GameState, Message

DIRECTIONS: dict[str, tuple[int, int]] = {
    "NORTH": (-1, 0),
    "SOUTH": (1, 0),
    "EAST": (0, 1),
    "WEST": (0, -1),
}


# ---------------------------------------------------------------------------
# Tool result wrapper
# ---------------------------------------------------------------------------

class ToolResult:
    __slots__ = ("state", "text", "success")

    def __init__(self, state: GameState, text: str, success: bool) -> None:
        self.state = state
        self.text = text
        self.success = success


# ---------------------------------------------------------------------------
# MOVE
# ---------------------------------------------------------------------------

def tool_move(
    state: GameState,
    agent_id: str,
    direction: str,
    bus: EventBus,
) -> ToolResult:
    direction = direction.upper()
    agent = state.agents[agent_id]
    r, c = agent.position

    if direction not in DIRECTIONS:
        intent = IntentPayload(agent_id, "move", f"Move {direction} (invalid direction)", raw_args={"direction": direction})
        resolution = ResolutionPayload(agent_id, "move", f"Invalid direction '{direction}'", actual_position=agent.position, success=False, reason="INVALID_DIRECTION")
        bus.emit_tool_call(state.turn, intent, resolution)
        new_state = state.with_agent(agent_id, last_tool="move", consecutive_drift_count=agent.consecutive_drift_count + 1)
        return ToolResult(new_state, resolution.description, False)

    dr, dc = DIRECTIONS[direction]
    nr, nc = r + dr, c + dc

    intent = IntentPayload(
        agent_id, "move",
        f"Moving {direction} from ({r},{c}) to ({nr},{nc})",
        expected_position=(nr, nc),
        raw_args={"direction": direction},
    )

    # Bounds check
    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
        resolution = ResolutionPayload(agent_id, "move", f"Agent stays at ({r},{c}) — out of bounds", actual_position=(r, c), success=False, reason="OUT_OF_BOUNDS")
        bus.emit_tool_call(state.turn, intent, resolution)
        new_state = state.with_agent(agent_id, last_tool="move", consecutive_drift_count=agent.consecutive_drift_count + 1)
        return ToolResult(new_state, resolution.description, False)

    # Wall check
    target_cell = state.grid[nr][nc]
    if target_cell == Cell.WALL:
        resolution = ResolutionPayload(agent_id, "move", f"Agent stays at ({r},{c}) — wall at ({nr},{nc})", actual_position=(r, c), success=False, reason="WALL_BLOCKED")
        bus.emit_tool_call(state.turn, intent, resolution)
        new_state = state.with_agent(agent_id, last_tool="move", consecutive_drift_count=agent.consecutive_drift_count + 1)
        return ToolResult(new_state, resolution.description, False)

    # Trap — agent can enter but takes a penalty (loses next turn concept; for now just a warning)
    trap_warning = ""
    if target_cell == Cell.TRAP:
        trap_warning = " [TRAP! Lost orientation — next observe may be unreliable.]"

    # Successful move
    recent = (agent.recent_positions + ((r, c),))[-5:]  # keep last 5
    resolution = ResolutionPayload(agent_id, "move", f"Moved {direction} to ({nr},{nc}){trap_warning}", actual_position=(nr, nc), success=True)
    bus.emit_tool_call(state.turn, intent, resolution)

    new_state = state.with_agent(
        agent_id,
        position=(nr, nc),
        last_tool="move",
        consecutive_drift_count=0,
        recent_positions=recent,
    )
    return ToolResult(new_state, resolution.description, True)


# ---------------------------------------------------------------------------
# OBSERVE
# ---------------------------------------------------------------------------

def tool_observe(
    state: GameState,
    agent_id: str,
    bus: EventBus,
) -> ToolResult:
    agent = state.agents[agent_id]
    r, c = agent.position

    intent = IntentPayload(agent_id, "observe", f"Observing adjacent cells from ({r},{c})")

    surroundings: dict[str, str] = {}
    new_known = set(agent.known_positions)
    new_known.add((r, c))
    knows_key = agent.knows_key_location
    knows_door = agent.knows_door_location

    for dir_name, (dr, dc) in DIRECTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            cell = state.grid[nr][nc]
            # Check if another agent is there
            other_agent_here = any(
                a.position == (nr, nc) for aid, a in state.agents.items() if aid != agent_id
            )
            label = cell.value
            if other_agent_here:
                label += "(agent)"
            surroundings[dir_name] = label
            new_known.add((nr, nc))
            if cell == Cell.KEY:
                knows_key = True
            if cell == Cell.DOOR:
                knows_door = True
        else:
            surroundings[dir_name] = "EDGE"

    # Current cell
    current_cell = state.grid[r][c]
    surroundings["HERE"] = current_cell.value

    description = ", ".join(f"{k}: {v}" for k, v in sorted(surroundings.items()))
    resolution = ResolutionPayload(agent_id, "observe", description, actual_position=(r, c), success=True)
    bus.emit_tool_call(state.turn, intent, resolution)

    new_state = state.with_agent(
        agent_id,
        last_tool="observe",
        known_positions=frozenset(new_known),
        knows_key_location=knows_key,
        knows_door_location=knows_door,
    )
    return ToolResult(new_state, f"Position ({r},{c}): {description}", True)


# ---------------------------------------------------------------------------
# PICKUP
# ---------------------------------------------------------------------------

def tool_pickup(
    state: GameState,
    agent_id: str,
    bus: EventBus,
) -> ToolResult:
    agent = state.agents[agent_id]
    r, c = agent.position
    cell = state.grid[r][c]

    intent = IntentPayload(agent_id, "pickup", f"Picking up item at ({r},{c})", expected_pickup=cell.value)

    if cell != Cell.KEY:
        resolution = ResolutionPayload(agent_id, "pickup", f"Nothing to pick up at ({r},{c}) — cell is '{cell.value}'", success=False, reason="NO_ITEM")
        bus.emit_tool_call(state.turn, intent, resolution)
        new_state = state.with_agent(agent_id, last_tool="pickup", consecutive_drift_count=agent.consecutive_drift_count + 1)
        return ToolResult(new_state, resolution.description, False)

    # Remove key from grid
    grid_list = [list(row) for row in state.grid]
    grid_list[r][c] = Cell.EMPTY
    new_grid = tuple(tuple(row) for row in grid_list)

    resolution = ResolutionPayload(agent_id, "pickup", f"Picked up KEY at ({r},{c})", success=True)
    bus.emit_tool_call(state.turn, intent, resolution)

    new_state = replace(state, grid=new_grid)
    new_state = new_state.with_agent(agent_id, has_key=True, last_tool="pickup", consecutive_drift_count=0)
    return ToolResult(new_state, resolution.description, True)


# ---------------------------------------------------------------------------
# SEND_MESSAGE
# ---------------------------------------------------------------------------

def tool_send_message(
    state: GameState,
    agent_id: str,
    content: str,
    bus: EventBus,
) -> ToolResult:
    agent = state.agents[agent_id]
    other_ids = [aid for aid in state.agents if aid != agent_id]
    recipient = other_ids[0]  # 2-agent game

    intent = IntentPayload(agent_id, "send_message", f"Sending message to {recipient}: '{content}'")

    msg = Message(sender=agent_id, recipient=recipient, content=content, turn=state.turn)
    new_state = state.add_message(msg)

    resolution = ResolutionPayload(agent_id, "send_message", f"Message queued for {recipient}", success=True)
    bus.emit_tool_call(state.turn, intent, resolution)

    from .events import Event, EventType
    bus.emit(Event(EventType.MESSAGE_SENT, state.turn, agent_id, {"recipient": recipient, "content": content}))

    new_state = new_state.with_agent(agent_id, last_tool="send_message")
    return ToolResult(new_state, f"Message sent to {recipient}.", True)


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "move": tool_move,
    "observe": tool_observe,
    "pickup": tool_pickup,
    "send_message": tool_send_message,
}


def execute_tool(
    state: GameState,
    agent_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    bus: EventBus,
) -> ToolResult:
    """Dispatch a tool call. Returns ToolResult with new state and output text."""
    if tool_name not in TOOL_REGISTRY:
        intent = IntentPayload(agent_id, tool_name, f"Unknown tool '{tool_name}'")
        resolution = ResolutionPayload(agent_id, tool_name, f"Tool '{tool_name}' does not exist", success=False, reason="UNKNOWN_TOOL")
        bus.emit_tool_call(state.turn, intent, resolution)
        new_state = state.with_agent(agent_id, last_tool=tool_name)
        return ToolResult(new_state, resolution.description, False)

    fn = TOOL_REGISTRY[tool_name]
    if tool_name == "move":
        return fn(state, agent_id, tool_args.get("direction", ""), bus)
    elif tool_name == "observe":
        return fn(state, agent_id, bus)
    elif tool_name == "pickup":
        return fn(state, agent_id, bus)
    elif tool_name == "send_message":
        return fn(state, agent_id, tool_args.get("content", ""), bus)
    else:
        raise ValueError(f"Unhandled tool: {tool_name}")
