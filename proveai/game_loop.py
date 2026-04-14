"""Central game loop.

Maintains an immutable GameState that is replaced each turn. Orchestrates:
  - Agent turn order
  - Message delivery
  - Tool execution
  - Win/loss condition checks
  - Legibility updates (behavioral state inference, event emission)
  - Observability (Langfuse tracing + structured JSONL records)
"""

from __future__ import annotations

from .agent import BaseAgent
from .events import Event, EventBus, EventType
from .legibility import format_turn_summary, update_behavioral_state
from .observability import (
    SpanHandle,
    StepRecordLogger,
    TracerFacade,
    build_step_record,
    generate_run_id,
)
from .state import Cell, GameState
from .tools import execute_tool


def check_win_condition(state: GameState) -> tuple[bool, str]:
    """Check if both agents are at the door and one has the key.

    Returns (is_won, reason).
    """
    door_positions: list[tuple[int, int]] = []
    for r in range(len(state.grid)):
        for c in range(len(state.grid[0])):
            if state.grid[r][c] == Cell.DOOR:
                door_positions.append((r, c))

    if not door_positions:
        return False, ""

    door_pos = door_positions[0]

    agents_at_door = [
        aid for aid, a in state.agents.items() if a.position == door_pos
    ]
    any_has_key = any(state.agents[aid].has_key for aid in agents_at_door)

    if len(agents_at_door) == len(state.agents) and any_has_key:
        return True, "Both agents at door, key held — dungeon solved!"

    return False, ""


def run_game(
    initial_state: GameState,
    agents: dict[str, BaseAgent],
    bus: EventBus,
    verbose: bool = True,
    tracer: TracerFacade | None = None,
    run_id: str | None = None,
    record_logger: StepRecordLogger | None = None,
) -> GameState:
    """Run the game loop until win, loss, or max turns.

    Returns the final GameState.
    """
    # --- Observability setup ---
    if run_id is None:
        run_id = generate_run_id()

    if tracer is None:
        tracer = TracerFacade(enabled=False)

    tracer.start_game_trace(run_id, {
        "max_turns": initial_state.max_turns,
        "grid_size": f"{len(initial_state.grid)}x{len(initial_state.grid[0])}",
        "agent_ids": sorted(initial_state.agents.keys()),
    })

    jsonl_path = None
    if record_logger is not None:
        jsonl_path = record_logger.open(run_id)

    state = initial_state
    last_result_text: dict[str, str] = {aid: "" for aid in state.agents}

    while not state.game_over and state.turn < state.max_turns:
        agent_id = state.current_agent_id
        agent_ctrl = agents[agent_id]
        agent_state = state.agents[agent_id]

        # --- Open turn span ---
        turn_span = tracer.start_turn_span(state.turn, agent_id, {
            "position": agent_state.position,
            "has_key": agent_state.has_key,
            "behavioral_state": agent_state.behavioral_state.value,
        })

        # --- Deliver pending messages ---
        state, delivered = state.pop_messages_for(agent_id)
        message_texts = [f"[From {m.sender} on turn {m.turn}]: {m.content}" for m in delivered]

        for m in delivered:
            bus.emit(Event(
                EventType.MESSAGE_DELIVERED,
                state.turn,
                agent_id,
                {"sender": m.sender, "content": m.content},
            ))

        # --- Build observation context (includes last tool output) ---
        r, c = agent_state.position
        observation = (
            f"You are {agent_id} at ({r},{c}). "
            f"Has key: {agent_state.has_key}. "
            f"Turn: {state.turn}. "
            f"Last result: {last_result_text.get(agent_id, '')}"
        )

        # --- Agent decides a tool call (LLM generation span) ---
        decide_span = tracer.start_decide_span(turn_span, agent_id, {
            "observation": observation,
            "messages": message_texts,
        })
        tool_call = agent_ctrl.decide(state, observation, message_texts)
        tracer.end_decide_span(decide_span, {
            "tool": tool_call.tool_name,
            "args": tool_call.tool_args,
        }, metrics=tool_call.metrics)

        # --- Execute the tool (tool span) ---
        tool_span = tracer.start_tool_span(turn_span, tool_call.tool_name, tool_call.tool_args)
        result = execute_tool(state, agent_id, tool_call.tool_name, tool_call.tool_args, bus)
        state = result.state
        last_result_text[agent_id] = result.text
        tracer.end_tool_span(tool_span, {
            "success": result.success,
            "text": result.text,
        })

        # --- Update behavioral state (legibility span) ---
        old_bs = state.agents[agent_id].behavioral_state.value
        behav_span = tracer.start_behavioral_span(turn_span)
        state = update_behavioral_state(state, agent_id, bus)
        new_bs = state.agents[agent_id].behavioral_state.value
        tracer.end_behavioral_span(behav_span, old_bs, new_bs)

        # --- Log structured step record ---
        record = build_step_record(
            run_id=run_id,
            trace_id=tracer.trace_id,
            turn=state.turn,
            agent_id=agent_id,
            agent_state=state.agents[agent_id],
            game_state=state,
            tool_call=tool_call,
            tool_result=result,
            messages=message_texts,
        )
        tracer.log_step_record(turn_span, record)
        if record_logger is not None:
            record_logger.log(record)

        # --- Close turn span ---
        turn_span.end()

        # --- Check win condition ---
        won, reason = check_win_condition(state)
        if won:
            bus.emit(Event(EventType.GAME_OVER, state.turn, agent_id, {"result": "WIN", "reason": reason}))
            state = state.set_win()

        # --- Verbose output ---
        if verbose:
            bs = state.agents[agent_id].behavioral_state.value
            print(format_turn_summary(state.turn, bus))
            print(f"  BEHAV   [{agent_id}] {bs}")
            print()

        # --- Advance turn ---
        state = state.next_turn()

    # --- Game over ---
    game_result = "WIN" if state.win else "LOSS"
    if not state.game_over:
        bus.emit(Event(EventType.GAME_OVER, state.turn, "system", {"result": "LOSS", "reason": "Max turns reached"}))
        state = state.set_loss("Max turns reached")
        if verbose:
            print(f"GAME OVER — max turns ({state.max_turns}) reached without solving the dungeon.")

    tracer.log_game_over({"result": game_result, "total_turns": state.turn, "win": state.win})
    tracer.end_game_trace({"result": game_result, "total_turns": state.turn})
    tracer.flush()

    if record_logger is not None:
        record_logger.close()
        if verbose and jsonl_path:
            print(f"Step records written to: {jsonl_path}")

    return state
